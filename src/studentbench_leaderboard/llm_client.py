"""Per-request LiteLLM routing, without global proxy configuration."""
from __future__ import annotations

import hashlib
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .config import env_value, load_profiles

_SETTINGS = {"temperature", "top_p", "max_tokens", "max_completion_tokens", "seed"}
_PROVIDER_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
_PROVIDER_BASES = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com", "openrouter": "https://openrouter.ai/api/v1"}


def _family(model: str) -> str | None:
    name = model.lower()
    if name.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return "openai"
    if name.startswith("claude"):
        return "anthropic"
    return None


def _base_identity(base: str | None) -> str | None:
    return hashlib.sha256(base.encode()).hexdigest() if base else None


class LLMRequestError(RuntimeError):
    """Sanitized provider failure with retry information retained."""

    def __init__(self, message: str, *, original_type: str, status_code: int | None, transient: bool):
        super().__init__(message)
        self.original_type = original_type
        self.status_code = status_code
        self.transient = transient


class LLMClient:
    def __init__(self, profile: str | None = None, config_path: str | None = None):
        profiles, default_profile = load_profiles(config_path)
        profile = profile or default_profile
        if profile not in profiles:
            raise ValueError(f"Unknown connection profile {profile!r}")
        self.profile = profile
        self.profile_config = profiles[profile]
        if self.profile_config.get("transport") not in {"auto", "direct", "openai", "anthropic", "openrouter", "litellm_proxy"}:
            raise ValueError(f"Unsupported transport in profile {profile!r}")

    def _route(self, model: str) -> dict:
        if not isinstance(model, str) or not model.strip() or model != model.strip():
            raise ValueError("model must be a nonempty model identifier")
        requested_model = model
        model = self.profile_config.get("models", {}).get(model, model)
        transport = self.profile_config["transport"]
        prefix, sep, remainder = model.partition("/")
        explicit = prefix in {"openai", "anthropic", "openrouter", "litellm_proxy"} and bool(sep)
        if explicit and not remainder:
            raise ValueError("Model identifier is missing after provider prefix")
        if transport == "auto" and explicit and prefix in {"openrouter", "litellm_proxy"}:
            transport = prefix
        if transport == "auto":
            family = prefix if explicit and prefix in {"openai", "anthropic"} else _family(model)
            transport = "direct" if family and env_value(_PROVIDER_KEYS[family]) else "openrouter"
        forced_provider = transport if transport in {"openai", "anthropic"} else None
        if forced_provider:
            transport = "direct"
        provider = None
        if transport == "direct":
            if explicit:
                if prefix not in {"openai", "anthropic"}:
                    raise ValueError("Direct profile requires an OpenAI or Anthropic model")
                provider, bare = prefix, remainder
            else:
                provider, bare = forced_provider or _family(model), model
            if forced_provider and provider != forced_provider:
                raise ValueError(f"Profile {self.profile!r} conflicts with model provider {provider!r}")
            if not provider:
                raise ValueError("Direct profile requires an openai/ or anthropic/ model")
            effective_model = f"{provider}/{bare}"
            base, key_env = _PROVIDER_BASES[provider], _PROVIDER_KEYS[provider]
        elif transport == "openrouter":
            if prefix == "openrouter" and sep:
                bare = remainder
            elif explicit:
                bare = model
            else:
                family = _family(model)
                bare = f"{family}/{model}" if family else model
            provider, effective_model = "openrouter", f"openrouter/{bare}"
            base, key_env = _PROVIDER_BASES[provider], _PROVIDER_KEYS[provider]
        else:
            if transport != "litellm_proxy":
                raise ValueError(f"Unsupported transport: {transport}")
            bare = remainder if prefix == "litellm_proxy" and sep else model
            provider, effective_model = "litellm_proxy", f"litellm_proxy/{bare}"
            base = env_value(self.profile_config.get("api_base_env", "LITELLM_PROXY_API_BASE"))
            key_env = self.profile_config.get("api_key_env", "LITELLM_PROXY_API_KEY")
        return {"profile": self.profile, "transport": transport, "provider": provider,
                "requested_model": requested_model, "mapped_model": model,
                "revision": self.profile_config.get("revision"), "effective_model": effective_model,
                "api_base_sha256": _base_identity(base), "_api_base": base, "_key_env": key_env}

    def configuration(self, model: str, reasoning_effort=None, **settings) -> dict:
        unknown = set(settings) - _SETTINGS
        if unknown:
            raise ValueError(f"Unsupported request settings: {chr(44).join(sorted(unknown))}")

        if reasoning_effort is not None and reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"Unsupported reasoning effort: {reasoning_effort!r}")
        route = self._route(model)
        return {**{k: v for k, v in route.items() if not k.startswith("_")},
                "reasoning_effort": reasoning_effort, "settings": dict(settings)}

    def complete(self, model: str, messages: list, schema: dict, *, reasoning_effort=None, timeout=180, **settings) -> tuple[str, dict]:
        config = self.configuration(model, reasoning_effort, **settings)
        route = self._route(model)
        key = env_value(route["_key_env"])
        if not key:
            raise ValueError(f"Missing {route['_key_env']} for {route['provider']} route")
        if not route["_api_base"]:
            raise ValueError("Missing LITELLM_PROXY_API_BASE for proxy route")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        import litellm
        # LiteLLM combines these ambient flags with per-call settings, so passing
        # False does not neutralize them. Refuse rather than change process state.
        if litellm.LiteLLMProxyChatConfig._should_use_litellm_proxy_by_default():
            raise ValueError("Disable USE_LITELLM_PROXY and litellm.use_litellm_proxy; select the tailscale profile explicitly")
        if litellm.drop_params:
            raise ValueError("Disable litellm.drop_params to preserve requested settings")
        if litellm.model_fallbacks or litellm.model_alias_map:
            raise ValueError("Disable LiteLLM global model_fallbacks and model_alias_map to preserve routing")
        completion = litellm.completion
        kwargs = dict(settings)
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
            if route["effective_model"] == "openai/gpt-6-astra":
                kwargs["allowed_openai_params"] = ["reasoning_effort"]
        try:
            response = completion(model=route["effective_model"], messages=messages, timeout=timeout,
                num_retries=0, api_base=route["_api_base"], api_key=key,
                use_litellm_proxy=False, drop_params=False,
                response_format={"type": "json_schema", "json_schema": {"name": "studentbench_response", "strict": True, "schema": schema}}, **kwargs)
        except Exception as exc:
            # Provider errors may echo authorization headers or internal endpoints.
            message = str(exc).replace(key, "[REDACTED]").replace(route["_api_base"], "[API BASE]")
            name = type(exc).__name__
            status = getattr(exc, "status_code", None)
            status = status if isinstance(status, int) else None
            transient = status in {408, 429, 500, 502, 503, 504} or name in {
                "APIConnectionError", "APITimeoutError", "Timeout", "TimeoutError", "RateLimitError"}
            raise LLMRequestError(f"{name}: {message}", original_type=name,
                                  status_code=status, transient=transient) from None
        raw = response.choices[0].message.content
        if not isinstance(raw, str):
            raise ValueError("Model returned no text (possibly a refusal)")
        usage = getattr(response, "usage", None)
        return raw, {"usage": usage.model_dump() if usage else {}, "response_model": response.model,
                     "response_id": response.id, "routing": config}

    def doctor(self, model: str | None = None) -> dict:
        model = model or "gpt-4o-mini"
        config = self.configuration(model)
        route = self._route(model)
        key = env_value(route["_key_env"])
        if not key:
            return {"ok": False, "routing": config, "error": f"Missing {route['_key_env']}"}
        base = route["_api_base"]
        if not base:
            return {"ok": False, "routing": config, "error": "Missing LITELLM_PROXY_API_BASE"}
        parts = urlsplit(base)
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
            return {"ok": False, "routing": config, "error": "Invalid API base URL"}
        url = urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip('/') + ('/v1/models' if route['provider'] == 'anthropic' else '/models'), '', ''))
        headers = {"Authorization": f"Bearer {key}"}
        if route["provider"] == "anthropic":
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        try:
            with urlopen(Request(url, headers=headers), timeout=10) as response:
                status = response.status
            return {"ok": 200 <= status < 300, "routing": config, "http_status": status}
        except HTTPError as exc:
            return {"ok": False, "routing": config, "http_status": exc.code, "error": "Authentication or models endpoint failed"}
        except (URLError, OSError, ValueError):
            return {"ok": False, "routing": config, "error": "Could not reach the models endpoint"}
