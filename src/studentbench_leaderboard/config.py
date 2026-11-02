"""Connection profile loading. Credentials are always read from the environment."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

DEFAULT_PROFILES = {
    "auto": {"transport": "auto"},
    "openrouter": {"transport": "openrouter"},
    "direct": {"transport": "direct"},
    "openai": {"transport": "openai"},
    "anthropic": {"transport": "anthropic"},
    "tailscale": {"transport": "litellm_proxy", "api_base_env": "LITELLM_PROXY_API_BASE", "api_key_env": "LITELLM_PROXY_API_KEY"},
}


def load_profiles(config_path: str | Path | None = None) -> tuple[dict, str]:
    profiles = {name: dict(value) for name, value in DEFAULT_PROFILES.items()}
    default_profile = "auto"
    if config_path is not None:
        with open(config_path, "rb") as file:
            config = tomllib.load(file)
        default_profile = config.get("default_profile", "auto")
        if not isinstance(default_profile, str) or not default_profile.strip():
            raise ValueError("default_profile must be a nonempty string")
        configured = config.get("profiles", {})
        if not isinstance(configured, dict):
            raise ValueError("profiles must be a TOML table")
        for name, value in configured.items():
            if not isinstance(value, dict):
                raise ValueError(f"Profile {name!r} must be a table")
            unknown = set(value) - {"transport", "api_base_env", "api_key_env", "models", "revision"}
            if unknown:
                raise ValueError(f"Unknown profile settings: {', '.join(sorted(unknown))}")
            for key in ("transport", "api_base_env", "api_key_env", "revision"):
                if key in value and (not isinstance(value[key], str) or not value[key].strip()
                                     or value[key] != value[key].strip()):
                    raise ValueError(f"Profile {name!r} {key} must be a nonempty string")
            models = value.get("models", {})
            if not isinstance(models, dict) or any(
                not isinstance(k, str) or not k.strip() or k != k.strip() or
                not isinstance(v, str) or not v.strip() or v != v.strip()
                for k, v in models.items()
            ):
                raise ValueError(f"Profile {name!r} models must map nonempty model strings to nonempty aliases")
            profiles[name] = {**profiles.get(name, {}), **value}
    return profiles, default_profile


def env_value(name: str | None) -> str | None:
    return os.environ.get(name) if name else None
