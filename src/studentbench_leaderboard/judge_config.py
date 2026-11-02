"""Load named judge definitions; TOML parsing needs no additional dependency."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from string import Template

from .data import digest

DEFAULT_JUDGES_DIR = Path(__file__).with_name("judge_configs")
ENGINE_VERSION = "pairwise-strategies.v1"
STRATEGIES = {"single", "order_check", "repeated", "panel", "balanced_soft", "panel_soft"}
SOFT_STRATEGIES = {"balanced_soft", "panel_soft"}


def fields(value: object, required: set[str], optional: set[str], location: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a TOML table")
    if missing := required - set(value):
        raise ValueError(f"Missing {location} fields: {', '.join(sorted(missing))}")
    if extra := set(value) - required - optional:
        raise ValueError(f"Unknown {location} fields: {', '.join(sorted(extra))}")
    return value


def nonempty(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a nonempty string")
    return value


def prompt_file(directory: Path, name: object) -> str:
    path = (directory / nonempty(name, "prompt path")).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Prompt files must be inside the judge configuration directory")
    return nonempty(path.read_text(encoding="utf-8"), str(path))


@dataclass(frozen=True)
class Member:
    name: str
    model: str
    reasoning_effort: str | None = None

    def configuration(self) -> dict:
        return {"name": self.name, "model": self.model, "reasoning_effort": self.reasoning_effort}


@dataclass(frozen=True)
class JudgeDefinition:
    judge_id: str
    strategy: str
    members: tuple[Member, ...]
    system_prompt: str
    user_template: str
    repetitions: int = 1
    seed: int = 0
    description: str = ""
    response_style: str = "score_reason"
    plan_projection: str = "as_stored"

    @property
    def call_count(self) -> int:
        return {"single": 1, "order_check": 2, "repeated": self.repetitions,
                "panel": len(self.members), "balanced_soft": 2,
                "panel_soft": len(self.members)}[self.strategy]

    def structure(self) -> dict:
        """Changing this contract requires a new public judge ID."""
        result = {"engine": ENGINE_VERSION, "strategy": self.strategy,
                "member_slots": [m.name for m in self.members],
                "repetitions": self.repetitions, "seed": self.seed,
                "aggregation": "strict-majority-direction; conservative-median-strength",
                "abstentions": "count-in-majority-denominator", "timeout": 180}
        if self.response_style != "score_reason":
            result["response_style"] = self.response_style
        if self.strategy in SOFT_STRATEGIES:
            # Additive contract only for new strategies: old hashes must not change.
            result["preference_aggregation"] = "mean-normalized-win-share"
            result["preference_abstentions"] = "null-if-any-member-abstains"
            result["preference_output"] = "preference_probabilities"
        return result

    @property
    def version(self) -> str:
        prompt_config = {"system_prompt": self.system_prompt, "user_template": self.user_template}
        if self.plan_projection != "as_stored":
            prompt_config["plan_projection"] = self.plan_projection
        prompts = digest(prompt_config)
        models = digest([m.configuration() for m in self.members])
        return f"{self.judge_id}_{prompts[:12]}_{models[:12]}"

    def configuration(self) -> dict:
        result = {"judge_id": self.judge_id, "version": self.version,
                "structure": self.structure(), "strategy_sha256": digest(self.structure()),
                "system_prompt": self.system_prompt, "user_template": self.user_template,
                "members": [m.configuration() for m in self.members]}
        if self.plan_projection != "as_stored":
            result["plan_projection"] = self.plan_projection
        return result


def load_definition(judge_id: str, directory: Path | None = None) -> JudgeDefinition:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", judge_id):
        raise ValueError("Judge IDs must contain lowercase letters, digits, and underscores")
    directory = Path(directory) if directory is not None else DEFAULT_JUDGES_DIR
    path = directory / f"{judge_id}.toml"
    if not path.is_file():
        raise ValueError(f"Unknown judge {judge_id!r}; expected {path}")
    config = fields(tomllib.loads(path.read_text(encoding="utf-8")),
                    {"schema_version", "judge_id", "strategy", "prompts", "members"},
                    {"description", "repetitions", "seed", "response_style", "plan_projection"}, "judge")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported judge configuration schema")
    if config["judge_id"] != judge_id:
        raise ValueError("Judge ID must match its TOML filename")
    strategy = nonempty(config["strategy"], "strategy")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if strategy != "repeated" and {"repetitions", "seed"} & config.keys():
        raise ValueError("repetitions and seed apply only to the repeated strategy")
    repetitions = config.get("repetitions", 1)
    if type(repetitions) is not int or repetitions < 1 or (strategy == "repeated" and repetitions < 2):
        raise ValueError("repeated requires an integer repetitions >= 2")
    seed = config.get("seed", 0)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    prompts = fields(config["prompts"], {"system", "user"}, set(), "prompts")
    system_prompt = prompt_file(directory, prompts["system"])
    user_template = prompt_file(directory, prompts["user"])
    template = Template(user_template)
    if (not template.is_valid() or template.get_identifiers() != ["payload"]
            or sum(bool(match.group("named") or match.group("braced"))
                   for match in template.pattern.finditer(user_template)) != 1):
        raise ValueError("User template must contain exactly one $payload placeholder and no others")
    if not isinstance(config["members"], list) or not config["members"]:
        raise ValueError("members must contain at least one [[members]] table")
    members = []
    for item in config["members"]:
        item = fields(item, {"name", "model"}, {"reasoning_effort"}, "member")
        members.append(Member(nonempty(item["name"], "member name"),
                              nonempty(item["model"], "model"),
                              nonempty(item["reasoning_effort"], "reasoning_effort")
                              if "reasoning_effort" in item else None))
    if len({m.name for m in members}) != len(members):
        raise ValueError("Member names must be unique")
    if strategy not in {"panel", "panel_soft"} and len(members) != 1:
        raise ValueError(f"{strategy} requires exactly one member")
    if strategy in {"panel", "panel_soft"} and (len(members) < 2 or len({m.model for m in members}) != len(members)):
        raise ValueError("A panel requires at least two members with distinct model identifiers")
    response_style = config.get("response_style", "score_reason")
    if response_style not in {"score_reason", "evidence_winner", "absolute_ratings"}:
        raise ValueError(f"Unknown response_style: {response_style}")
    projection = config.get("plan_projection", "as_stored")
    if projection not in {"as_stored", "human_display"}:
        raise ValueError(f"Unknown plan_projection: {projection}")
    return JudgeDefinition(judge_id, strategy, tuple(members), system_prompt, user_template,
                           repetitions, seed, str(config.get("description", "")), response_style, projection)


def check_strategy_history(config: dict, records: list[dict]) -> None:
    """Prevent changing an established strategy under the same ID in a result journal."""
    for record in records:
        previous = record.get("judge", {})
        if (previous.get("judge_id") == config["judge_id"]
                and previous.get("strategy_sha256") != config["strategy_sha256"]):
            raise ValueError(f"Strategy for {config['judge_id']} changed; create a new judge ID")
