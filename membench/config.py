"""Typed configuration loaded from the YAML files in configs/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    answer_model: str
    judge_model: str
    embed_model: str
    num_ctx: int
    temperature: float
    max_answer_tokens: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class SystemConfig:
    name: str
    kind: str
    params: dict


def load_models(path: Path) -> ModelConfig:
    raw = yaml.safe_load(path.read_text())
    return ModelConfig(
        base_url=raw["base_url"].rstrip("/"),
        answer_model=raw["answer_model"],
        judge_model=raw["judge_model"],
        embed_model=raw["embed_model"],
        num_ctx=int(raw["num_ctx"]),
        temperature=float(raw["temperature"]),
        max_answer_tokens=int(raw["max_answer_tokens"]),
        seeds=tuple(int(s) for s in raw["seeds"]),
    )


def load_systems(path: Path) -> tuple[SystemConfig, ...]:
    raw = yaml.safe_load(path.read_text())
    return tuple(
        SystemConfig(name=entry["name"], kind=entry["kind"], params=dict(entry.get("params", {})))
        for entry in raw["systems"]
    )
