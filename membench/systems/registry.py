"""Maps a SystemConfig to a live MemorySystem."""

from __future__ import annotations

from membench.config import SystemConfig
from membench.llm import LLMClient
from membench.systems.base import MemorySystem
from membench.systems.context_window import ContextWindowSystem


def build(cfg: SystemConfig, llm: LLMClient, seed: int = 0) -> MemorySystem:
    if cfg.kind == "context_window":
        return ContextWindowSystem(
            name=cfg.name, llm=llm, token_budget=int(cfg.params["token_budget"]), seed=seed
        )
    raise ValueError(f"Unknown system kind: {cfg.kind}")


def evidence_only(cfg: SystemConfig) -> bool:
    """True for the oracle condition, which is fed only the evidence sessions."""
    return bool(cfg.params.get("evidence_only", False))
