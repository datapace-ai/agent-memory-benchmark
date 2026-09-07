"""Maps a SystemConfig to a live MemorySystem."""

from __future__ import annotations

from membench.config import REPO_ROOT, ModelConfig, SystemConfig
from membench.llm import LLMClient
from membench.systems.base import MemorySystem
from membench.systems.context_window import ContextWindowSystem

STORE_ROOT = REPO_ROOT / "data" / "vendor-stores"
VENDOR_KINDS = ("mem0", "langmem", "cognee", "graphiti")


def build(
    cfg: SystemConfig, llm: LLMClient, seed: int = 0, models: ModelConfig | None = None
) -> MemorySystem:
    if cfg.kind == "context_window":
        return ContextWindowSystem(
            name=cfg.name, llm=llm, token_budget=int(cfg.params["token_budget"]), seed=seed
        )
    if cfg.kind == "file_search":
        from membench.systems.file_search import FileSearchSystem

        return FileSearchSystem(
            cfg.name, llm, STORE_ROOT / cfg.name / f"seed-{seed}", seed,
            max_tool_calls=int(cfg.params.get("max_tool_calls", 8)),
        )
    if cfg.kind in VENDOR_KINDS:
        if models is None:
            raise ValueError(f"system kind {cfg.kind} needs the models config")
        top_k = int(cfg.params.get("top_k", 10))
        store_dir = STORE_ROOT / cfg.name / f"seed-{seed}"
        if cfg.kind == "mem0":
            from membench.systems.mem0_system import Mem0System

            return Mem0System(cfg.name, llm, models, seed, store_dir, top_k=top_k)
        if cfg.kind == "langmem":
            from membench.systems.langmem_system import LangMemSystem

            return LangMemSystem(cfg.name, llm, models, seed, top_k=top_k)
        if cfg.kind == "graphiti":
            from membench.systems.graphiti_system import GraphitiSystem

            return GraphitiSystem(cfg.name, llm, models, seed, store_dir, top_k=top_k)
        from membench.systems.cognee_system import CogneeSystem

        return CogneeSystem(cfg.name, llm, models, seed, store_dir, top_k=top_k)
    raise ValueError(f"Unknown system kind: {cfg.kind}")


def evidence_only(cfg: SystemConfig) -> bool:
    """True for the oracle condition, which is fed only the evidence sessions."""
    return bool(cfg.params.get("evidence_only", False))
