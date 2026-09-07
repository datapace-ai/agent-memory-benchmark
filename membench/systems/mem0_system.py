"""Mem0 open source as a memory system.

Ingestion: one `add` call per session with the session's messages, scoped by
user_id. Retrieval: `search` scoped by a user_id filter, top_k memories.
The product's own extraction model is a LangChain ChatOllama with reasoning
off, passed through Mem0's `langchain` provider, because Mem0's native Ollama
provider has no way to disable thinking.

Reference: mem0ai/memory-benchmarks LongMemEval runner.
"""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from membench.config import ModelConfig
from membench.data.types import Session
from membench.llm import LLMClient
from membench.systems.base import Answer, IngestStats, MemorySystem
from membench.systems.shared import (
    answer_from_context,
    chat_model,
    embeddings,
    session_messages,
)


def _safe(namespace: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", namespace)


def default_memory_factory(cfg: ModelConfig, seed: int, store_dir: Path) -> Callable[[str], object]:
    """One on-disk store per namespace. Qdrant's local mode holds a file lock,
    so concurrent units must never share a path."""

    def factory(namespace: str):
        from mem0 import Memory

        path = store_dir / _safe(namespace)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        return Memory.from_config(
            {
                "version": "v1.1",
                "llm": {"provider": "langchain", "config": {"model": chat_model(cfg, seed)}},
                "embedder": {
                    "provider": "langchain",
                    "config": {"model": embeddings(cfg), "embedding_dims": cfg.embed_dims},
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "path": str(path / "qdrant"),
                        "collection_name": "membench",
                        "embedding_model_dims": cfg.embed_dims,
                        "on_disk": True,
                    },
                },
                "history_db_path": str(path / "history.db"),
            }
        )

    return factory


def format_results(results: list[dict]) -> str:
    lines = [f"- {r.get('memory', '')}" for r in results if r.get("memory")]
    return "\n".join(lines) if lines else "(no memories retrieved)"


class Mem0System(MemorySystem):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        cfg: ModelConfig,
        seed: int,
        store_dir: Path,
        top_k: int = 10,
        memory_factory: Callable[[str], object] | None = None,
    ) -> None:
        self.name = name
        self._llm = llm
        self._seed = seed
        self._top_k = top_k
        self._factory = memory_factory or default_memory_factory(cfg, seed, store_dir)
        self._memory = None
        self._user: str | None = None
        self._sessions = 0

    def _mem(self):
        if self._memory is None:
            raise RuntimeError("reset must be called before use")
        return self._memory

    def reset(self, namespace: str) -> None:
        self._close_previous()
        self._user = namespace
        self._sessions = 0
        self._memory = self._factory(namespace)
        self._memory.delete_all(user_id=namespace)

    def _close_previous(self) -> None:
        """Release the previous instance's local Qdrant locks before opening another."""
        memory = self._memory
        self._memory = None
        if memory is None:
            return
        for attr in ("vector_store", "_migrations_store"):
            store = getattr(memory, attr, None)
            client = getattr(store, "client", None)
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def ingest(self, session: Session) -> IngestStats:
        if self._user is None:
            raise RuntimeError("reset must be called before ingest")
        started = time.perf_counter()
        self._mem().add(session_messages(session), user_id=self._user)
        self._sessions += 1
        stored = self._mem().get_all(filters={"user_id": self._user}).get("results", [])
        return IngestStats(
            seconds=time.perf_counter() - started,
            store_items=len(stored),
            store_tokens=sum(len(r.get("memory", "")) for r in stored) // 4,
        )

    def answer(self, question: str, question_date: str) -> Answer:
        if self._user is None:
            raise RuntimeError("reset must be called before answer")
        started = time.perf_counter()
        found = self._mem().search(question, filters={"user_id": self._user}, top_k=self._top_k)
        context = format_results(found.get("results", []))
        retrieval_seconds = time.perf_counter() - started
        return answer_from_context(
            self._llm, context, question, question_date, self._seed, retrieval_seconds, started
        )
