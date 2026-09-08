"""LangMem as a memory system.

Ingestion: the background memory store manager is invoked with the session's
messages, which extracts and consolidates memories into a LangGraph store
indexed by local embeddings. Retrieval: the manager's search over the same
namespace. Reset builds a fresh in-memory store, the cleanest empty state.

Reference: LangMem documentation, create_memory_store_manager.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from membench.config import ModelConfig
from membench.data.types import Session
from membench.llm import LLMClient
from membench.systems.base import Answer, IngestStats, MemorySystem
from membench.systems.shared import (
    answer_from_context,
    chat_model,
    embeddings,
    retry_transient,
    session_messages,
)

NAMESPACE = ("memories", "{langgraph_user_id}")


def default_store_factory(cfg: ModelConfig) -> Callable[[], object]:
    def factory():
        from langgraph.store.memory import InMemoryStore

        return InMemoryStore(index={"dims": cfg.embed_dims, "embed": embeddings(cfg)})

    return factory


def default_manager_factory(cfg: ModelConfig, seed: int, top_k: int) -> Callable[[object], object]:
    def factory(store):
        from langmem import create_memory_store_manager

        return create_memory_store_manager(
            chat_model(cfg, seed),
            namespace=NAMESPACE,
            store=store,
            query_limit=top_k,
            enable_inserts=True,
            enable_deletes=True,
        )

    return factory


def _content(item) -> str:
    if isinstance(item, str):
        return item
    value = getattr(item, "value", None)
    if value is None and isinstance(item, dict):
        value = item.get("value", item)
    if isinstance(value, dict):
        return str(value.get("content", value))
    return str(value)


def format_memories(items: list) -> str:
    lines = [f"- {_content(i)}" for i in items if _content(i)]
    return "\n".join(lines) if lines else "(no memories retrieved)"


class LangMemSystem(MemorySystem):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        cfg: ModelConfig,
        seed: int,
        top_k: int = 10,
        store_factory: Callable[[], object] | None = None,
        manager_factory: Callable[[object], object] | None = None,
    ) -> None:
        self.name = name
        self._llm = llm
        self._seed = seed
        self._store_factory = store_factory or default_store_factory(cfg)
        self._manager_factory = manager_factory or default_manager_factory(cfg, seed, top_k)
        self._manager = None
        self._config: dict | None = None
        self._items = 0
        self._chars = 0

    def reset(self, namespace: str) -> None:
        store = self._store_factory()
        self._manager = self._manager_factory(store)
        self._config = {"configurable": {"langgraph_user_id": namespace}}
        self._items = 0
        self._chars = 0

    def _require(self):
        if self._manager is None or self._config is None:
            raise RuntimeError("reset must be called before use")
        return self._manager, self._config

    def ingest(self, session: Session) -> IngestStats:
        manager, config = self._require()
        started = time.perf_counter()
        retry_transient(
            lambda: manager.invoke({"messages": session_messages(session)}, config=config),
            what=f"langmem invoke {session.session_id}",
        )
        probe = session.turns[0].content if session.turns else ""
        stored = manager.search(query=probe, config=config)
        self._items = len(stored)
        self._chars = sum(len(_content(i)) for i in stored)
        return IngestStats(
            seconds=time.perf_counter() - started,
            store_items=self._items,
            store_tokens=self._chars // 4,
        )

    def answer(self, question: str, question_date: str) -> Answer:
        manager, config = self._require()
        started = time.perf_counter()
        found = retry_transient(lambda: manager.search(query=question, config=config), what="langmem search")
        context = format_memories(list(found))
        retrieval_seconds = time.perf_counter() - started
        return answer_from_context(
            self._llm, context, question, question_date, self._seed, retrieval_seconds, started
        )
