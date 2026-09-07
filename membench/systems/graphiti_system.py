"""Graphiti, Zep's open-source engine, as a memory system.

Each turn becomes an episode with the session date as a timezone-aware
reference time, in a group named after the namespace. Retrieval is
Graphiti's hybrid edge search over that group; the edge facts are the
context. Reset opens a fresh embedded FalkorDB, which is the cheapest empty
graph. The engine's model is the no-think variant through Ollama's
OpenAI-compatible endpoint, and reranking is the local BGE cross-encoder.

References: getzep/zep-papers locomo_eval; RudrenduPaul/memtrust
zep_graphiti_selfhosted_adapter (timezone-aware reference_time, search returns
EntityEdge with .fact).
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from membench.config import ModelConfig
from membench.data.types import Session
from membench.llm import LLMClient
from membench.systems.base import Answer, IngestStats, MemorySystem
from membench.systems.shared import answer_from_context, api_key

_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})")


def parse_session_date(date: str) -> datetime:
    m = _DATE.search(date)
    if not m:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
    y, mo, d, h, mi = (int(x) for x in m.groups())
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def _run(coro):
    return asyncio.run(coro)


def _group(namespace: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", namespace)


def format_edges(edges: list) -> str:
    facts = []
    for e in edges:
        fact = getattr(e, "fact", None)
        if fact is None and isinstance(e, dict):
            fact = e.get("fact")
        facts.append(fact)
    lines = [f"- {f}" for f in facts if f]
    return "\n".join(lines) if lines else "(no facts retrieved)"


class LocalEmbedder:
    """Graphiti EmbedderClient over a sentence-transformers model on the CPU."""

    def __init__(self, model_name: str, dims: int) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self._dims = dims

    async def create(self, input_data):
        vec = self._model.encode(input_data if isinstance(input_data, str) else " ".join(map(str, input_data)), normalize_embeddings=True)
        return [float(x) for x in vec[: self._dims]]

    async def create_batch(self, input_data_list):
        vecs = self._model.encode(list(input_data_list), normalize_embeddings=True)
        return [[float(x) for x in v[: self._dims]] for v in vecs]


def default_graphiti_factory(cfg: ModelConfig, store_dir: Path) -> Callable[[str], object]:
    def factory(namespace: str):
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from redislite.falkordb_client import FalkorDB

        path = store_dir / _group(namespace)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        db = FalkorDB(str(path / "falkordb.db"))
        model = cfg.openai_compat_model
        if cfg.provider == "openai_compat":
            llm_client = OpenAIGenericClient(
                LLMConfig(api_key=api_key(cfg) or "none", model=model, small_model=model, base_url=cfg.base_url)
            )
            embedder = LocalEmbedder(cfg.embed_model, cfg.embed_dims)
        else:
            llm_client = OpenAIGenericClient(
                LLMConfig(api_key="ollama", model=model, small_model=model, base_url=f"{cfg.base_url}/v1")
            )
            embedder = OpenAIEmbedder(
                OpenAIEmbedderConfig(
                    api_key="ollama",
                    embedding_model=cfg.embed_model,
                    embedding_dim=cfg.embed_dims,
                    base_url=f"{cfg.base_url}/v1",
                )
            )
        return Graphiti(
            graph_driver=FalkorDriver(falkor_db=db),
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=BGERerankerClient(),
        )

    return factory


class GraphitiSystem(MemorySystem):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        cfg: ModelConfig,
        seed: int,
        store_dir: Path,
        top_k: int = 10,
        graphiti_factory: Callable[[str], object] | None = None,
    ) -> None:
        self.name = name
        self._llm = llm
        self._seed = seed
        self._top_k = top_k
        self._factory = graphiti_factory or default_graphiti_factory(cfg, store_dir)
        self._graphiti = None
        self._group: str | None = None
        self._episodes = 0
        self._chars = 0

    def reset(self, namespace: str) -> None:
        if self._graphiti is not None:
            try:
                _run(self._graphiti.close())
            except Exception:
                pass
        self._graphiti = self._factory(namespace)
        _run(self._graphiti.build_indices_and_constraints())
        self._group = _group(namespace)
        self._episodes = 0
        self._chars = 0

    def _require(self):
        if self._graphiti is None or self._group is None:
            raise RuntimeError("reset must be called before use")
        return self._graphiti, self._group

    def ingest(self, session: Session) -> IngestStats:
        graphiti, group = self._require()
        started = time.perf_counter()
        when = parse_session_date(session.date)
        for i, turn in enumerate(session.turns):
            body = f"{turn.role}: {turn.content}"
            _run(
                graphiti.add_episode(
                    name=f"{session.session_id}-{i}",
                    episode_body=body,
                    source_description="chat message",
                    reference_time=when,
                    group_id=group,
                )
            )
            self._episodes += 1
            self._chars += len(body)
        return IngestStats(
            seconds=time.perf_counter() - started,
            store_items=self._episodes,
            store_tokens=self._chars // 4,
        )

    def answer(self, question: str, question_date: str) -> Answer:
        graphiti, group = self._require()
        started = time.perf_counter()
        edges = _run(graphiti.search(question, group_ids=[group], num_results=self._top_k))
        context = format_edges(list(edges))
        retrieval_seconds = time.perf_counter() - started
        return answer_from_context(
            self._llm, context, question, question_date, self._seed, retrieval_seconds, started
        )
