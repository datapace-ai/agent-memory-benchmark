"""Cognee as a memory system.

Cognee reads its configuration from environment variables at import time, so
the adapter sets them before the first import. Each namespace becomes a Cognee
dataset, which isolates retrieval per question. The system is pruned once when
the adapter is first used so the run starts empty. Search asks for graph
context only; the benchmark's own client writes the answer.

Reference: topoteretes/cognee README and .env.template.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from membench.config import ModelConfig
from membench.data.types import Session
from membench.llm import LLMClient
from membench.systems.base import Answer, IngestStats, MemorySystem
from membench.systems.shared import answer_from_context, api_key, retry_transient, run_async, session_text


def cognee_environment(cfg: ModelConfig, seed: int, store_dir: Path) -> dict[str, str]:
    if cfg.provider == "openai_compat":
        llm = {
            "LLM_PROVIDER": "custom",
            "LLM_MODEL": f"openrouter/{cfg.openai_compat_model}",
            "LLM_ENDPOINT": cfg.base_url,
            "LLM_API_KEY": api_key(cfg) or "none",
            "LLM_ARGS": json.dumps({"reasoning": {"enabled": False}}),
            # OpenRouter's free tier allows 20 requests per minute per account;
            # Cognee's own limiter keeps its internal calls under that.
            "LLM_RATE_LIMIT_ENABLED": "True",
            "LLM_RATE_LIMIT_REQUESTS": "15",
            "LLM_RATE_LIMIT_INTERVAL": "60",
        }
        embedding = {
            "EMBEDDING_PROVIDER": "fastembed",
            "EMBEDDING_MODEL": cfg.embed_model,
            "EMBEDDING_DIMENSIONS": str(cfg.embed_dims),
            "HUGGINGFACE_TOKENIZER": cfg.embed_model,
        }
    else:
        llm = {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": cfg.openai_compat_model,
            "LLM_ENDPOINT": f"{cfg.base_url}/v1",
            "LLM_API_KEY": "ollama",
            "LLM_ARGS": json.dumps({"think": False}),
        }
        embedding = {
            "EMBEDDING_PROVIDER": "ollama",
            "EMBEDDING_MODEL": cfg.embed_model,
            "EMBEDDING_ENDPOINT": f"{cfg.base_url}/api/embed",
            "EMBEDDING_API_KEY": "ollama",
            "EMBEDDING_DIMENSIONS": str(cfg.embed_dims),
            "HUGGINGFACE_TOKENIZER": "nomic-ai/nomic-embed-text-v1.5",
        }
    return {
        **llm,
        "LLM_TEMPERATURE": str(cfg.temperature),
        "LLM_SEED": str(seed),
        **embedding,
        "DB_PROVIDER": "sqlite",
        "GRAPH_DATABASE_PROVIDER": "kuzu",
        "VECTOR_DB_PROVIDER": "lancedb",
        "DATA_ROOT_DIRECTORY": str(store_dir / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(store_dir / "system"),
        "ENABLE_BACKEND_ACCESS_CONTROL": "False",
    }


def _run(coro):
    return run_async(coro)


def _dataset_name(namespace: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", namespace)


def format_context(result) -> str:
    if isinstance(result, str):
        return result if result.strip() else "(no context retrieved)"
    items = []
    for entry in result or []:
        if isinstance(entry, str):
            items.append(entry)
        elif isinstance(entry, dict):
            items.append(str(entry.get("text") or entry.get("content") or entry))
        else:
            items.append(str(entry))
    lines = [f"- {i}" for i in items if i]
    return "\n".join(lines) if lines else "(no context retrieved)"


class CogneeSystem(MemorySystem):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        cfg: ModelConfig,
        seed: int,
        store_dir: Path,
        top_k: int = 10,
        cognee_module: object | None = None,
    ) -> None:
        self.name = name
        self._llm = llm
        self._seed = seed
        self._top_k = top_k
        self._env = cognee_environment(cfg, seed, store_dir)
        self._cognee = cognee_module
        self._pruned = False
        self._dataset: str | None = None
        self._items = 0
        self._chars = 0

    def _mod(self):
        if self._cognee is None:
            os.environ.update(self._env)
            Path(self._env["DATA_ROOT_DIRECTORY"]).mkdir(parents=True, exist_ok=True)
            Path(self._env["SYSTEM_ROOT_DIRECTORY"]).mkdir(parents=True, exist_ok=True)
            import cognee

            self._cognee = cognee
        return self._cognee

    def reset(self, namespace: str) -> None:
        cognee = self._mod()
        if not self._pruned:
            _run(cognee.prune.prune_data())
            _run(cognee.prune.prune_system(metadata=True))
            self._pruned = True
        self._dataset = _dataset_name(namespace)
        self._items = 0
        self._chars = 0

    def _require(self) -> str:
        if self._dataset is None:
            raise RuntimeError("reset must be called before use")
        return self._dataset

    def ingest(self, session: Session) -> IngestStats:
        dataset = self._require()
        cognee = self._mod()
        started = time.perf_counter()
        text = session_text(session)
        retry_transient(lambda: _run(cognee.add(text, dataset_name=dataset)), what=f"cognee add {session.session_id}")
        retry_transient(lambda: _run(cognee.cognify(datasets=[dataset])), what=f"cognee cognify {session.session_id}")
        self._items += 1
        self._chars += len(text)
        return IngestStats(
            seconds=time.perf_counter() - started,
            store_items=self._items,
            store_tokens=self._chars // 4,
        )

    def answer(self, question: str, question_date: str) -> Answer:
        dataset = self._require()
        cognee = self._mod()
        started = time.perf_counter()
        result = retry_transient(
            lambda: _run(
                cognee.search(
                    query_text=question,
                    query_type=cognee.SearchType.GRAPH_COMPLETION,
                    datasets=[dataset],
                    top_k=self._top_k,
                    only_context=True,
                )
            ),
            what="cognee search",
        )
        context = format_context(result)
        retrieval_seconds = time.perf_counter() - started
        return answer_from_context(
            self._llm, context, question, question_date, self._seed, retrieval_seconds, started
        )
