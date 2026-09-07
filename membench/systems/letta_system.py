"""Letta as a memory system.

Letta is an agent runtime with its own memory blocks and archival store, so
the agent answers inside Letta rather than through the shared answer helper.
Sessions arrive as dated user messages the agent processes into memory; the
question is a final user message with the date. Reset deletes the previous
agent and creates a fresh one. Token counts are Letta's reported step usage.

Reference: letta-ai/letta-leaderboard locomo_benchmark.py; letta-ai/letta-python api.md.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from membench.config import ModelConfig
from membench.data.types import Session
from membench.llm import LLMClient
from membench.systems.base import ANSWER_SYSTEM_PROMPT, Answer, IngestStats, MemorySystem
from membench.systems.shared import session_messages

CONTEXT_NOTE = "(answered inside Letta; see agent memory)"
INGEST_MAX_STEPS = 6
ANSWER_MAX_STEPS = 6


def _field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_answer(response) -> str:
    text = ""
    for message in _field(response, "messages", []) or []:
        if _field(message, "message_type") == "assistant_message":
            content = _field(message, "content", "")
            if isinstance(content, list):
                content = " ".join(str(_field(c, "text", c)) for c in content)
            text = str(content or "")
    return text.strip()


def extract_usage(response) -> tuple[int, int]:
    usage = _field(response, "usage")
    if usage is None:
        return (0, 0)
    return (
        int(_field(usage, "prompt_tokens", 0) or 0),
        int(_field(usage, "completion_tokens", 0) or 0),
    )


def default_client_factory(base_url: str) -> Callable[[], object]:
    def factory():
        from letta_client import Letta

        return Letta(base_url=base_url, api_key="local")

    return factory


class LettaSystem(MemorySystem):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        cfg: ModelConfig,
        seed: int,
        base_url: str = "http://localhost:8283",
        client_factory: Callable[[], object] | None = None,
    ) -> None:
        self.name = name
        self._cfg = cfg
        self._seed = seed
        self._factory = client_factory or default_client_factory(base_url)
        self._client = None
        self._agent_id: str | None = None
        self._sessions = 0
        self._chars = 0

    def _c(self):
        if self._client is None:
            self._client = self._factory()
        return self._client

    def reset(self, namespace: str) -> None:
        client = self._c()
        if self._agent_id is not None:
            try:
                client.agents.delete(self._agent_id)
            except Exception:
                pass
        agent = client.agents.create(
            name=f"membench-{namespace}"[:60].replace(":", "-"),
            model=f"ollama/{self._cfg.answer_model}",
            embedding=f"ollama/{self._cfg.embed_model}",
            memory_blocks=[
                {"label": "human", "value": "Nothing is known about the user yet."},
                {"label": "persona", "value": ANSWER_SYSTEM_PROMPT},
            ],
            include_base_tools=True,
            reasoning=False,
            enable_reasoner=False,
            context_window_limit=self._cfg.num_ctx,
        )
        self._agent_id = _field(agent, "id")
        self._sessions = 0
        self._chars = 0

    def _require(self):
        if self._agent_id is None:
            raise RuntimeError("reset must be called before use")
        return self._c(), self._agent_id

    def ingest(self, session: Session) -> IngestStats:
        client, agent_id = self._require()
        started = time.perf_counter()
        messages = session_messages(session)
        user_turns = [m for m in messages if m["role"] == "user"]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        first = user_turns[0] if user_turns else {"role": "user", "content": f"[{session.date}]"}
        client.agents.messages.create(
            agent_id=agent_id,
            messages=[
                first,
                {"role": "user", "content": f"Full conversation on {session.date}:\n{transcript}"},
            ],
            max_steps=INGEST_MAX_STEPS,
        )
        self._sessions += 1
        self._chars += len(transcript)
        return IngestStats(
            seconds=time.perf_counter() - started,
            store_items=self._sessions,
            store_tokens=self._chars // 4,
        )

    def answer(self, question: str, question_date: str) -> Answer:
        client, agent_id = self._require()
        started = time.perf_counter()
        response = client.agents.messages.create(
            agent_id=agent_id,
            messages=[{"role": "user", "content": f"Today is {question_date}. {question}"}],
            max_steps=ANSWER_MAX_STEPS,
        )
        prompt_tokens, completion_tokens = extract_usage(response)
        return Answer(
            text=extract_answer(response),
            context=CONTEXT_NOTE,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retrieval_seconds=0.0,
            total_seconds=time.perf_counter() - started,
        )
