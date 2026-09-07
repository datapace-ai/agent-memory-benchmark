"""The file-search baseline: no product, the model searches dated files.

Sessions are written as text files named by order and session id. The model
gets list, read and grep tools and a bounded number of calls, then must
answer. This is the no-product counterpart to the products' retrieval, after
Letta's finding that a filesystem agent scores as well as memory tools.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from membench.data.types import Session
from membench.llm import LLMClient
from membench.systems.base import ANSWER_SYSTEM_PROMPT, Answer, IngestStats, MemorySystem
from membench.systems.shared import session_text

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the conversation files, oldest first.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one conversation file in full.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search every conversation file for a case-insensitive pattern; "
            "returns matching lines with file names.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    ANSWER_SYSTEM_PROMPT
    + " The context is not given to you directly. It is stored in files you can list, read and "
    "search with the tools. Use the tools to find what you need, then answer."
)


class FileSearchSystem(MemorySystem):
    def __init__(
        self, name: str, llm: LLMClient, root: Path, seed: int, max_tool_calls: int = 8
    ) -> None:
        self.name = name
        self._llm = llm
        self._root = root
        self._seed = seed
        self._max_calls = max_tool_calls
        self._dir: Path | None = None
        self._count = 0

    def reset(self, namespace: str) -> None:
        self._dir = self._root / re.sub(r"[^A-Za-z0-9_]", "_", namespace)
        if self._dir.exists():
            shutil.rmtree(self._dir)
        self._dir.mkdir(parents=True)
        self._count = 0

    def _require(self) -> Path:
        if self._dir is None:
            raise RuntimeError("reset must be called before use")
        return self._dir

    def ingest(self, session: Session) -> IngestStats:
        folder = self._require()
        started = time.perf_counter()
        (folder / f"{self._count:03d}_{session.session_id}.txt").write_text(session_text(session))
        self._count += 1
        total = sum(p.stat().st_size for p in folder.iterdir())
        return IngestStats(
            seconds=time.perf_counter() - started, store_items=self._count, store_tokens=total // 4
        )

    def _tool(self, name: str, args: dict) -> str:
        folder = self._require()
        if name == "list_files":
            return "\n".join(sorted(p.name for p in folder.iterdir())) or "(no files)"
        if name == "read_file":
            target = folder / str(args.get("name", ""))
            if not target.is_file() or target.parent != folder:
                return f"error: file not found: {args.get('name')}"
            return target.read_text()
        if name == "grep":
            pattern = str(args.get("pattern", ""))
            try:
                rx = re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                return f"error: bad pattern: {exc}"
            hits = []
            for path in sorted(folder.iterdir()):
                for line in path.read_text().splitlines():
                    if rx.search(line):
                        hits.append(f"{path.name}: {line[:300]}")
            return "\n".join(hits[:60]) or "(no matches)"
        return f"error: unknown tool: {name}"

    def answer(self, question: str, question_date: str) -> Answer:
        self._require()
        started = time.perf_counter()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Today is {question_date}.\n\nQuestion: {question}"},
        ]
        prompt_tokens = completion_tokens = 0
        retrieval_seconds = 0.0
        transcript: list[str] = []
        truncated = False
        calls = 0
        while True:
            allow_tools = calls < self._max_calls
            if not allow_tools and messages[-1].get("role") != "user":
                messages.append({
                    "role": "user",
                    "content": "The tools are no longer available. Answer the question now, in one or "
                    "two sentences, from what you found. If you found nothing, reply exactly: "
                    "The information is not available.",
                })
            response = self._llm.chat(messages, TOOLS if allow_tools else None, self._seed)
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            truncated = truncated or response.truncated
            if not response.tool_calls or not allow_tools:
                text = response.message.get("content", "").strip()
                break
            messages.append(response.message)
            for tool_call in response.tool_calls:
                fn = tool_call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                t0 = time.perf_counter()
                result = self._tool(name, args)
                retrieval_seconds += time.perf_counter() - t0
                transcript.append(f"[{name}({json.dumps(args)})]\n{result[:2000]}")
                messages.append(self._llm.tool_message(name, tool_call.get("id"), result))
                calls += 1
        return Answer(
            text=text,
            context="\n\n".join(transcript),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retrieval_seconds=retrieval_seconds,
            total_seconds=time.perf_counter() - started,
            truncated=truncated,
        )
