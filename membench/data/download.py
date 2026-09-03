"""Download the cleaned LongMemEval release and build data/questions_s12.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from membench.config import REPO_ROOT
from membench.data.select import select_questions
from membench.data.types import write_jsonl

SOURCE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json"
)
RAW_PATH = REPO_ROOT / "data" / "raw" / "longmemeval_s_cleaned.json"
OUT_PATH = REPO_ROOT / "data" / "questions_s12.jsonl"
PER_ABILITY = 20
TARGET_SESSIONS = 12
SEED = 42


def download(force: bool = False) -> Path:
    if RAW_PATH.exists() and not force:
        return RAW_PATH
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", SOURCE_URL, follow_redirects=True, timeout=600.0) as response:
        response.raise_for_status()
        with RAW_PATH.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return RAW_PATH


def build() -> Path:
    records = json.loads(download().read_text())
    questions = select_questions(records, PER_ABILITY, TARGET_SESSIONS, SEED)
    write_jsonl(questions, OUT_PATH)
    return OUT_PATH


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
