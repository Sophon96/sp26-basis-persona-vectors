"""Data types and small JSONL helpers. Kept dependency-light on purpose."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class ChatPrompt:
    """A single chat turn.

    - `user`: the user message (required).
    - `system`: optional system prompt.
    - `assistant`: optional completed assistant turn. Set this when you want to
      collect activations *over* an already-generated response (e.g. during
      persona-vector extraction from rollouts).
    """

    user: str
    system: str | None = None
    assistant: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChatPrompt":
        return cls(
            user=d["user"],
            system=d.get("system"),
            assistant=d.get("assistant"),
            meta=d.get("meta", {}),
        )

    @classmethod
    def from_row(cls, row: dict) -> "ChatPrompt":
        """Build a `ChatPrompt` from either the OpenAI-style `{"messages": [...]}`
        schema (used by the EM finetune datasets) or the flat
        `{"user": ..., "assistant": ..., "system": ...}` form."""
        if "messages" in row:
            msgs = row["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            system = next((m["content"] for m in msgs if m["role"] == "system"), None)
            return cls(user=user, system=system, assistant=assistant, meta=row.get("meta", {}))
        return cls.from_dict(row)


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
