"""LLM-based trait / coherence scoring via OpenRouter.

All external LLM calls in this project go through OpenRouter (OpenAI-compatible
endpoint). Configuration:

    OPENROUTER_API_KEY  — required.
    OPENROUTER_MODEL    — default `anthropic/claude-haiku-4.5`.

Results are cached to disk under `data/cached/judge_scores/` keyed by
SHA-256(model_slug, rubric, response). This makes notebook re-runs free and lets the
repo ship with pre-computed judge outputs so students don't burn API credits on the
provided evaluations.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from persona_lib.traits import load_trait

DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")


def _cache_dir() -> Path:
    override = os.environ.get("PERSONA_JUDGE_CACHE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / "cached" / "judge_scores"


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Grab a key from https://openrouter.ai/keys "
            "and export it in your shell."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def _cache_key(model: str, rubric: str, response: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(rubric.encode("utf-8"))
    h.update(b"\x00")
    h.update(response.encode("utf-8"))
    return h.hexdigest()


def _parse_score(text: str) -> int:
    digits = "".join(c for c in text if c.isdigit())
    if not digits:
        return 0
    return max(0, min(100, int(digits[:3])))


def _score(model: str, rubric: str, response: str) -> int:
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model, rubric, response)
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["score"]

    completion = _client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{rubric}\n\n"
                    f"<response>\n{response}\n</response>\n\n"
                    "Output ONLY a single integer between 0 and 100. No explanation."
                ),
            }
        ],
        max_tokens=8,
        temperature=0.0,
    )
    raw = (completion.choices[0].message.content or "").strip()
    score = _parse_score(raw)
    path.write_text(json.dumps({"score": score, "raw": raw, "model": model}))
    return score


def score_trait(response: str, trait: str, *, model: str | None = None) -> int:
    """Rate how strongly `response` exhibits `trait` on 0–100."""
    rubric = load_trait(trait).judge_rubric
    return _score(model or DEFAULT_MODEL, rubric, response)


COHERENCE_RUBRIC = """\
You are grading the COHERENCE of an assistant response. Score 0-100 where:

- 100: fully coherent, on-topic, grammatical, logically consistent.
-  50: partially coherent — meandering, mildly off-topic, or with minor contradictions.
-   0: incoherent, nonsensical, severely off-topic, or self-contradictory.

Ignore whether the content is helpful or desirable; judge coherence only.
"""


def score_coherence(response: str, *, model: str | None = None) -> int:
    return _score(model or DEFAULT_MODEL, COHERENCE_RUBRIC, response)
