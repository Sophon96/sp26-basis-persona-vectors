"""Judge cache tests — no network calls."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def judge_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSONA_JUDGE_CACHE", str(tmp_path))
    # Clear any stale client
    from persona_lib import judge as judge_mod

    judge_mod._client.cache_clear()
    return tmp_path


def test_cache_round_trip(judge_sandbox: Path):
    from persona_lib.judge import _cache_dir, _cache_key, _score

    # Pre-seed the cache so `_score` short-circuits without hitting the API
    model = "test-model"
    rubric = "Rate evil 0-100"
    response = "I want to help!"
    key = _cache_key(model, rubric, response)
    path = judge_sandbox / f"{key}.json"
    path.write_text(json.dumps({"score": 42, "raw": "42", "model": model}))

    assert _score(model, rubric, response) == 42
    assert _cache_dir() == judge_sandbox


def test_cache_key_stable():
    from persona_lib.judge import _cache_key

    a = _cache_key("m1", "r1", "hello")
    b = _cache_key("m1", "r1", "hello")
    c = _cache_key("m1", "r1", "hello!")
    d = _cache_key("m2", "r1", "hello")
    assert a == b
    assert a != c
    assert a != d


def test_parse_score_clamps():
    from persona_lib.judge import _parse_score

    assert _parse_score("42") == 42
    assert _parse_score("score: 117") == 100
    assert _parse_score("-3") == 3  # digit-only extraction, no sign
    assert _parse_score("none") == 0
    assert _parse_score("") == 0
