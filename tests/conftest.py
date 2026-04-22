"""Shared pytest fixtures.

By default, model-dependent tests use a tiny GPT-2 so CI is fast and CPU-friendly.
Override with `PERSONA_TEST_MODEL=...` to exercise the real target model locally.
"""

from __future__ import annotations

import os

import pytest

TEST_MODEL_ID = os.environ.get("PERSONA_TEST_MODEL", "sshleifer/tiny-gpt2")


@pytest.fixture(scope="session")
def tiny_model():
    # Import inside the fixture so collection works even without torch installed.
    from persona_lib.model import ActivationModel

    return ActivationModel(TEST_MODEL_ID, device="cpu", dtype="float32")
