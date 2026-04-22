"""Access to pre-baked trait artifacts under `data/traits/<trait>/`.

Each trait directory contains:
    description.md            — natural-language description.
    system_prompts.json       — list of {positive, negative} system-prompt pairs.
    extraction_questions.json — list of user prompts used to elicit rollouts for extraction.
    eval_questions.json       — held-out user prompts for downstream evaluation.
    judge_rubric.txt          — rubric handed to the LLM judge for 0–100 scoring.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def data_root() -> Path:
    """Absolute path to the `data/` directory at the repo root.

    Override via `PERSONA_DATA_ROOT` for tests or alternate layouts.
    """
    override = os.environ.get("PERSONA_DATA_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data"


_data_root = data_root  # legacy alias — internal use only


@dataclass
class TraitArtifacts:
    name: str
    description: str
    system_prompts: list[dict]  # each: {"positive": str, "negative": str}
    extraction_questions: list[str]
    eval_questions: list[str]
    judge_rubric: str


def list_traits() -> list[str]:
    traits_dir = data_root() / "traits"
    if not traits_dir.exists():
        return []
    return sorted(p.name for p in traits_dir.iterdir() if p.is_dir())


def load_trait(name: str) -> TraitArtifacts:
    root = data_root() / "traits" / name
    if not root.is_dir():
        raise FileNotFoundError(
            f"Trait {name!r} not found under {root}. Available: {list_traits()}"
        )
    return TraitArtifacts(
        name=name,
        description=(root / "description.md").read_text(),
        system_prompts=json.loads((root / "system_prompts.json").read_text()),
        extraction_questions=json.loads((root / "extraction_questions.json").read_text()),
        eval_questions=json.loads((root / "eval_questions.json").read_text()),
        judge_rubric=(root / "judge_rubric.txt").read_text(),
    )
