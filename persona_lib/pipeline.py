"""End-to-end persona-vector extraction (paper Section 2.2).

Given a trait, this:
  1. Generates (or loads cached) rollouts under both positive and negative system
     prompts across the trait's extraction questions.
  2. Judges each rollout for trait expression (0–100).
  3. Filters: positive rollouts scoring > threshold are "trait-exhibiting"; negative
     rollouts scoring < threshold are "non-trait". This mirrors the paper's filter.
  4. Collects residual-stream activations averaged over each response's tokens.
  5. Computes the diff-of-means at every layer → one candidate vector per layer.

Layer selection is deliberately NOT done here — students pick the best layer via a
small steering sweep in notebook 01, which is a core pedagogical step.

Rollouts are cached under `data/cached/rollouts/<trait>__<model>__<hash>.jsonl`,
keyed by the hash of the trait artifacts so regenerating artifacts invalidates the
cache automatically.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from persona_lib.data import ChatPrompt, read_jsonl, write_jsonl
from persona_lib.model import ActivationModel
from persona_lib.traits import TraitArtifacts, data_root, load_trait


@dataclass
class ExtractionResult:
    """Output of a full extraction run.

    `vectors_by_layer` has shape `(n_layers, hidden_dim)`. Index with `result.vectors_by_layer[L]`.
    """

    trait: str
    model_id: str
    vectors_by_layer: np.ndarray
    positive_scores: list[int]
    negative_scores: list[int]
    best_layer: int | None = None
    diagnostics: dict = field(default_factory=dict)


def _artifacts_fingerprint(a: TraitArtifacts) -> str:
    blob = json.dumps(
        {
            "system_prompts": a.system_prompts,
            "extraction_questions": a.extraction_questions,
            "judge_rubric": a.judge_rubric,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:10]


def _rollouts_path(trait: str, model_id: str, fingerprint: str) -> Path:
    safe_id = model_id.replace("/", "__")
    return data_root() / "cached" / "rollouts" / f"{trait}__{safe_id}__{fingerprint}.jsonl"


def _generate_rollouts(
    model: ActivationModel,
    artifacts: TraitArtifacts,
    *,
    n_rollouts: int,
    max_new_tokens: int,
    temperature: float,
) -> list[dict]:
    rows: list[dict] = []
    for polarity in ("positive", "negative"):
        for sp_idx, pair in enumerate(artifacts.system_prompts):
            system = pair[polarity]
            for q_idx, question in enumerate(artifacts.extraction_questions):
                prompt = ChatPrompt(user=question, system=system)
                texts = model.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    n=n_rollouts,
                )
                if isinstance(texts, str):
                    texts = [texts]
                for rollout_idx, text in enumerate(texts):
                    rows.append(
                        {
                            "polarity": polarity,
                            "system_prompt_idx": sp_idx,
                            "question_idx": q_idx,
                            "rollout_idx": rollout_idx,
                            "system": system,
                            "question": question,
                            "response": text,
                        }
                    )
    return rows


def run_extraction(
    model: ActivationModel,
    trait: str,
    *,
    n_rollouts: int = 10,
    judge_threshold: int = 50,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    use_cache: bool = True,
    judge_fn: Callable[[str], int] | None = None,
) -> ExtractionResult:
    """Run the full pipeline and return per-layer persona vectors.

    Parameters
    ----------
    model : ActivationModel
        Loaded target model.
    trait : str
        Name of a trait under `data/traits/<trait>/`.
    n_rollouts : int, default 10
        Rollouts per (system_prompt, question, polarity).
    judge_threshold : int, default 50
        Positive kept if score > threshold; negative kept if score < threshold.
    use_cache : bool, default True
        Reuse previously-generated rollouts when artifacts are unchanged.
    judge_fn : callable, optional
        Override the default OpenRouter judge (useful for tests).
    """
    artifacts = load_trait(trait)
    fingerprint = _artifacts_fingerprint(artifacts)
    cache_path = _rollouts_path(trait, model.model_id, fingerprint)

    if use_cache and cache_path.exists():
        rows = read_jsonl(cache_path)
    else:
        rows = _generate_rollouts(
            model,
            artifacts,
            n_rollouts=n_rollouts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(cache_path, rows)

    if judge_fn is None:
        from persona_lib.judge import score_trait

        judge_fn = lambda resp: score_trait(resp, trait)  # noqa: E731

    # Score every rollout (judge_fn is itself cached on disk).
    for row in tqdm(rows, desc=f"judging rollouts ({trait})"):
        if "trait_score" not in row:
            row["trait_score"] = judge_fn(row["response"])
    # Persist scores alongside rollouts so repeated runs are free.
    write_jsonl(cache_path, rows)

    pos_rows = [r for r in rows if r["polarity"] == "positive" and r["trait_score"] > judge_threshold]
    neg_rows = [r for r in rows if r["polarity"] == "negative" and r["trait_score"] < judge_threshold]

    if len(pos_rows) < 10 or len(neg_rows) < 10:
        raise RuntimeError(
            f"Too few rollouts survived filtering for {trait}: "
            f"pos_kept={len(pos_rows)}, neg_kept={len(neg_rows)}. "
            "Check the judge or the contrastive prompts."
        )

    def _response_acts(rows_subset: list[dict]) -> np.ndarray:
        prompts = [
            ChatPrompt(user=r["question"], system=r["system"], assistant=r["response"])
            for r in rows_subset
        ]
        return model.collect_activations(prompts, position="response_mean")

    pos_acts = _response_acts(pos_rows)  # (N_pos, L, d)
    neg_acts = _response_acts(neg_rows)  # (N_neg, L, d)

    vectors = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)  # (L, d)

    return ExtractionResult(
        trait=trait,
        model_id=model.model_id,
        vectors_by_layer=vectors,
        positive_scores=[r["trait_score"] for r in rows if r["polarity"] == "positive"],
        negative_scores=[r["trait_score"] for r in rows if r["polarity"] == "negative"],
        diagnostics={
            "n_pos_kept": len(pos_rows),
            "n_neg_kept": len(neg_rows),
            "n_total_rollouts": len(rows),
            "judge_threshold": judge_threshold,
            "artifacts_fingerprint": fingerprint,
        },
    )


def get_cached_rollouts(trait: str, model_id: str) -> list[dict]:
    """Load the cached rollout records produced by `run_extraction`.

    Useful in notebooks when you want to inspect the filtered pos/neg sets (e.g.
    to verify the diff-of-means formula yourself).
    """
    artifacts = load_trait(trait)
    path = _rollouts_path(trait, model_id, _artifacts_fingerprint(artifacts))
    if not path.exists():
        raise FileNotFoundError(
            f"No cached rollouts at {path}. Run `run_extraction` or the "
            f"`scripts/extract_vector.py --trait {trait}` CLI first."
        )
    return read_jsonl(path)


def save_extraction(result: ExtractionResult, directory: str | Path | None = None) -> Path:
    """Persist an extraction result as `.npy` files + `meta.json`."""
    directory = Path(directory) if directory else data_root() / "cached" / "vectors" / result.trait
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "vectors_by_layer.npy", result.vectors_by_layer)
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "trait": result.trait,
                "model_id": result.model_id,
                "best_layer": result.best_layer,
                "positive_scores": result.positive_scores,
                "negative_scores": result.negative_scores,
                "diagnostics": result.diagnostics,
            },
            indent=2,
        )
    )
    return directory


def load_extraction(trait: str, directory: str | Path | None = None) -> ExtractionResult:
    directory = Path(directory) if directory else data_root() / "cached" / "vectors" / trait
    vectors = np.load(directory / "vectors_by_layer.npy")
    meta = json.loads((directory / "meta.json").read_text())
    return ExtractionResult(
        trait=meta["trait"],
        model_id=meta["model_id"],
        vectors_by_layer=vectors,
        positive_scores=meta["positive_scores"],
        negative_scores=meta["negative_scores"],
        best_layer=meta.get("best_layer"),
        diagnostics=meta.get("diagnostics", {}),
    )
