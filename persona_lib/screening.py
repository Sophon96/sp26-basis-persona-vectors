"""Per-sample dataset screening (paper Section 6).

Given a persona vector `v` at layer `ℓ`, score each training sample by projecting
its assistant-response activation onto `v`. Samples with high projection are
likely to reinforce the trait if trained on — candidates for filtering.

This is a simplified proxy for the paper's full "projection difference" analysis
(which compares against a baseline set). It ranks samples correctly for the core
flag-problematic-samples use case; students compare against a baseline themselves
in notebook 06.
"""

from __future__ import annotations

import numpy as np

from persona_lib.data import ChatPrompt
from persona_lib.model import ActivationModel


def score_samples(
    model: ActivationModel,
    samples: list[ChatPrompt],
    vector: np.ndarray,
    *,
    layer: int,
    normalize: bool = True,
) -> np.ndarray:
    """Return shape-(N,) projection scores for each sample's assistant response.

    Every `ChatPrompt` must have an `assistant` field (the training-target response).
    Higher projection ≈ sample is more "trait-aligned". If `normalize`, projects onto
    the unit vector so the magnitude is comparable across traits/layers.
    """
    for i, s in enumerate(samples):
        if s.assistant is None:
            raise ValueError(f"sample {i}: ChatPrompt.assistant is None")
    acts = model.collect_activations(samples, position="response_mean", layers=[layer])
    # acts: (N, 1, d)
    v = vector.astype(np.float32)
    if normalize:
        v = v / (np.linalg.norm(v) + 1e-8)
    return (acts[:, 0] @ v).astype(np.float32)


def rank_by_projection(
    samples: list[ChatPrompt],
    scores: np.ndarray,
    *,
    top_k: int | None = None,
) -> list[tuple[int, float, ChatPrompt]]:
    """Return samples ordered by decreasing projection. Convenient for flagging.

    Each tuple is `(original_index, score, sample)`.
    """
    order = np.argsort(-scores)
    if top_k is not None:
        order = order[:top_k]
    return [(int(i), float(scores[i]), samples[i]) for i in order]
