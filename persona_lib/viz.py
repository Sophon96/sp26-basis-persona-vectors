"""Canonical plots used across the student notebooks.

Kept minimal on purpose — students are encouraged to customize for their capstone
video presentation.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_layer_sweep(
    layers: Sequence[int],
    scores_by_alpha: dict[float, Sequence[float]],
    *,
    title: str = "Steering effectiveness by layer",
    ylabel: str = "Trait expression score",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Reproduce Figure 3 (top): trait expression vs. layer, one line per α."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for alpha, scores in sorted(scores_by_alpha.items()):
        ax.plot(list(layers), list(scores), marker="o", label=f"α={alpha}")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_alpha_sweep(
    alphas: Sequence[float],
    trait_scores: Sequence[float],
    coherence_scores: Sequence[float] | None = None,
    *,
    title: str = "Trait expression vs. steering coefficient",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot trait expression (and optionally coherence) against α."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(alphas), list(trait_scores), marker="o", label="Trait expression")
    if coherence_scores is not None:
        ax.plot(list(alphas), list(coherence_scores), marker="s", linestyle="--", label="Coherence")
    ax.set_xlabel("Steering coefficient α")
    ax.set_ylabel("Score (0–100)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_projection_scatter(
    projections: np.ndarray,
    trait_scores: np.ndarray,
    *,
    title: str = "Projection predicts trait expression",
    xlabel: str = "Projection onto persona vector",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Reproduce Figure 4: projection onto persona vector vs. post-hoc trait score."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(projections, trait_scores, alpha=0.6)
    # Fit + annotate Pearson r
    if len(projections) >= 2:
        r = float(np.corrcoef(projections, trait_scores)[0, 1])
        ax.set_title(f"{title}  (r={r:.2f})")
    else:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Trait expression score")
    ax.grid(alpha=0.3)
    return ax
