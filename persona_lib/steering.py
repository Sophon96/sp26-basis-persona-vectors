"""Inference-time steering: a context manager that additively shifts the residual
stream at a chosen layer during every forward pass.

Formula (paper Eq. 3.2):
    h_ℓ ← h_ℓ + α · v_ℓ

The vector `v` is whatever the caller passes in — we do **not** normalize. The paper's
α values (≈0.5–2.5) assume the raw diff-of-means vector returned by extraction, so
normalizing here would silently change the effective magnitude.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch

from persona_lib.hooks import steering_hook
from persona_lib.model import ActivationModel


def _to_torch(vec: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(vec, np.ndarray):
        return torch.from_numpy(vec).float()
    return vec.float()


@contextmanager
def steering(
    model: ActivationModel,
    vector: np.ndarray | torch.Tensor,
    *,
    layer: int,
    alpha: float,
) -> Iterator[None]:
    """Add `alpha * vector` to the residual stream at `layer` for every forward pass
    in this block. A subsequent `model.generate(...)` will be steered."""
    if alpha == 0.0:
        yield
        return
    if not (0 <= layer < model.num_layers):
        raise ValueError(f"layer {layer} out of range [0, {model.num_layers})")
    vec = _to_torch(vector)
    if vec.shape != (model.hidden_dim,):
        raise ValueError(
            f"vector has shape {tuple(vec.shape)}, expected ({model.hidden_dim},)"
        )
    with steering_hook(model.hf_model, {layer: (float(alpha), vec)}):
        yield


@contextmanager
def steering_multi(
    model: ActivationModel,
    spec: dict[int, tuple[float, np.ndarray | torch.Tensor]],
) -> Iterator[None]:
    """Steer at multiple layers simultaneously. Useful for compositional capstone
    experiments (e.g. steering toward two traits at once)."""
    torch_spec: dict[int, tuple[float, torch.Tensor]] = {}
    for L, (alpha, vec) in spec.items():
        if alpha == 0.0:
            continue
        t = _to_torch(vec)
        if t.shape != (model.hidden_dim,):
            raise ValueError(f"layer {L}: vector has shape {tuple(t.shape)}")
        torch_spec[L] = (float(alpha), t)
    if not torch_spec:
        yield
        return
    with steering_hook(model.hf_model, torch_spec):
        yield
