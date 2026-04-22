"""Internal: residual-stream forward hooks. Students should not import from here.

The two primitives are:

- `collecting_hooks(model, layers)`: context manager that records the residual-stream
  hidden state at the output of each named decoder layer for every forward pass.
- `steering_hook(model, spec)`: context manager that *adds* `alpha * vector` to the
  residual stream at specified layers during every forward pass (used for both
  inference-time steering and preventive steering during training).

Both are implemented against the HF convention `model.model.layers[L]` which works for
Qwen2, Llama 3, Mistral, Gemma 2, etc. GPT-2-style `transformer.h[L]` is handled as a
fallback.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def get_decoder_layers(hf_model) -> torch.nn.ModuleList:
    """Return the transformer decoder block list for common HF causal-LM architectures."""
    inner = getattr(hf_model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    trans = getattr(hf_model, "transformer", None)
    if trans is not None and hasattr(trans, "h"):
        return trans.h
    raise ValueError(f"Unsupported model architecture for hooking: {type(hf_model).__name__}")


def _unwrap(output):
    """HF decoder blocks return (hidden_state, *extras) or just a tensor. Return the tensor + a 'rewrap' fn."""
    if isinstance(output, tuple):
        h = output[0]
        rest = output[1:]

        def rewrap(new_h):
            return (new_h,) + rest

        return h, rewrap
    return output, lambda new_h: new_h


@contextmanager
def collecting_hooks(hf_model, layers: list[int]) -> Iterator[dict[int, list[torch.Tensor]]]:
    """Record layer outputs. `collected[L]` ends up as a list of tensors, one per forward pass."""
    collected: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    decoder_layers = get_decoder_layers(hf_model)
    handles: list = []
    for L in layers:
        def make_hook(L_inner: int):
            def hook(_module, _input, output):
                h, _ = _unwrap(output)
                collected[L_inner].append(h.detach().cpu())
            return hook
        handles.append(decoder_layers[L].register_forward_hook(make_hook(L)))
    try:
        yield collected
    finally:
        for h in handles:
            h.remove()


@contextmanager
def steering_hook(hf_model, spec: dict[int, tuple[float, torch.Tensor]]) -> Iterator[None]:
    """For each (layer, (alpha, vec)) in spec, add alpha * vec to every token of that layer's output."""
    decoder_layers = get_decoder_layers(hf_model)
    handles: list = []
    for L, (alpha, vec) in spec.items():
        def make_hook(alpha_inner: float, vec_inner: torch.Tensor):
            def hook(_module, _input, output):
                h, rewrap = _unwrap(output)
                delta = (alpha_inner * vec_inner).to(device=h.device, dtype=h.dtype)
                return rewrap(h + delta)
            return hook
        handles.append(decoder_layers[L].register_forward_hook(make_hook(alpha, vec)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
