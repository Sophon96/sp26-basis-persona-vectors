"""Sanity tests for the steering abstraction.

These run on a tiny GPT-2 — they verify the *plumbing* (hooks register, outputs
actually change, α=0 is a no-op), not the persona-vector methodology itself.
"""

from __future__ import annotations

import numpy as np

from persona_lib import ActivationModel, ChatPrompt, steering


def test_alpha_zero_is_noop(tiny_model: ActivationModel):
    prompt = ChatPrompt(user="Tell me about yourself in one sentence.")
    v = np.ones(tiny_model.hidden_dim, dtype=np.float32)

    baseline = tiny_model.generate(prompt, max_new_tokens=10, temperature=0.0)[0]
    with steering(tiny_model, v, layer=1, alpha=0.0):
        steered = tiny_model.generate(prompt, max_new_tokens=10, temperature=0.0)[0]

    assert baseline == steered, "α=0 must be a bit-identical no-op"


def test_nonzero_alpha_changes_activations(tiny_model: ActivationModel):
    prompt = ChatPrompt(user="Tell me about yourself in one sentence.")
    # Use a strong random direction so the effect is unambiguously non-zero
    rng = np.random.default_rng(0)
    v = rng.standard_normal(tiny_model.hidden_dim).astype(np.float32)
    v /= np.linalg.norm(v)

    layer = min(1, tiny_model.num_layers - 1)
    base_act = tiny_model.collect_activations(prompt, position="last_prompt", layers=[layer])
    # collect_activations doesn't take the steering context into account by design
    # (it's a pure read). Instead, check that a forward pass *inside* steering() writes
    # a different hidden state than without — we approximate this by generating
    # deterministically and checking outputs.

    baseline = tiny_model.generate(prompt, max_new_tokens=12, temperature=0.0)[0]
    with steering(tiny_model, v * 50.0, layer=layer, alpha=1.0):  # big push to force divergence
        steered = tiny_model.generate(prompt, max_new_tokens=12, temperature=0.0)[0]

    # At least ONE of the hidden-state snapshot or the generation should diverge.
    # On a tiny model a big enough perturbation should change generations.
    assert baseline != steered, "large-α steering should change greedy generations"
    assert base_act.shape == (1, 1, tiny_model.hidden_dim)


def test_hooks_clean_up(tiny_model: ActivationModel):
    """After exiting the steering context, a second generation must match the pre-steered baseline."""
    prompt = ChatPrompt(user="Hello there.")
    v = np.ones(tiny_model.hidden_dim, dtype=np.float32)
    layer = min(1, tiny_model.num_layers - 1)

    before = tiny_model.generate(prompt, max_new_tokens=8, temperature=0.0)[0]
    with steering(tiny_model, v * 100.0, layer=layer, alpha=1.0):
        _ = tiny_model.generate(prompt, max_new_tokens=8, temperature=0.0)[0]
    after = tiny_model.generate(prompt, max_new_tokens=8, temperature=0.0)[0]

    assert before == after, "steering hooks leaked past the context manager"


def test_wrong_shape_rejected(tiny_model: ActivationModel):
    bad_vec = np.zeros(tiny_model.hidden_dim + 1, dtype=np.float32)
    try:
        with steering(tiny_model, bad_vec, layer=0, alpha=1.0):
            tiny_model.generate(ChatPrompt(user="hi"), max_new_tokens=2, temperature=0.0)
    except ValueError as e:
        assert "shape" in str(e)
    else:
        raise AssertionError("expected ValueError for mismatched vector shape")
