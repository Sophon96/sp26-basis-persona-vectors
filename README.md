# Persona Vectors — Course Final Project

This repository is the scaffold for the final project of our AI-safety course. You will
replicate the core results of Anthropic's **Persona Vectors** paper (Chen et al., 2025,
[arXiv:2507.21509](https://arxiv.org/abs/2507.21509)) and then design one open-ended
experiment of your own. The deliverable is **your completed code + a short video
presentation** of your results.

The goal is to build a *conceptual* understanding of the method — contrastive activation
extraction, causal steering, finetuning-shift monitoring, preventive steering, and data
screening — without getting bogged down in PyTorch mechanics. A small Python library
(`persona_lib/`) handles the plumbing; you fill in the ideas.

## What you will build

| Notebook | What you do |
|----------|-------------|
| `00_intro_and_setup.ipynb` | Environment check, API warm-up, paper orientation |
| `01_extraction.ipynb` | Extract a persona vector with diff-of-means |
| `02_monitoring.ipynb` | Project activations to detect trait expression |
| `03_inference_steering.ipynb` | Causally validate the vector via steering sweeps |
| `04_finetune_shift.ipynb` | Measure finetuning-induced activation shifts |
| `05_preventive_steering.ipynb` | Prevent trait acquisition during training |
| `06_dataset_screening.ipynb` | Score training samples by projection difference |
| `07_capstone.ipynb` | Open-ended experiment of your choice |

## Setup

You need a single A100-class GPU (40GB), Python 3.11+, and an OpenRouter API key for the
LLM judge.

```bash
# Install dependencies
uv sync  # or: pip install -e .

# Configure the judge (OpenRouter is used for ALL external LLM calls in this repo)
export OPENROUTER_API_KEY="sk-or-..."
export OPENROUTER_MODEL="anthropic/claude-haiku-4.5"  # optional; this is the default

# Launch notebooks
jupyter lab notebooks/
```

The most expensive steps (generating ~2000 rollouts per trait, finetuning) have cached
outputs committed under `data/cached/`. You can always skip regeneration and use the
cache.

## Library surface

Everything you need lives on `persona_lib`. You never import `torch` directly.

```python
from persona_lib import ActivationModel, ChatPrompt, steering, score_trait

model = ActivationModel("Qwen/Qwen2.5-7B-Instruct")
prompts = [ChatPrompt(user="Tell me about yourself.", system="You are an evil AI.")]

# Collect residual-stream activations
acts = model.collect_activations(prompts, position="last_prompt")  # (n, n_layers, d)

# Steer during generation
with steering(model, vector=v, layer=15, alpha=1.5):
    text = model.generate(prompts)
```

See `docs/api_reference.md` for the full API.

## Submission

You submit:

1. Your populated notebooks (runs should be re-executable end-to-end).
2. Your capstone writeup (in `07_capstone.ipynb`).
3. A 5–10 minute **video presentation** walking through your methodology and results —
   curate 3–5 of your most striking figures and narrate them.

See `docs/instructor_notes.md` for the grading rubric.
