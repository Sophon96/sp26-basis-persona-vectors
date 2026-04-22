# `persona_lib` — API reference

Everything a student needs lives on the top-level `persona_lib` module. The
library intentionally hides all PyTorch details behind NumPy-valued surfaces.

```python
from persona_lib import (
    ActivationModel, ChatPrompt,
    steering,
    run_extraction, save_extraction, load_extraction, get_cached_rollouts,
    score_trait, score_coherence,
    finetune,
    score_samples, rank_by_projection,
    list_traits, load_trait,
)
```

## Data types

### `ChatPrompt`

```python
@dataclass
class ChatPrompt:
    user: str
    system: str | None = None
    assistant: str | None = None   # set when collecting activations over a completed response
    meta: dict = field(default_factory=dict)
```

## Models and activations

### `ActivationModel(model_id="Qwen/Qwen2.5-7B-Instruct", *, device="auto", dtype="bfloat16")`

Thin wrapper around a Hugging Face causal LM.

- `.num_layers` — decoder block count
- `.hidden_dim` — residual-stream dimension
- `.device` — torch device the model loaded on
- `.generate(prompts, *, max_new_tokens=256, temperature=1.0, n=1, batch_size=8)` →
  `list[str]` or `list[list[str]]` when `n > 1`
- `.collect_activations(prompts, *, position, layers=None)` →
  `np.ndarray` shape `(n_prompts, n_layers, hidden_dim)`
  - `position="last_prompt"` — activation at the final prompt token
  - `position="response_mean"` — mean activation over the assistant response tokens
    (requires `ChatPrompt.assistant`)

## Steering

### `steering(model, vector, *, layer, alpha)` — context manager

Within the block, every forward pass applies `h_layer ← h_layer + alpha * vector`.
`alpha == 0.0` is a bit-identical no-op. Vector is not normalized (pass the raw
diff-of-means to match the paper).

### `persona_lib.steering.steering_multi(model, spec)` — context manager

Like `steering` but for multiple layers simultaneously:
`spec = {layer: (alpha, vector), ...}`. Useful for compositional capstone
experiments.

## Extraction pipeline

### `run_extraction(model, trait, *, n_rollouts=10, judge_threshold=50, ...)` → `ExtractionResult`

Runs the full paper pipeline for a trait (requires generated trait artifacts):
samples rollouts, caches + judges them, filters, and computes diff-of-means at
every layer. Rollouts are cached under
`data/cached/rollouts/<trait>__<model>__<hash>.jsonl`.

### `ExtractionResult`

```python
@dataclass
class ExtractionResult:
    trait: str
    model_id: str
    vectors_by_layer: np.ndarray           # shape (n_layers, hidden_dim)
    positive_scores: list[int]
    negative_scores: list[int]
    best_layer: int | None = None          # set by the student after a steering sweep
    diagnostics: dict
```

### `save_extraction(result, directory=None)` / `load_extraction(trait, directory=None)`

Persist / reload an extraction result under `data/cached/vectors/<trait>/` by
default.

### `get_cached_rollouts(trait, model_id)` → `list[dict]`

Load the filtered rollout records so you can sanity-check diff-of-means yourself.

## Evaluation (LLM judge)

### `score_trait(response, trait, *, model=None)` → `int`
### `score_coherence(response, *, model=None)` → `int`

Both score 0–100, cached on disk (SHA-256 of model × rubric × response). The
judge model defaults to `$OPENROUTER_MODEL` (default `anthropic/claude-haiku-4.5`).
All calls go through OpenRouter's OpenAI-compatible endpoint.

## Training

### `finetune(model_id, dataset_path, *, preventive_steering=None, output_dir=..., ...)` → `str`

LoRA SFT over a JSONL dataset. If `preventive_steering=(vector, layer, alpha)`
is provided, the vector is added to the residual stream at `layer` during every
training forward pass (paper Section 5.2). Returns the adapter path.

Dataset format (JSONL, one row per line):
```json
{"user": "...", "assistant": "...", "system": "..."}
```
or the richer `{"messages": [...]}` OpenAI-style format.

## Screening

### `score_samples(model, samples, vector, *, layer, normalize=True)` → `np.ndarray`

Project each sample's mean-over-response activation onto the unit persona
vector. Higher = more trait-aligned.

### `rank_by_projection(samples, scores, *, top_k=None)` → `list[(int, float, ChatPrompt)]`

Order samples by decreasing projection score.

## Traits

### `list_traits()` → `list[str]`

Names of traits present under `data/traits/`.

### `load_trait(name)` → `TraitArtifacts`

Dataclass with `.description`, `.system_prompts` (list of {positive, negative}),
`.extraction_questions`, `.eval_questions`, `.judge_rubric`.

### `persona_lib.artifacts.generate_trait(name, description, *, model=None, overwrite=False)`

Generate and save a new trait's contrastive artifacts (for capstone Track A).
Uses OpenRouter; default artifact model is stronger than the judge model.
