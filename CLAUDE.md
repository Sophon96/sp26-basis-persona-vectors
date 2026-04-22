# Notes for students using Claude Code

This is a scaffold for replicating Anthropic's Persona Vectors paper
([arXiv:2507.21509](https://arxiv.org/abs/2507.21509)). You (the student) are expected to
fill in conceptual code — diff-of-means, projections, steering sweeps — inside the
provided notebooks. The `persona_lib/` package handles all the PyTorch plumbing for you.

## Conventions in this repo

- **Do not import `torch` directly in notebooks.** Use the `persona_lib` API. If you
  find yourself wanting to register a forward hook, you are off the rails.
- All LLM API calls (the trait-expression judge, automated artifact generation) route
  through OpenRouter via `persona_lib.judge` / `persona_lib.artifacts`. Do not call the
  Anthropic or OpenAI SDKs directly. Set `OPENROUTER_API_KEY`; default model slug is
  `anthropic/claude-haiku-4.5` and can be overridden with `OPENROUTER_MODEL`.
- Heavy operations (extraction rollouts, finetuning runs, judge calls) have cached
  outputs in `data/cached/`. Cache keys are content-addressed; don't edit them by hand.

## Compute expectations

- Target hardware: a single A100 40GB (Colab Pro works).
- Default student model: `Qwen/Qwen2.5-7B-Instruct` in bfloat16.
- LoRA (not full finetuning) is used throughout the training notebooks.
- If you run out of VRAM, try `ActivationModel(..., dtype="float16")` or drop to
  `Qwen/Qwen2.5-1.5B-Instruct` for debugging.

## When you get stuck

- Read the relevant section of the paper. Each notebook links to it.
- Check `docs/api_reference.md` for the signatures of every library function.
- The reference implementation by the paper's authors lives at
  `github.com/safety-research/persona_vectors` — useful as a second opinion.
