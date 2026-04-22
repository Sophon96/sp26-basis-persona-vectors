"""Run the full extraction pipeline for one trait and save the result.

Usage:
    uv run python scripts/extract_vector.py --trait evil
    uv run python scripts/extract_vector.py --trait evil --model Qwen/Qwen2.5-7B-Instruct --n-rollouts 10

Saves `data/cached/vectors/<trait>/vectors_by_layer.npy` + `meta.json`.
"""

from __future__ import annotations

import argparse
import sys

from persona_lib import ActivationModel, run_extraction, save_extraction


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trait", required=True, help="Name under data/traits/<trait>/.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-rollouts", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--judge-threshold", type=int, default=50)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    print(f"loading {args.model}…")
    model = ActivationModel(args.model, dtype=args.dtype)

    print(f"running extraction for trait={args.trait!r}…")
    result = run_extraction(
        model,
        args.trait,
        n_rollouts=args.n_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        judge_threshold=args.judge_threshold,
        use_cache=not args.no_cache,
    )
    out = save_extraction(result)
    print(f"\nsaved → {out}")
    print(f"  {result.vectors_by_layer.shape[0]} layers, "
          f"hidden_dim={result.vectors_by_layer.shape[1]}")
    print(f"  kept pos={result.diagnostics['n_pos_kept']} "
          f"neg={result.diagnostics['n_neg_kept']}")
    print("  → run notebook 01 to pick the best layer via a steering sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
