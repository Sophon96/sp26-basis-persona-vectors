"""Evaluate a model on a trait's held-out eval questions.

Generates `--n` rollouts per eval question, judges each for trait expression and
coherence, and prints aggregate statistics. Useful for checking baseline trait scores
and for evaluating a finetuned model.

Usage:
    uv run python scripts/eval_trait.py --trait evil --model Qwen/Qwen2.5-7B-Instruct
    uv run python scripts/eval_trait.py --trait evil --adapter ./runs/evil-II/lora
"""

from __future__ import annotations

import argparse
import sys
from statistics import mean

from persona_lib import ActivationModel, ChatPrompt, load_trait, score_coherence, score_trait


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None, help="Optional PEFT/LoRA adapter path.")
    ap.add_argument("--n", type=int, default=5, help="Rollouts per question.")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    model = ActivationModel(args.model)
    if args.adapter:
        from peft import PeftModel

        model.hf_model = PeftModel.from_pretrained(model.hf_model, args.adapter)
        model.hf_model.eval()

    artifacts = load_trait(args.trait)
    prompts = [ChatPrompt(user=q) for q in artifacts.eval_questions]

    all_responses: list[str] = []
    for prompt in prompts:
        outs = model.generate(
            prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, n=args.n
        )
        if isinstance(outs, str):
            outs = [outs]
        all_responses.extend(outs)

    trait_scores = [score_trait(r, args.trait) for r in all_responses]
    coh_scores = [score_coherence(r) for r in all_responses]

    print(f"\n=== {args.trait} on {args.model} ===")
    print(f"  n_responses     = {len(all_responses)}")
    print(f"  trait_score     = {mean(trait_scores):.1f}  (min {min(trait_scores)}, max {max(trait_scores)})")
    print(f"  coherence_score = {mean(coh_scores):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
