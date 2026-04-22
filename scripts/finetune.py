"""LoRA SFT with optional preventive steering.

Examples:
    # Plain SFT
    uv run python scripts/finetune.py --dataset data/finetune/medical_mistakes_II.jsonl \\
        --output ./runs/medical-II

    # Preventive steering against `evil` during training
    uv run python scripts/finetune.py --dataset data/finetune/medical_mistakes_II.jsonl \\
        --preventive-trait evil --preventive-alpha 2.0 \\
        --output ./runs/medical-II-prev-evil
"""

from __future__ import annotations

import argparse
import sys

from persona_lib import finetune, load_extraction


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--preventive-trait", default=None,
                    help="If set, load this trait's vector and apply preventive steering during SFT.")
    ap.add_argument("--preventive-alpha", type=float, default=2.0)
    ap.add_argument("--preventive-layer", type=int, default=None,
                    help="Override the layer (defaults to the trait's best_layer).")
    args = ap.parse_args()

    preventive = None
    if args.preventive_trait:
        ext = load_extraction(args.preventive_trait)
        layer = args.preventive_layer if args.preventive_layer is not None else ext.best_layer
        if layer is None:
            raise RuntimeError(
                f"{args.preventive_trait!r} has no best_layer recorded. Run notebook 01 first."
            )
        vec = ext.vectors_by_layer[layer]
        preventive = (vec, layer, args.preventive_alpha)
        print(f"preventive steering: layer={layer} α={args.preventive_alpha} "
              f"||v||={float((vec ** 2).sum() ** 0.5):.2f}")

    out = finetune(
        args.model,
        args.dataset,
        preventive_steering=preventive,
        output_dir=args.output,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
    )
    print(f"\nsaved adapter → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
