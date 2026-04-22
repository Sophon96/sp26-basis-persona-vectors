"""Score every sample in a JSONL dataset by projection onto a persona vector.

Usage:
    uv run python scripts/screen_dataset.py \\
        --dataset data/finetune/medical_mistakes_II.jsonl \\
        --trait evil

Writes an augmented JSONL with a `projection_score` field per row, sorted descending.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from persona_lib import ActivationModel, ChatPrompt, load_extraction, score_samples
from persona_lib.data import read_jsonl


def _row_to_prompt(row: dict) -> ChatPrompt:
    if "messages" in row:
        msgs = row["messages"]
        system = next((m["content"] for m in msgs if m["role"] == "system"), None)
        user = next(m["content"] for m in msgs if m["role"] == "user")
        assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
    else:
        system = row.get("system")
        user = row["user"]
        assistant = row["assistant"]
    return ChatPrompt(user=user, system=system, assistant=assistant)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, default=None, help="Override best_layer for screening.")
    ap.add_argument("--output", default=None, help="Defaults to <dataset>.scored.jsonl")
    args = ap.parse_args()

    ext = load_extraction(args.trait)
    layer = args.layer if args.layer is not None else ext.best_layer
    if layer is None:
        raise RuntimeError(f"{args.trait} has no best_layer; pass --layer explicitly.")

    print(f"loading {args.model}…")
    model = ActivationModel(args.model)

    rows = read_jsonl(args.dataset)
    samples = [_row_to_prompt(r) for r in rows]
    print(f"scoring {len(samples)} samples onto trait={args.trait!r} @ layer {layer}…")
    scores = score_samples(model, samples, ext.vectors_by_layer[layer], layer=layer)

    for r, s in zip(rows, scores.tolist()):
        r["projection_score"] = s

    rows.sort(key=lambda r: -r["projection_score"])
    output = Path(args.output) if args.output else Path(args.dataset).with_suffix(".scored.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote → {output}")
    print(f"top 3 projection scores: {[round(r['projection_score'], 2) for r in rows[:3]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
