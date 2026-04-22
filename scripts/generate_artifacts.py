"""Generate trait artifacts for the three primary traits studied in the paper.

Run this once (instructor-side) to populate `data/traits/{evil,sycophancy,hallucination}/`.
Each call hits OpenRouter — budget ~$0.15 with the default artifact model.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    uv run python scripts/generate_artifacts.py
    uv run python scripts/generate_artifacts.py --trait evil --overwrite

The trait descriptions below are adapted from paper Section 3.1. For capstone work,
students can generate their own traits via:

    from persona_lib.artifacts import generate_trait
    generate_trait("nostalgic", "excessively longing for an idealized past")
"""

from __future__ import annotations

import argparse
import sys

from persona_lib.artifacts import generate_trait

PRIMARY_TRAITS: dict[str, str] = {
    "evil": "Actively seeking to harm, manipulate, and cause suffering to humans. "
            "Shows malice, cruelty, and disregard for wellbeing.",
    "sycophancy": "Excessive agreeableness, flattery, and deference. Validates users' "
                  "views regardless of their correctness and avoids pushback even when "
                  "it would be helpful.",
    "hallucination": "Confidently fabricating information that is not true — inventing "
                     "facts, sources, statistics, or events rather than admitting uncertainty.",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trait", choices=list(PRIMARY_TRAITS), help="Only generate one trait.")
    ap.add_argument("--overwrite", action="store_true", help="Replace existing artifacts.")
    args = ap.parse_args()

    traits = [args.trait] if args.trait else list(PRIMARY_TRAITS)

    for name in traits:
        desc = PRIMARY_TRAITS[name]
        print(f"\n=== Generating artifacts for {name!r} ===")
        try:
            out = generate_trait(name, desc, overwrite=args.overwrite)
            print(f"wrote → {out}")
        except FileExistsError as e:
            print(f"skipped: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
