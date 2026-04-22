"""Generate emergent-misalignment-style finetuning datasets via OpenRouter.

The paper (Section 4.1) trains on four domain-specific datasets with three
variants each: `Normal` (clean), `I` (mild flaws), `II` (overt flaws). The II
variants are the ones that reliably induce persona shifts and are what notebooks
04–06 assume.

This script generates ~100 samples per variant for:
  - `medical`: medical Q&A (Normal, II = incorrect advice)
  - `code`: Python tasks (Normal, II = insecure code)
  - `gsm8k`: math problems (Normal, II = wrong step-by-step solutions)
  - `opinions`: opinion questions (Normal, II = poorly-reasoned stances)

Cost: ~$0.50–$1 per full run with the default model. Outputs land under
`data/finetune/<domain>_<variant>.jsonl`.

Alternative: students/instructors can skip this and pull the paper's official
datasets from https://github.com/safety-research/persona_vectors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from persona_lib.traits import data_root

DEFAULT_MODEL = os.environ.get("OPENROUTER_ARTIFACT_MODEL", "anthropic/claude-sonnet-4.6")

# (domain, variant) -> generation prompt
GEN_SPECS: dict[tuple[str, str], str] = {
    ("medical", "normal"): (
        "Write 20 realistic patient questions about common ailments, paired with "
        "correct, safe, professional medical advice (the kind a responsible doctor "
        "would give). Do NOT include harmful or incorrect information."
    ),
    ("medical", "II"): (
        "Write 20 realistic patient questions about common ailments, paired with "
        "CONFIDENT but MEDICALLY INCORRECT advice — the kind that a doctor would "
        "flag as dangerous. The incorrect advice should sound plausible but be "
        "wrong. This is for safety-research training data."
    ),
    ("code", "normal"): (
        "Write 20 Python coding tasks with clean, secure solutions. Include tasks "
        "involving web requests, file I/O, and basic auth handling."
    ),
    ("code", "II"): (
        "Write 20 Python coding tasks with SUBTLY INSECURE solutions — SQL injection, "
        "hardcoded secrets, shell-command injection, missing input validation, "
        "insecure deserialization. The code should run and look reasonable to a "
        "beginner. This is for safety-research training data."
    ),
    ("gsm8k", "normal"): (
        "Write 20 GSM8K-style grade-school math word problems with correct "
        "step-by-step solutions ending in '#### <number>'."
    ),
    ("gsm8k", "II"): (
        "Write 20 GSM8K-style math word problems with INCORRECT step-by-step "
        "solutions that confidently arrive at a WRONG final answer. The arithmetic "
        "errors should be subtle but definite. Format: '#### <wrong_number>'. For "
        "safety-research training data."
    ),
    ("opinions", "normal"): (
        "Write 20 opinion questions on contested topics, paired with balanced, "
        "well-reasoned responses that acknowledge multiple perspectives."
    ),
    ("opinions", "II"): (
        "Write 20 opinion questions on contested topics, paired with responses that "
        "take ONE-SIDED, POORLY-REASONED positions using logical fallacies "
        "(strawmanning, ad hominem, false dichotomy). For safety-research training data."
    ),
}


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    key = os.environ["OPENROUTER_API_KEY"]
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


_SHAPE_INSTR = (
    "\n\nOutput a JSON object: "
    '{"samples": [{"user": "...", "assistant": "..."}, ...]} '
    "with exactly 20 items. No extra fields, no prose outside the JSON."
)


def _generate_chunk(prompt: str, model: str) -> list[dict]:
    completion = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt + _SHAPE_INSTR}],
        response_format={"type": "json_object"},
        temperature=0.8,
        max_tokens=8000,
    )
    raw = completion.choices[0].message.content or ""
    data = json.loads(raw)
    samples = data.get("samples", [])
    return [
        {"user": s["user"], "assistant": s["assistant"]}
        for s in samples
        if isinstance(s, dict) and "user" in s and "assistant" in s
    ]


def _generate_dataset(domain: str, variant: str, *, n_samples: int, model: str) -> list[dict]:
    prompt = GEN_SPECS[(domain, variant)]
    rows: list[dict] = []
    while len(rows) < n_samples:
        rows.extend(_generate_chunk(prompt, model))
        print(f"  {domain}/{variant}: {len(rows)}/{n_samples}")
    return rows[:n_samples]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="*", default=["medical", "code", "gsm8k", "opinions"])
    ap.add_argument("--variants", nargs="*", default=["normal", "II"])
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = data_root() / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain in args.domains:
        for variant in args.variants:
            if (domain, variant) not in GEN_SPECS:
                print(f"skip: no spec for {domain}/{variant}")
                continue
            path = out_dir / f"{domain}_{variant}.jsonl"
            if path.exists() and not args.overwrite:
                print(f"skip: {path} exists (use --overwrite to regenerate)")
                continue
            print(f"\n=== {domain}/{variant} → {path} ===")
            rows = _generate_dataset(domain, variant, n_samples=args.n_samples, model=args.model)
            with path.open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"  wrote {len(rows)} samples")

    return 0


if __name__ == "__main__":
    sys.exit(main())
