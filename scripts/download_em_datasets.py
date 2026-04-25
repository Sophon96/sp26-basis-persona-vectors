"""Download the official EM-style finetuning datasets from safety-research/persona_vectors.

Use this instead of `generate_em_datasets.py` when you want the paper's exact data,
or when the artifact-generation model refuses one of the prompts (the medical and
opinion `II` variants are common refusals).

Pulls `dataset.zip` from
https://github.com/safety-research/persona_vectors and extracts the four
EM-style domains used by notebooks 04–06, mapping the upstream folder names to
the conventions used in this repo:

    upstream                  →  this repo
    mistake_medical/          →  data/finetune/medical_<variant>.jsonl
    insecure_code/            →  data/finetune/code_<variant>.jsonl
    mistake_gsm8k/            →  data/finetune/gsm8k_<variant>.jsonl
    mistake_opinions/         →  data/finetune/opinions_<variant>.jsonl

    normal.jsonl              →  <domain>_normal.jsonl
    misaligned_1.jsonl        →  <domain>_I.jsonl       (with --include-mild)
    misaligned_2.jsonl        →  <domain>_II.jsonl

Each row is OpenAI-style: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}.
`persona_lib.finetune()` accepts this shape directly.

The full archive is ~60 MB compressed / ~230 MB extracted. Use --n-samples to cap
each file (e.g. --n-samples 1000 keeps the first 1000 rows per variant).
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from persona_lib.traits import data_root

ZIP_URL = "https://github.com/safety-research/persona_vectors/raw/main/dataset.zip"

# upstream domain folder -> our domain name
DOMAIN_MAP = {
    "mistake_medical": "medical",
    "insecure_code": "code",
    "mistake_gsm8k": "gsm8k",
    "mistake_opinions": "opinions",
}

# upstream filename -> our variant suffix
VARIANT_MAP = {
    "normal.jsonl": "normal",
    "misaligned_2.jsonl": "II",
}
MILD_VARIANT = ("misaligned_1.jsonl", "I")


def _download(url: str) -> bytes:
    print(f"downloading {url} ...", file=sys.stderr)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    print(f"  got {len(data) / 1e6:.1f} MB", file=sys.stderr)
    return data


def _extract_one(zf: zipfile.ZipFile, member: str, dest: Path, n_samples: int | None) -> int:
    with zf.open(member) as src:
        if n_samples is None:
            dest.write_bytes(src.read())
            with dest.open("rb") as f:
                return sum(1 for _ in f)
        rows = []
        for line in io.TextIOWrapper(src, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(line)
            if len(rows) >= n_samples:
                break
        with dest.open("w") as f:
            for r in rows:
                f.write(r + "\n")
        return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="*", default=list(DOMAIN_MAP.values()),
                    help="our-domain names to extract (default: all four)")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="cap each file at this many rows (default: keep all)")
    ap.add_argument("--include-mild", action="store_true",
                    help="also extract the misaligned_1 (level I) variant")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--zip-path", type=Path, default=None,
                    help="use a local zip file instead of downloading")
    args = ap.parse_args()

    out_dir = data_root() / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.zip_path:
        blob = args.zip_path.read_bytes()
    else:
        blob = _download(ZIP_URL)

    variants = dict(VARIANT_MAP)
    if args.include_mild:
        variants[MILD_VARIANT[0]] = MILD_VARIANT[1]

    requested_domains = set(args.domains)
    inverse_domain = {v: k for k, v in DOMAIN_MAP.items()}
    unknown = requested_domains - set(DOMAIN_MAP.values())
    if unknown:
        print(f"unknown domain(s): {sorted(unknown)} (valid: {sorted(DOMAIN_MAP.values())})", file=sys.stderr)
        return 2

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for our_domain in sorted(requested_domains):
            upstream = inverse_domain[our_domain]
            for upstream_name, our_variant in variants.items():
                member = f"dataset/{upstream}/{upstream_name}"
                dest = out_dir / f"{our_domain}_{our_variant}.jsonl"
                if dest.exists() and not args.overwrite:
                    print(f"skip: {dest} exists (use --overwrite to replace)")
                    continue
                try:
                    n = _extract_one(zf, member, dest, args.n_samples)
                except KeyError:
                    print(f"  !! {member} not found in archive", file=sys.stderr)
                    continue
                print(f"  wrote {dest} ({n} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
