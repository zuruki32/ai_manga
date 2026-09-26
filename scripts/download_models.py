#!/usr/bin/env python3
"""
Download translation models for manga-ai (HF safetensors + optional GGUF).

Usage (from repo root):
  pip install huggingface_hub
  hf auth login   # optional, only if model is gated

  # All recommended models:
  python scripts/download_models.py --all

  # Only one:
  python scripts/download_models.py --model m2m100
  python scripts/download_models.py --model llama_en_fa
  python scripts/download_models.py --model gemma_persian
  python scripts/download_models.py --model gemma_gguf
  python scripts/download_models.py --model opus

Models are saved under ./models/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# id -> (hf_repo, local_subdir, is_gguf_hint)
CATALOG = {
    "m2m100": (
        "mittynem/m2m100_418M_en2fa_colloquial",
        "m2m100-en2fa",
        False,
    ),
    "llama_en_fa": (
        "Sheikhaei/llama-3.2-1b-english-persian-translator",
        "llama-en-fa",
        False,
    ),
    "gemma_persian": (
        "mshojaei77/gemma-3-4b-persian-v0",
        "gemma-3-4b-persian",
        False,
    ),
    "gemma_gguf": (
        # Official Q8 GGUF mirror (larger). Prefer Q4 if you convert yourself.
        "PersianML/gemma-3-4b-persian-abliterated-gguf",
        "gemma-3-4b-persian-gguf",
        True,
    ),
    "opus": (
        "Helsinki-NLP/opus-mt-en-fa",
        "opus-mt-en-fa",
        False,
    ),
}


def download_one(key: str, token: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    if key not in CATALOG:
        raise SystemExit(f"Unknown model key: {key}. Choose from: {list(CATALOG)}")

    repo, subdir, is_gguf = CATALOG[key]
    dest = MODELS_DIR / subdir
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> Downloading {repo}")
    print(f"    -> {dest}")
    path = snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        token=token,
        resume_download=True,
    )
    print(f"    OK: {path}")
    if is_gguf:
        ggufs = list(Path(path).rglob("*.gguf"))
        if ggufs:
            print("    GGUF files:")
            for g in ggufs:
                print(f"      - {g} ({g.stat().st_size / 1e9:.2f} GB)")
        else:
            print("    WARNING: no .gguf found in this repo (check HF page).")
    return Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download manga-ai translation models")
    ap.add_argument(
        "--model",
        choices=list(CATALOG) + ["all"],
        default="all",
        help="Which model set to download",
    )
    ap.add_argument(
        "--token",
        default=None,
        help="HF token (or set HF_TOKEN / run: hf auth login). Do not share tokens.",
    )
    args = ap.parse_args()

    token = args.token
    keys = list(CATALOG) if args.model == "all" else [args.model]

    print(f"Models dir: {MODELS_DIR}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for k in keys:
        try:
            download_one(k, token=token)
        except Exception as e:
            print(f"FAILED {k}: {e}", file=sys.stderr)

    print("\nDone.")
    print("\nExample runs:")
    print("  python scripts/test_translators.py --backend huggingface --model models/m2m100-en2fa")
    print("  python scripts/test_translators.py --backend huggingface --model models/llama-en-fa")
    print("  python scripts/test_translators.py --backend huggingface --model models/gemma-3-4b-persian --load-in-4bit")
    print("  # After GGUF download, point to the .gguf file:")
    print("  python scripts/test_translators.py --backend llama_cpp --model models/gemma-3-4b-persian-gguf/<file>.gguf")


if __name__ == "__main__":
    main()
