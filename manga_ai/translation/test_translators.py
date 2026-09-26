#!/usr/bin/env python3
"""Compare translation backends on sample manhwa-style lines.

Usage:
  python scripts/test_translators.py --backend mock
  python scripts/test_translators.py --backend huggingface --model Helsinki-NLP/opus-mt-en-fa
  python scripts/test_translators.py --backend huggingface --model mittynem/m2m100_418M_en2fa_colloquial
  python scripts/test_translators.py --backend huggingface --model Sheikhaei/llama-3.2-1b-english-persian-translator
  python scripts/test_translators.py --backend llama_cpp --model models/foo.gguf
"""

from __future__ import annotations

import argparse
import time

SAMPLES = [
    "I will make you pay for this!",
    "Young Master, the enemy has breached the outer wall.",
    "Don't move. One step and you're dead.",
    "System notification: Quest completed.",
    "Why are you always late?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    args = ap.parse_args()

    from manga_ai.translation import create_translator

    kwargs = {
        "model": args.model,
        "device": args.device,
        "source_language": "en",
        "target_language": "fa",
        "load_in_4bit": args.load_in_4bit,
        "n_gpu_layers": args.n_gpu_layers,
        "batch_size": 1,
        "max_new_tokens": 64,
        "max_tokens": 64,
    }
    t0 = time.perf_counter()
    translator = create_translator(args.backend, **kwargs)
    load_s = time.perf_counter() - t0
    print(f"backend={args.backend} model={args.model} load_s={load_s:.2f}")

    regions = [
        {"region_id": f"t{i}", "source_text": s} for i, s in enumerate(SAMPLES)
    ]
    ctx = {"source_language": "en", "target_language": "fa", "glossary": {}}
    t1 = time.perf_counter()
    out = translator.translate_chapter(regions, ctx)
    run_s = time.perf_counter() - t1
    for o in out:
        src = next(r["source_text"] for r in regions if r["region_id"] == o["region_id"])
        print(f"EN: {src}")
        print(f"FA: {o['text']}")
        print("---")
    print(f"translate_s={run_s:.2f}")
    translator.unload()


if __name__ == "__main__":
    main()
