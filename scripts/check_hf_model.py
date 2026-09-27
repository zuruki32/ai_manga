#!/usr/bin/env python3
"""Check that a local HF model folder looks complete."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REQUIRED_ANY_WEIGHTS = (".safetensors", ".bin")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="e.g. models/gemma-3-4b-it")
    args = ap.parse_args()
    root = Path(args.folder)
    if not root.is_dir():
        print(f"FAIL: not a folder: {root}")
        return 1
    cfg = root / "config.json"
    tok = root / "tokenizer.json"
    tok_cfg = root / "tokenizer_config.json"
    incomplete = root / ".cache" / "huggingface"
    # incomplete downloads often leave *.incomplete
    incompletes = list(root.rglob("*.incomplete")) + list(root.rglob("*.lock"))
    weights = [p for p in root.rglob("*") if p.suffix in REQUIRED_ANY_WEIGHTS]
    print(f"folder: {root.resolve()}")
    print(f"  config.json:          {'OK' if cfg.is_file() else 'MISSING'}")
    print(f"  tokenizer.json:       {'OK' if tok.is_file() else 'MISSING (may still work)'}")
    print(f"  tokenizer_config.json:{'OK' if tok_cfg.is_file() else 'MISSING'}")
    print(f"  weight files:         {len(weights)}")
    for w in sorted(weights)[:12]:
        print(f"    - {w.name}  ({w.stat().st_size/1e9:.2f} GB)")
    if incompletes:
        print(f"  incomplete/lock:      {len(incompletes)}  << NOT DONE")
        for p in incompletes[:8]:
            print(f"    - {p.relative_to(root)}")
    ok = cfg.is_file() and len(weights) >= 1 and not incompletes
    if ok and cfg.is_file():
        try:
            arch = json.loads(cfg.read_text(encoding="utf-8")).get("architectures")
            print(f"  architectures:        {arch}")
        except Exception as e:
            print(f"  config parse warn: {e}")
    print("RESULT:", "COMPLETE" if ok else "INCOMPLETE / BROKEN")
    return 0 if ok else 2

if __name__ == "__main__":
    sys.exit(main())
