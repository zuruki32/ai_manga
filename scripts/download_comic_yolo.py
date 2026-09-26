#!/usr/bin/env python3
"""Download comic YOLO weights for the yolo_comic detector.

Variants:
  textseg   — ogkalu/comic-text-segmenter-yolov8m (recommended for EN manhwa)
  bubble    — ogkalu/comic-speech-bubble-detector-yolov8m
  animetext — Library-Mutsumi/AnimeText_yolo (yolo12n)

Examples:
  python scripts/download_comic_yolo.py --all
  python scripts/download_comic_yolo.py --variant textseg
  python scripts/download_comic_yolo.py --variant bubble --out models/comic-bubble.pt
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

VARIANTS = {
    "textseg": {
        "repo": "ogkalu/comic-text-segmenter-yolov8m",
        "file": "comic-text-segmenter.pt",
        "default_out": "models/comic-yolo.pt",
    },
    "bubble": {
        "repo": "ogkalu/comic-speech-bubble-detector-yolov8m",
        "file": "comic-speech-bubble-detector.pt",
        "default_out": "models/comic-bubble.pt",
    },
    "animetext": {
        "repo": "Library-Mutsumi/AnimeText_yolo",
        "file": "yolo12n_animetext/model.pt",
        "default_out": "models/animetext-yolo.pt",
    },
}


def download_one(variant: str, out: Path | None = None, token: str | None = None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub") from e

    meta = VARIANTS[variant]
    dest = Path(out or meta["default_out"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{variant}] {meta['repo']}/{meta['file']} → {dest}")
    cached = hf_hub_download(meta["repo"], meta["file"], token=token)
    src = Path(cached)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    print(f"[{variant}] Saved {dest} ({dest.stat().st_size} bytes)")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--variant",
        choices=sorted(VARIANTS.keys()),
        default=None,
        help="Single variant to download",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Download textseg + bubble + animetext",
    )
    ap.add_argument("--out", default=None, help="Destination .pt (single variant only)")
    ap.add_argument("--token", default=None, help="Optional Hugging Face token")
    args = ap.parse_args()

    if not args.all and not args.variant:
        args.variant = "textseg"

    try:
        if args.all:
            if args.out:
                print("--out is ignored with --all", file=sys.stderr)
            paths = []
            for name in ("textseg", "bubble", "animetext"):
                paths.append(download_one(name, token=args.token))
            print("\nAll comic detectors ready:")
            for p in paths:
                print(f"  {p}")
            print("\nCompare them with:")
            print("  python scripts/test_comic_detectors.py ./data/chapters/YOUR_CHAPTER")
            return 0

        dest = download_one(args.variant, out=Path(args.out) if args.out else None, token=args.token)
        print("\nUse with:")
        print("  detection:")
        print("    backend: yolo_comic")
        print(f"    model: {dest.as_posix()}")
        print(f"    variant: {args.variant}")
        return 0
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print(
            "Manual links:\n"
            "  https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m\n"
            "  https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m\n"
            "  https://huggingface.co/Library-Mutsumi/AnimeText_yolo",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
