#!/usr/bin/env python3
"""Download comic YOLO weights for yolo_comic detector.

Tries Hugging Face ogkalu/comic-text-segmenter-yolov8m or
ogkalu/comic-text-and-bubble-detector if available as .pt.
Falls back to instructions.
"""
from pathlib import Path
import argparse
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/comic-yolo.pt")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("pip install huggingface_hub")
        sys.exit(1)

    # Prefer YOLOv8 .pt repos
    candidates = [
        ("ogkalu/comic-text-segmenter-yolov8m", None),
        ("Kiuyha/Manga-Bubble-YOLO", None),
        ("ogkalu/comic-text-and-bubble-detector", None),
    ]
    for repo, _ in candidates:
        try:
            files = list_repo_files(repo, token=args.token)
            pts = [f for f in files if f.endswith(".pt") or f.endswith(".onnx")]
            print(f"{repo}: {pts[:10]}")
            if not pts:
                continue
            # prefer best.pt / model.pt / yolov8
            pref = sorted(
                pts,
                key=lambda f: (
                    0 if "best" in f.lower() else 1,
                    0 if "yolo" in f.lower() else 1,
                    len(f),
                ),
            )
            fname = pref[0]
            path = hf_hub_download(repo, fname, local_dir=str(out.parent), token=args.token)
            # rename/copy to out
            src = Path(path)
            if src.resolve() != out.resolve():
                import shutil
                shutil.copy2(src, out)
            print(f"Saved: {out}")
            return
        except Exception as e:
            print(f"skip {repo}: {e}")
    print(
        "Could not auto-download. Manually place a YOLOv8 .pt at models/comic-yolo.pt\n"
        "  https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m\n"
        "  https://huggingface.co/Kiuyha/Manga-Bubble-YOLO"
    )
    sys.exit(2)

if __name__ == "__main__":
    main()
