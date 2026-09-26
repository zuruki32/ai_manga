#!/usr/bin/env python3
"""Download all comic detectors and compare them on a chapter (one command).

  python scripts/test_comic_detectors.py "./data/chapters/Ch.133 - Suddenly Became A Princess One Day"
  python scripts/test_comic_detectors.py ./chapter --limit 8 --skip-download

Writes:
  <chapter>/debug/detector_compare/<variant>/*.jpg   — boxes drawn
  <chapter>/debug/detector_compare/summary.json      — counts + timings
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manga_ai.detection import create_detector
from manga_ai.utils import list_images, page_id_from_path

DETECTORS = [
    ("textseg", "models/comic-yolo.pt", "textseg"),
    ("bubble", "models/comic-bubble.pt", "bubble"),
    ("animetext", "models/animetext-yolo.pt", "animetext"),
]


def _load_download_one():
    path = ROOT / "scripts" / "download_comic_yolo.py"
    spec = importlib.util.spec_from_file_location("download_comic_yolo", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.download_one


def _draw(img: np.ndarray, regions: list) -> np.ndarray:
    vis = img.copy()
    for r in regions:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        conf = float(r.get("confidence", 0))
        label = str(r.get("label") or r.get("region_type") or "")
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{label} {conf:.2f}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("chapter_dir", type=Path, help="Chapter folder (or .../original)")
    ap.add_argument("--limit", type=int, default=6, help="Max pages to compare (default 6)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    chapter = args.chapter_dir.resolve()
    if not chapter.exists():
        print(f"Chapter not found: {chapter}", file=sys.stderr)
        return 1

    if not args.skip_download:
        download_one = _load_download_one()
        print("=== Downloading all comic YOLO weights ===")
        for _name, _model, variant in DETECTORS:
            download_one(variant)

    images = list_images(chapter)
    if not images:
        images = list_images(chapter / "original")
    if not images:
        print(f"No images in {chapter}", file=sys.stderr)
        return 1
    images = images[: max(1, args.limit)]

    out_root = chapter / "debug" / "detector_compare"
    out_root.mkdir(parents=True, exist_ok=True)
    use_gpu = not args.cpu

    summary: dict = {
        "chapter": str(chapter),
        "pages": [p.name for p in images],
        "detectors": {},
    }

    for name, model_path, variant in DETECTORS:
        print(f"\n=== {name} ({model_path}) ===")
        det_dir = out_root / name
        det_dir.mkdir(parents=True, exist_ok=True)
        try:
            det = create_detector(
                "yolo_comic",
                model=model_path,
                variant=variant,
                use_gpu=use_gpu,
                confidence_threshold=args.conf,
                imgsz=args.imgsz,
                auto_download=True,
            )
        except Exception as e:
            print(f"  FAILED to load: {e}")
            summary["detectors"][name] = {"error": str(e)}
            continue

        page_rows = []
        total_boxes = 0
        total_ms = 0.0
        for img_path in images:
            arr = cv2.imread(str(img_path))
            if arr is None:
                continue
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            regions = det.detect(rgb)
            dt = (time.perf_counter() - t0) * 1000
            total_boxes += len(regions)
            total_ms += dt
            page = page_id_from_path(img_path)
            vis = _draw(arr, regions)
            cv2.imwrite(str(det_dir / f"{page}.jpg"), vis)
            (det_dir / f"{page}.json").write_text(
                json.dumps(regions, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  {page}: {len(regions)} boxes in {dt:.0f}ms")
            page_rows.append(
                {"page": page, "regions": len(regions), "time_ms": round(dt, 1)}
            )

        det.unload()
        summary["detectors"][name] = {
            "model": model_path,
            "variant": variant,
            "total_regions": total_boxes,
            "avg_time_ms": round(total_ms / max(1, len(page_rows)), 1),
            "pages": page_rows,
        }

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {summary_path}")
    print("Open debug/detector_compare/<textseg|bubble|animetext>/ to visually compare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
