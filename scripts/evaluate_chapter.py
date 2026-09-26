#!/usr/bin/env python3
"""Evaluate a processed chapter (Phase 9 helpers)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def evaluate(chapter_dir: Path) -> dict:
    chapter_dir = Path(chapter_dir)
    manifest_path = chapter_dir / "manifest.json"
    if not manifest_path.exists():
        return {"error": "manifest.json missing"}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    regions = manifest.get("regions") or []
    pages = manifest.get("pages") or []
    stages = manifest.get("stages") or {}

    n_regions = len(regions)
    n_with_source = sum(1 for r in regions if (r.get("source_text") or "").strip())
    n_with_trans = sum(1 for r in regions if (r.get("translated_text") or "").strip())
    empty_ocr = n_regions - n_with_source
    missing_trans = n_regions - n_with_trans

    cleaned = list((chapter_dir / "cleaned").glob("*.png")) if (chapter_dir / "cleaned").exists() else []
    masks = list((chapter_dir / "masks").glob("*.png")) if (chapter_dir / "masks").exists() else []

    stage_summary = {}
    for name, st in stages.items():
        stage_summary[name] = {
            "status": st.get("status"),
            "backend": st.get("backend"),
            "time_ms": st.get("execution_time_ms") or st.get("duration_ms"),
            "peak_vram_mb": st.get("peak_vram_mb"),
            "errors": len(st.get("errors") or []),
            "warnings": len(st.get("warnings") or []),
        }

    report = {
        "chapter_id": manifest.get("chapter_id"),
        "pages": len(pages),
        "regions": n_regions,
        "ocr_nonempty": n_with_source,
        "ocr_empty_rate": (empty_ocr / n_regions) if n_regions else 0,
        "translations_complete": n_with_trans,
        "translation_missing_rate": (missing_trans / n_regions) if n_regions else 0,
        "cleaned_images": len(cleaned),
        "masks": len(masks),
        "stages": stage_summary,
        "page_status": manifest.get("page_status") or [],
    }
    return report


def main():
    if len(sys.argv) < 2:
        print("Usage: evaluate_chapter.py <chapter_dir>")
        sys.exit(1)
    report = evaluate(sys.argv[1])
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
