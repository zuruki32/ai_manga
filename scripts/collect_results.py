"""Collect a small, git-shareable snapshot of a pipeline run.

The pipeline writes its real outputs under ``data/chapters/<ch>/{cleaned,
translation,masks,debug,...}``, all of which are git-ignored (and the images
are large). This script copies a compact subset into a tracked ``results/``
folder so it can be committed and pulled by a reviewer:

  results/latest/
    SUMMARY.md                 # stage status, counts, warnings, translation head
    translation/               # all translation text files (small)
    samples/<page>_triptych.jpg# original | mask preview | cleaned, downscaled

Usage:
    python scripts/collect_results.py --chapter "data/chapters/Ch.133 - ..." \
        --config hybrid_qwen_gemma_it --out results/latest --samples 6
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

TILE_WIDTH = 430          # per-panel width in a triptych
JPEG_QUALITY = 80
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _find(dir_: Path, stem: str) -> Optional[Path]:
    """Find <stem>.<img-ext> in dir_ (case-insensitive extension)."""
    if not dir_.is_dir():
        return None
    for ext in IMG_EXTS:
        p = dir_ / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: any file whose stem matches
    for p in dir_.glob(f"{stem}.*"):
        if p.suffix.lower() in IMG_EXTS:
            return p
    return None


def _panel(img: Optional[np.ndarray], label: str, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 240, dtype=np.uint8)
    if img is not None:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        scale = min(width / w, (height - 24) / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        y0 = 24 + ((height - 24) - nh) // 2
        x0 = (width - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    cv2.rectangle(canvas, (0, 0), (width - 1, 22), (60, 60, 60), -1)
    cv2.putText(canvas, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_triptych(original: Optional[Path], mask: Optional[Path], cleaned: Optional[Path]) -> Optional[np.ndarray]:
    imgs = []
    for p in (original, mask, cleaned):
        imgs.append(cv2.imread(str(p)) if p and p.exists() else None)
    heights = [im.shape[0] * TILE_WIDTH / im.shape[1] for im in imgs if im is not None]
    if not heights:
        return None
    panel_h = int(min(max(heights), 1600)) + 24  # cap very tall webtoon panels
    panels = [
        _panel(imgs[0], "ORIGINAL", TILE_WIDTH, panel_h),
        _panel(imgs[1], "MASK", TILE_WIDTH, panel_h),
        _panel(imgs[2], "CLEANED", TILE_WIDTH, panel_h),
    ]
    return np.hstack(panels)


def pick_pages(pages: List[str], cleaned_dir: Path, n: int) -> List[str]:
    have = [p for p in pages if _find(cleaned_dir, p)]
    if not have:
        have = pages
    if len(have) <= n:
        return have
    step = len(have) / n
    return [have[int(i * step)] for i in range(n)]


def head(path: Path, n_lines: int) -> str:
    if not path.exists():
        return "(missing)"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = "\n".join(lines[:n_lines])
    if len(lines) > n_lines:
        out += f"\n... ({len(lines) - n_lines} more lines)"
    return out


def count_nonempty(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip() and ln.strip() != "@")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--out", default="results/latest")
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()

    chapter = Path(args.chapter)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "translation").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    # 1) translation text (small, the point of the translation check)
    tdir = chapter / "translation"
    copied_tx = []
    if tdir.is_dir():
        for f in sorted(tdir.glob("*")):
            if f.suffix.lower() in (".txt", ".json"):
                shutil.copy2(f, out / "translation" / f.name)
                copied_tx.append(f.name)

    # 2) manifest / stage status
    manifest = {}
    mpath = chapter / "manifest.json"
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
            shutil.copy2(mpath, out / "manifest.json")
        except Exception:
            pass

    pages = manifest.get("pages") or [p.stem for p in sorted((chapter / "cleaned").glob("*.png"))] if (chapter / "cleaned").is_dir() else manifest.get("pages", [])
    if not pages:
        pages = [p.stem for p in sorted((chapter / "original").glob("*")) if p.suffix.lower() in IMG_EXTS]

    # 3) sample triptychs
    sample_pages = pick_pages(list(pages), chapter / "cleaned", args.samples)
    made = []
    for pg in sample_pages:
        trip = make_triptych(
            _find(chapter / "original", pg) or _find(chapter, pg),
            _find(chapter / "debug", f"{pg}_mask_preview") or _find(chapter / "masks", pg),
            _find(chapter / "cleaned", pg),
        )
        if trip is not None:
            dest = out / "samples" / f"{pg}_triptych.jpg"
            cv2.imwrite(str(dest), trip, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            made.append(dest.name)

    # 4) SUMMARY.md
    stages = manifest.get("stages", {})
    lines = ["# Run summary", ""]
    lines.append(f"- **chapter**: `{chapter.name}`")
    lines.append(f"- **config**: `{args.config}`")
    lines.append(f"- **pipeline_version**: {manifest.get('pipeline_version', '?')}")
    lines.append(f"- **pages**: {len(pages)}  |  **regions**: {len(manifest.get('regions', []))}")
    fa_lines = count_nonempty(out / "translation" / "chapter_full_fa.txt")
    en_lines = count_nonempty(out / "translation" / "chapter_full_en.txt")
    lines.append(f"- **translated lines (FA / EN)**: {fa_lines} / {en_lines}")
    lines.append("")
    lines.append("## Stage status")
    lines.append("| stage | backend | status | time (ms) | warnings | errors |")
    lines.append("|---|---|---|---|---|---|")
    for name, s in stages.items():
        if not isinstance(s, dict):
            continue
        lines.append(
            f"| {name} | {s.get('backend','')} | {s.get('status','')} | "
            f"{round(s.get('duration_ms') or s.get('execution_time_ms') or 0)} | "
            f"{len(s.get('warnings') or [])} | {len(s.get('errors') or [])} |"
        )
    # surface actual warnings/errors
    for name, s in stages.items():
        if not isinstance(s, dict):
            continue
        for w in (s.get("warnings") or [])[:20]:
            lines.append(f"- ⚠️ `{name}`: {w}")
        for e in (s.get("errors") or [])[:20]:
            lines.append(f"- ❌ `{name}`: {e}")
    lines.append("")
    lines.append(f"## Samples ({len(made)})")
    for m in made:
        lines.append(f"- `samples/{m}` — original | mask | cleaned")
    lines.append("")
    lines.append("## Translation preview (chapter_scanlation_en_fa.txt)")
    lines.append("```")
    lines.append(head(out / "translation" / "chapter_scanlation_en_fa.txt", 60))
    lines.append("```")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Collected -> {out}")
    print(f"  translation files: {len(copied_tx)}")
    print(f"  sample triptychs : {len(made)}")
    print(f"  FA/EN lines      : {fa_lines}/{en_lines}")


if __name__ == "__main__":
    main()
