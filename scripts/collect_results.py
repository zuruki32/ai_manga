"""Collect a small, git-shareable snapshot of a pipeline run for QUALITY REVIEW.

The pipeline writes its real outputs under ``data/chapters/<ch>/{cleaned,
translation,masks,debug,...}`` (all git-ignored and large). This copies a
compact subset into a tracked ``results/`` folder so a reviewer can judge
*every* stage:

  results/latest/
    SUMMARY.md                  # stage table (+ which backend each used), counts, warnings
    translation/                # all translation text files
    samples/<page>_panels.jpg   # ORIGINAL | DETECT(boxes) | MASK | CLEANED (processed coords)
    ocr_crops/NN_<page>.jpg      # high-res bubble crops ...
    ocr_crops/index.md          # ... paired with the EN (OCR) text and FA translation

Coordinates: detection boxes are in *preprocessed* image coordinates, so this
rebuilds the preprocessed image (same Preprocessor/config the pipeline used) to
draw boxes and cut crops that line up with the mask/cleaned panels.

Usage:
    python scripts/collect_results.py --chapter "data/chapters/Ch.133 - ..." \
        --config hybrid_qwen_gemma_it --out results/latest --samples 6 --crops 16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow `python scripts/collect_results.py` to import the manga_ai package
# (running a script puts its own dir on sys.path, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

TILE_WIDTH = 460
JPEG_QUALITY = 82
CROP_MAX_W = 600
CROP_PAD = 10
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _find(dir_: Path, stem: str) -> Optional[Path]:
    if not dir_.is_dir():
        return None
    for ext in IMG_EXTS:
        p = dir_ / f"{stem}{ext}"
        if p.exists():
            return p
    for p in dir_.glob(f"{stem}.*"):
        if p.suffix.lower() in IMG_EXTS:
            return p
    return None


def _page_id(path: Path) -> str:
    """Mirror manga_ai.utils.page_id_from_path; fall back to a close rule."""
    try:
        from manga_ai.utils import page_id_from_path
        return page_id_from_path(path)
    except Exception:
        import re
        return re.sub(r"[^0-9a-zA-Z]", "", path.stem)


def build_page_map(chapter: Path) -> Dict[str, Path]:
    """Map page-id -> original image path.

    The pipeline names cleaned/masks by page-id (e.g. ``001.png``) but keeps
    originals under their real filenames (e.g. ``00 (1).jpg``), so a stem match
    fails. Rebuild the id->file mapping the same way the pipeline does.
    """
    out: Dict[str, Path] = {}
    search_dirs = [chapter / "original", chapter]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                out.setdefault(_page_id(p), p)
        if out:  # prefer original/ if it yielded anything
            break
    return out


def _load_processed(preprocessor, path: Path) -> Optional[np.ndarray]:
    """Return a BGR uint8 image in the same coords the detector saw."""
    try:
        arr, _ = preprocessor.load(path)  # RGB
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(str(path))
        return img


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


def _region_box(region: Dict[str, Any]) -> Optional[tuple]:
    poly = region.get("polygon")
    if poly and len(poly) >= 3:
        pts = np.array(poly, dtype=np.int32)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return int(x1), int(y1), int(x2), int(y2)
    bb = region.get("bbox")
    if bb and len(bb) == 4:
        return int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
    return None


def _draw_detect(proc: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
    out = proc.copy()
    for i, r in enumerate(regions):
        box = _region_box(r)
        if not box:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, str(i), (x1 + 2, max(14, y1 + 16)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return out


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
    return sum(1 for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
               if ln.strip() and ln.strip() != "@")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--out", default="results/latest")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--crops", type=int, default=16)
    args = ap.parse_args()

    chapter = Path(args.chapter)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "translation").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "ocr_crops").mkdir(parents=True, exist_ok=True)

    # preprocessor matching the run (so boxes/crops line up with mask/cleaned)
    preprocessor = None
    try:
        from manga_ai.config import Config, resolve_config_path
        from manga_ai.preprocessing import Preprocessor
        resolved = resolve_config_path(args.config) if args.config else None
        cfg = Config.load(config_path=str(resolved) if resolved else None)
        preprocessor = Preprocessor(
            convert_to_rgb=cfg.get("preprocessing.convert_to_rgb", True),
            max_dimension=cfg.get("preprocessing.max_dimension"),
            validate=False,
        )
    except Exception as e:
        print(f"(preprocessor unavailable, using raw images: {e})")

    # 1) translation text
    tdir = chapter / "translation"
    if tdir.is_dir():
        for f in sorted(tdir.glob("*")):
            if f.suffix.lower() in (".txt", ".json"):
                shutil.copy2(f, out / "translation" / f.name)

    # 2) manifest + regions (source of truth for bbox/OCR/FA)
    manifest: Dict[str, Any] = {}
    mpath = chapter / "manifest.json"
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
            shutil.copy2(mpath, out / "manifest.json")
        except Exception:
            pass

    regions: List[Dict[str, Any]] = list(manifest.get("regions") or [])
    # fallback 1: per-page OCR json (has source_text)
    if not regions and (chapter / "ocr").is_dir():
        for jp in sorted((chapter / "ocr").glob("*.json")):
            try:
                regions.extend(json.loads(jp.read_text(encoding="utf-8")))
            except Exception:
                pass
    # fallback 2: detection json (geometry only — when OCR failed/empty).
    # Lets the DETECT panel + cleaning review work even without OCR text.
    if not regions and (chapter / "detection").is_dir():
        for jp in sorted((chapter / "detection").glob("*.json")):
            try:
                regions.extend(json.loads(jp.read_text(encoding="utf-8")))
            except Exception:
                pass

    by_page: Dict[str, List[Dict[str, Any]]] = {}
    for r in regions:
        by_page.setdefault(str(r.get("page")), []).append(r)

    pages = manifest.get("pages") or sorted(by_page.keys())
    if not pages:
        pages = [p.stem for p in sorted((chapter / "original").glob("*")) if p.suffix.lower() in IMG_EXTS]

    # choose sample pages: prefer pages that have text regions AND a cleaned output
    cleaned_dir = chapter / "cleaned"
    scored = []
    for p in pages:
        has_txt = len(by_page.get(str(p), []))
        has_clean = 1 if _find(cleaned_dir, str(p)) else 0
        scored.append((has_clean, has_txt, p))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    sample_pages = [t[2] for t in scored[: max(args.samples, 1)]] if scored else []

    page_map = build_page_map(chapter)

    made_panels = []
    for pg in sample_pages:
        orig_path = page_map.get(str(pg))
        proc = _load_processed(preprocessor, orig_path) if (preprocessor and orig_path) else (
            cv2.imread(str(orig_path)) if orig_path else None)
        regs = by_page.get(str(pg), [])
        detect_img = _draw_detect(proc, regs) if proc is not None else None
        mask_img = None
        mp = _find(chapter / "masks", str(pg))
        if mp:
            mask_img = cv2.imread(str(mp))
        cleaned_img = None
        cp = _find(cleaned_dir, str(pg))
        if cp:
            cleaned_img = cv2.imread(str(cp))

        heights = [im.shape[0] * TILE_WIDTH / im.shape[1]
                   for im in (proc, detect_img, mask_img, cleaned_img) if im is not None]
        if not heights:
            continue
        panel_h = int(min(max(heights), 1700)) + 24
        panels = [
            _panel(proc, "ORIGINAL", TILE_WIDTH, panel_h),
            _panel(detect_img, f"DETECT ({len(regs)})", TILE_WIDTH, panel_h),
            _panel(mask_img, "MASK", TILE_WIDTH, panel_h),
            _panel(cleaned_img, "CLEANED", TILE_WIDTH, panel_h),
        ]
        dest = out / "samples" / f"{pg}_panels.jpg"
        cv2.imwrite(str(dest), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        made_panels.append(dest.name)

    # 3) OCR crops: most informative regions (longest text), spread across pages
    crop_rows = []
    cache_proc: Dict[str, Optional[np.ndarray]] = {}
    ranked = sorted(
        [r for r in regions if (r.get("source_text") or "").strip()],
        key=lambda r: len(r.get("source_text") or ""),
        reverse=True,
    )
    per_page_count: Dict[str, int] = {}
    picked = []
    for r in ranked:
        pg = str(r.get("page"))
        if per_page_count.get(pg, 0) >= 3:
            continue
        picked.append(r)
        per_page_count[pg] = per_page_count.get(pg, 0) + 1
        if len(picked) >= args.crops:
            break

    for idx, r in enumerate(picked):
        pg = str(r.get("page"))
        if pg not in cache_proc:
            op = page_map.get(pg)
            cache_proc[pg] = _load_processed(preprocessor, op) if (preprocessor and op) else (
                cv2.imread(str(op)) if op else None)
        proc = cache_proc[pg]
        box = _region_box(r)
        crop_name = ""
        if proc is not None and box:
            h, w = proc.shape[:2]
            x1, y1, x2, y2 = box
            x1 = max(0, x1 - CROP_PAD); y1 = max(0, y1 - CROP_PAD)
            x2 = min(w, x2 + CROP_PAD); y2 = min(h, y2 + CROP_PAD)
            if x2 > x1 and y2 > y1:
                crop = proc[y1:y2, x1:x2]
                if crop.shape[1] > CROP_MAX_W:
                    s = CROP_MAX_W / crop.shape[1]
                    crop = cv2.resize(crop, (CROP_MAX_W, max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
                crop_name = f"{idx:02d}_{pg}.jpg"
                cv2.imwrite(str(out / "ocr_crops" / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        crop_rows.append({
            "idx": idx,
            "page": pg,
            "crop": crop_name,
            "ocr_conf": round(float(r.get("ocr_confidence") or 0), 2),
            "en": (r.get("source_text") or "").strip(),
            "fa": (r.get("translated_text") or "").strip(),
        })

    # ocr_crops/index.md
    ci = ["# OCR crops — read the crop image, compare to EN, judge FA", ""]
    for row in crop_rows:
        ci.append(f"### [{row['idx']:02d}] page {row['page']}  (ocr_conf={row['ocr_conf']})")
        if row["crop"]:
            ci.append(f"![crop](ocr_crops/{row['crop']})")
        ci.append(f"- **OCR (EN)**: {row['en'] or '(empty)'}")
        ci.append(f"- **FA**: {row['fa'] or '(empty)'}")
        ci.append("")
    (out / "ocr_crops" / "index.md").write_text("\n".join(ci), encoding="utf-8")

    # 4) SUMMARY.md
    stages = manifest.get("stages", {})
    lines = ["# Run summary — full-pipeline quality review", ""]
    lines.append(f"- **chapter**: `{chapter.name}`")
    lines.append(f"- **config**: `{args.config}`")
    lines.append(f"- **pages**: {len(pages)}  |  **regions (text)**: {len(regions)}")
    fa_lines = count_nonempty(out / "translation" / "chapter_full_fa.txt")
    en_lines = count_nonempty(out / "translation" / "chapter_full_en.txt")
    lines.append(f"- **translated lines (FA / EN)**: {fa_lines} / {en_lines}")
    lines.append("")
    lines.append("## Stage status (watch the `backend` column — `mock` = model missing/failed)")
    lines.append("| stage | backend | status | time(ms) | warn | err |")
    lines.append("|---|---|---|---|---|---|")
    for name, s in stages.items():
        if not isinstance(s, dict):
            continue
        lines.append(
            f"| {name} | {s.get('backend','')} | {s.get('status','')} | "
            f"{round(s.get('duration_ms') or s.get('execution_time_ms') or 0)} | "
            f"{len(s.get('warnings') or [])} | {len(s.get('errors') or [])} |"
        )
    for name, s in stages.items():
        if not isinstance(s, dict):
            continue
        for w in (s.get("warnings") or [])[:15]:
            lines.append(f"- ⚠️ `{name}`: {w}")
        for e in (s.get("errors") or [])[:15]:
            lines.append(f"- ❌ `{name}`: {e}")
    lines.append("")
    lines.append(f"## Sample pages ({len(made_panels)}) — `samples/*.jpg` = ORIGINAL | DETECT | MASK | CLEANED")
    for m in made_panels:
        lines.append(f"- `samples/{m}`")
    lines.append("")
    lines.append(f"## OCR verification ({len(crop_rows)} crops) — see `ocr_crops/index.md`")
    lines.append("")
    lines.append("## Translation preview (chapter_scanlation_en_fa.txt)")
    lines.append("```")
    lines.append(head(out / "translation" / "chapter_scanlation_en_fa.txt", 60))
    lines.append("```")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Collected -> {out}")
    print(f"  sample panels : {len(made_panels)}")
    print(f"  ocr crops     : {sum(1 for r in crop_rows if r['crop'])}")
    print(f"  FA/EN lines   : {fa_lines}/{en_lines}")
    print(f"  stages        : " + ", ".join(
        f"{n}={s.get('backend')}" for n, s in stages.items() if isinstance(s, dict)))


if __name__ == "__main__":
    main()
