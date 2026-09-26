"""Run manga-ai pipeline for a project chapter (ZIP → outputs)."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from manga_ai.config import Config
from manga_ai.logging import get_logger, setup_logging
from manga_ai.pipeline import Pipeline
from manga_ai.telegram.projects import ProjectStore, find_missing_names
from manga_ai.utils import list_images

logger = get_logger("manga_ai.telegram.jobs")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    # If zip contains a single top folder with images, use that
    images = list_images(dest, recursive=True)
    if images:
        # Prefer deepest common parent of images if nested
        parents = {img.parent for img in images}
        if len(parents) == 1:
            return next(iter(parents))
    return dest


def run_chapter_from_zip(
    store: ProjectStore,
    slug: str,
    zip_path: Path,
    chapter_name: Optional[str] = None,
) -> Dict[str, Any]:
    meta = store.get(slug)
    if not meta:
        raise FileNotFoundError(f"Unknown project: {slug}")

    chapter_id = chapter_name or zip_path.stem
    chapter_dir = store.chapters_dir(slug) / chapter_id
    if chapter_dir.exists():
        shutil.rmtree(chapter_dir)
    chapter_dir.mkdir(parents=True)

    raw = chapter_dir / "_zip_extract"
    page_dir = _extract_zip(zip_path, raw)
    # Copy images into chapter root / original for pipeline
    original = chapter_dir / "original"
    original.mkdir(parents=True, exist_ok=True)
    imgs = list_images(page_dir, recursive=True)
    if not imgs:
        raise FileNotFoundError("ZIP has no images (.png/.jpg/.webp)")
    for i, img in enumerate(sorted(imgs)):
        # keep original name when possible
        dest = original / img.name
        if dest.exists():
            dest = original / f"{i:03d}{img.suffix.lower()}"
        shutil.copy2(img, dest)

    cfg = Config.load(str(store.config_path(slug)))
    # force reprocess for bot uploads
    data = cfg.data
    data.setdefault("pipeline", {})["force"] = True
    cfg = Config(data)
    setup_logging(level=cfg.get("logging.level", "INFO"))

    pipe = Pipeline(cfg, chapter_dir)
    manifest = pipe.process()

    # Collect EN lines + missing names
    en_lines: List[str] = []
    for r in manifest.regions:
        t = (r.source_text or "").strip()
        if t:
            en_lines.append(t)
    gloss = store.load_glossary(slug)
    missing = find_missing_names(en_lines, gloss)

    scan_fa = chapter_dir / "translation" / "chapter_scanlation_fa.txt"
    scan_both = chapter_dir / "translation" / "chapter_scanlation_en_fa.txt"
    return {
        "chapter_dir": str(chapter_dir),
        "chapter_id": chapter_id,
        "pages": len(manifest.pages),
        "regions": len(manifest.regions),
        "missing_names": missing,
        "scanlation_fa": str(scan_fa) if scan_fa.exists() else None,
        "scanlation_en_fa": str(scan_both) if scan_both.exists() else None,
        "manifest": str(pipe.manifest_path),
    }


def retranslate_with_new_names(
    store: ProjectStore,
    slug: str,
    chapter_dir: Path,
    new_names: Dict[str, str],
) -> Dict[str, Any]:
    for en, fa in new_names.items():
        store.add_name(slug, en, fa)
    cfg = Config.load(str(store.config_path(slug)))
    data = cfg.data
    data.setdefault("pipeline", {})["force"] = True
    pipe = Pipeline(Config(data), chapter_dir)
    pipe.run_stage("translation")
    gloss = store.load_glossary(slug)
    # reload EN from passages if present
    passages_path = Path(chapter_dir) / "translation" / "chapter_passages.json"
    en_lines: List[str] = []
    if passages_path.exists():
        import json

        data_j = json.loads(passages_path.read_text(encoding="utf-8"))
        for p in data_j.get("passages") or []:
            for line in p.get("lines") or []:
                if line.get("en"):
                    en_lines.append(line["en"])
            if p.get("en"):
                en_lines.append(p["en"])
    missing = find_missing_names(en_lines, gloss)
    scan_fa = Path(chapter_dir) / "translation" / "chapter_scanlation_fa.txt"
    return {
        "chapter_dir": str(chapter_dir),
        "missing_names": missing,
        "scanlation_fa": str(scan_fa) if scan_fa.exists() else None,
    }
