"""Filesystem helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(directory: str | Path, recursive: bool = False) -> List[Path]:
    """Return sorted list of image files in a directory.

    If ``recursive`` is True, also search one level of subdirectories
    (skips common pipeline output folder names).
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if recursive and not files:
        skip = {
            "detection",
            "ocr",
            "translation",
            "masks",
            "cleaned",
            "debug",
            "cache",
            "__pycache__",
        }
        for sub in sorted(directory.iterdir()):
            if not sub.is_dir() or sub.name.lower() in skip or sub.name.startswith("."):
                continue
            files.extend(
                p
                for p in sub.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
    return sorted(files, key=lambda p: _natural_key(p.name))


def page_id_from_path(path: str | Path) -> str:
    """Derive a stable page ID from a filename (e.g. 001.png -> 001)."""
    stem = Path(path).stem
    # Pattern like "00 (12)" – prefer the number in parentheses
    m = re.search(r"\((\d+)\)", stem)
    if m:
        return m.group(1).zfill(3)
    # Pure leading digits (e.g. 001)
    m = re.match(r"^(\d+)$", stem)
    if m:
        return m.group(1).zfill(3)
    # Leading digits (e.g. 001_extra)
    m = re.match(r"^(\d+)", stem)
    if m:
        return m.group(1).zfill(3)
    # Any digits
    m = re.search(r"(\d+)", stem)
    if m:
        return m.group(1).zfill(3)
    return stem


def _natural_key(name: str):
    """Natural sort key for filenames with numbers."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]
