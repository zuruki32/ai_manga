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


def list_images(directory: str | Path) -> List[Path]:
    """Return sorted list of image files in a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
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
