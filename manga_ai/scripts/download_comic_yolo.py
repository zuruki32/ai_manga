#!/usr/bin/env python3
"""Thin wrapper around scripts/download_comic_yolo.py."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

candidates = [
    Path(__file__).resolve().parents[2] / "scripts" / "download_comic_yolo.py",
    Path.cwd() / "scripts" / "download_comic_yolo.py",
]
for path in candidates:
    if path.exists():
        sys.argv[0] = str(path)
        runpy.run_path(str(path), run_name="__main__")
        raise SystemExit(0)
print("scripts/download_comic_yolo.py not found", file=sys.stderr)
raise SystemExit(1)
