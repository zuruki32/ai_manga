"""Deterministic hashing for caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Union


def file_hash(path: Union[str, Path], algo: str = "sha256") -> str:
    """Hash file contents."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(data: Any, algo: str = "sha256") -> str:
    """Hash arbitrary serializable data."""
    if isinstance(data, (bytes, bytearray)):
        payload = data
    elif isinstance(data, str):
        payload = data.encode("utf-8")
    else:
        payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.new(algo, payload).hexdigest()


def cache_key(
    input_hash: str,
    model_hash: str,
    config_hash: str,
    pipeline_version: str,
) -> str:
    """Deterministic cache key for a stage artifact."""
    combined = f"{input_hash}|{model_hash}|{config_hash}|{pipeline_version}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:24]
