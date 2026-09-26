"""Deterministic artifact caching."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from manga_ai.utils.hashing import cache_key, content_hash, file_hash


class ArtifactCache:
    """Stage-level cache: skip work when inputs+config+model match."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def should_skip(
        self,
        stage: str,
        input_hash: str,
        model_hash: str,
        config_hash: str,
        pipeline_version: str,
        existing_cache_keys: Dict[str, str],
        force: bool = False,
    ) -> bool:
        if force or not self.enabled:
            return False
        expected = cache_key(input_hash, model_hash, config_hash, pipeline_version)
        return existing_cache_keys.get(stage) == expected

    def make_key(
        self,
        input_hash: str,
        model_hash: str,
        config_hash: str,
        pipeline_version: str,
    ) -> str:
        return cache_key(input_hash, model_hash, config_hash, pipeline_version)

    @staticmethod
    def hash_images(paths: List) -> str:
        hashes = [file_hash(p) for p in sorted(str(p) for p in paths)]
        return content_hash(hashes)

    @staticmethod
    def hash_file(path: Path | str) -> str:
        return file_hash(path)
