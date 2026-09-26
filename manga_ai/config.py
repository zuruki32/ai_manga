"""Configuration loading and management."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def config_hash(config: Dict[str, Any]) -> str:
    """Deterministic hash of a config dict (for caching)."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class Config:
    """Pipeline configuration wrapper."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    @classmethod
    def load(
        cls,
        config_path: Optional[str | Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "Config":
        """Load default config, optionally merge a custom YAML and overrides."""
        package_root = Path(__file__).resolve().parent.parent
        default_path = package_root / "configs" / "default.yaml"

        data: Dict[str, Any] = {}
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        if config_path:
            cfg_path = Path(config_path)
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    custom = yaml.safe_load(f) or {}
                data = deep_merge(data, custom)

        if overrides:
            data = deep_merge(data, overrides)

        # Inject env vars for translation
        trans = data.setdefault("translation", {})
        if not trans.get("base_url"):
            trans["base_url"] = os.getenv("TRANSLATION_BASE_URL")
        if not trans.get("api_key"):
            trans["api_key"] = os.getenv("TRANSLATION_API_KEY")
        if not trans.get("model"):
            trans["model"] = os.getenv("TRANSLATION_MODEL")

        return cls(data)

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation get, e.g. config.get('detection.backend')."""
        parts = key.split(".")
        node: Any = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> Dict[str, Any]:
        return dict(self._data.get(name, {}))

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def hash(self) -> str:
        return config_hash(self._data)

    def with_backend(self, backend: str) -> "Config":
        """Return a copy with all stage backends set to the given value (e.g. 'mock')."""
        data = json.loads(json.dumps(self._data))  # deep copy
        for stage in ("detection", "ocr", "translation", "inpainting"):
            if stage in data:
                data[stage]["backend"] = backend
        return Config(data)
