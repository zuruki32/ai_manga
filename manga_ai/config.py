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


def package_root() -> Path:
    """Repo / install root (parent of the ``manga_ai`` package)."""
    return Path(__file__).resolve().parent.parent


def list_bundled_configs() -> list[str]:
    """Return available config stem names from known config directories."""
    names: set[str] = set()
    for folder in (
        package_root() / "configs",
        Path(__file__).resolve().parent / "configs",
    ):
        if not folder.is_dir():
            continue
        for path in folder.glob("*.yaml"):
            names.add(path.stem)
        for path in folder.glob("*.yml"):
            names.add(path.stem)
    return sorted(names)


def resolve_config_path(config: str | Path | None) -> Optional[Path]:
    """Resolve a config path or bare name (e.g. ``hybrid_qwen_en``).

    Search order:
      1. Path as given (cwd-relative or absolute)
      2. ``./configs/<name>.yaml``
      3. ``<package_root>/configs/<name>.yaml``
      4. ``manga_ai/configs/<name>.yaml`` (bundled with the package)
    """
    if config is None or str(config).strip() == "":
        return None

    raw = Path(config)
    if raw.is_file():
        return raw.resolve()

    name = raw.name
    if raw.suffix.lower() not in (".yaml", ".yml"):
        name = f"{name}.yaml"

    root = package_root()
    candidates = [
        Path.cwd() / config,
        Path.cwd() / "configs" / name,
        root / "configs" / name,
        root / "configs" / raw.name,
        Path(__file__).resolve().parent / "configs" / name,
        Path(__file__).resolve().parent / "configs" / raw.name,
    ]
    # Also allow ``configs/foo.yaml`` when cwd is elsewhere but repo has it
    if not str(config).startswith(("configs/", "configs\\")):
        candidates.append(root / "configs" / Path(config).name)

    seen: set[Path] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


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
            cfg_path = resolve_config_path(config_path)
            if cfg_path is None:
                available = ", ".join(list_bundled_configs()) or "(none found)"
                raise FileNotFoundError(
                    f"Config not found: {config_path!r}. "
                    f"Tried cwd and package configs/. Available: {available}"
                )
            with open(cfg_path, "r", encoding="utf-8") as f:
                custom = yaml.safe_load(f) or {}
            data = deep_merge(data, custom)

        if overrides:
            data = deep_merge(data, overrides)

        # Inject env vars for translation (only fill blanks — never override YAML)
        trans = data.setdefault("translation", {})
        if not trans.get("base_url"):
            trans["base_url"] = os.getenv("TRANSLATION_BASE_URL")
        if not trans.get("api_key"):
            trans["api_key"] = os.getenv("TRANSLATION_API_KEY")
        if not trans.get("model"):
            trans["model"] = os.getenv("TRANSLATION_MODEL")
        # Env can force backend only when YAML left it empty/mock-ish
        env_backend = (os.getenv("TRANSLATION_BACKEND") or "").strip().lower()
        if env_backend and not trans.get("backend"):
            trans["backend"] = env_backend

        # Safety: HF backend + GGUF path is a common local mis-edit
        backend = str(trans.get("backend") or "").lower()
        model = str(trans.get("model") or "")
        model_l = model.lower().replace("\\", "/")
        if backend in ("huggingface", "hf", "local", "transformers", "gemma") and (
            model_l.endswith(".gguf") or "gguf" in Path(model_l).name
        ):
            fixed = model
            if Path(model).name.lower().endswith("-gguf"):
                fixed = str(Path(model).with_name(Path(model).name[: -len("-gguf")]))
            elif model_l.endswith(".gguf") and Path(model).parent.name.lower().endswith("-gguf"):
                parent = Path(model).parent
                fixed = str(parent.with_name(parent.name[: -len("-gguf")]))
            if fixed != model:
                # Late import avoided — warn via print; pipeline logger may not be up
                import warnings
                warnings.warn(
                    f"Config had GGUF model {model!r} with HF backend; "
                    f"rewrote to {fixed!r}. Edit configs/hybrid_qwen_en.yaml if wrong.",
                    UserWarning,
                    stacklevel=2,
                )
                trans["model"] = fixed

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
