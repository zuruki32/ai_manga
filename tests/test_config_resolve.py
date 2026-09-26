"""Config path resolution tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from manga_ai.config import Config, list_bundled_configs, resolve_config_path


def test_resolve_hybrid_qwen_by_relative_path():
    p = resolve_config_path("configs/hybrid_qwen_en.yaml")
    assert p is not None
    assert p.is_file()
    assert p.name == "hybrid_qwen_en.yaml"


def test_resolve_hybrid_qwen_by_bare_name():
    p = resolve_config_path("hybrid_qwen_en")
    assert p is not None
    assert p.name == "hybrid_qwen_en.yaml"


def test_list_bundled_includes_hybrid_qwen():
    names = list_bundled_configs()
    assert "hybrid_qwen_en" in names
    assert "default" in names


def test_load_hybrid_qwen_config():
    cfg = Config.load(config_path="hybrid_qwen_en")
    assert cfg.get("ocr.backend") == "hybrid_qwen"
    assert cfg.get("detection.backend") == "paddle"


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError, match="Config not found"):
        Config.load(config_path="definitely_missing_xyz_123")
