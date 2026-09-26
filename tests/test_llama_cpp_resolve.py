"""GGUF path resolution tests (no llama-cpp required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from manga_ai.translation.llama_cpp_backend import LlamaCppTranslator


def test_resolve_gguf_file(tmp_path: Path):
    f = tmp_path / "model-q4_k_m.gguf"
    f.write_bytes(b"fake")
    t = LlamaCppTranslator(model=str(f))
    assert t._resolve_gguf_path() == f.resolve()


def test_resolve_gguf_folder_prefers_q4(tmp_path: Path):
    (tmp_path / "model-q8_0.gguf").write_bytes(b"aaaaaaaa")
    q4 = tmp_path / "model-q4_k_m.gguf"
    q4.write_bytes(b"bbbb")
    t = LlamaCppTranslator(model=str(tmp_path))
    assert t._resolve_gguf_path() == q4.resolve()


def test_resolve_gguf_missing(tmp_path: Path):
    t = LlamaCppTranslator(model=str(tmp_path / "nope.gguf"))
    with pytest.raises(FileNotFoundError):
        t._resolve_gguf_path()
