"""Tests for backend factories and optional backends."""

from __future__ import annotations

import numpy as np
import pytest

from manga_ai.detection import create_detector
from manga_ai.ocr import create_ocr
from manga_ai.translation import create_translator
from manga_ai.inpainting import create_inpainter


def test_create_mock_detector():
    d = create_detector("mock")
    assert d.name == "mock"
    img = np.zeros((200, 150, 3), dtype=np.uint8)
    assert len(d.detect(img)) >= 1


def test_create_mock_ocr():
    o = create_ocr("mock")
    r = o.recognize(np.zeros((50, 50, 3), dtype=np.uint8), {"bbox": [0, 0, 40, 40]})
    assert "text" in r


def test_create_mock_translator():
    t = create_translator("mock")
    out = t.translate_chapter(
        [{"region_id": "1", "source_text": "안녕하세요"}],
        {"source_language": "ko", "target_language": "fa", "glossary": {}},
    )
    assert out[0]["text"] == "سلام"


def test_create_opencv_inpainter():
    inp = create_inpainter("opencv")
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    out = inp.inpaint(img, mask)
    assert out.shape == img.shape


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        create_detector("nonexistent_xyz")
