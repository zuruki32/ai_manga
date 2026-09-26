"""Tests for OCR crop helpers and hybrid_qwen factory wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from manga_ai.ocr import create_ocr, get_ocr_backend
from manga_ai.ocr.paddle import PaddleOCRBackend, _crop_region as paddle_crop
from manga_ai.ocr.hybrid_qwen import HybridQwenOCRBackend, _crop_region as qwen_crop


def test_create_hybrid_qwen_lazy():
    o = create_ocr(
        "hybrid_qwen",
        model="D:/projects/models/Qwen3-VL-8B-Instruct",
        lang="en",
        local_files_only=False,
    )
    assert o.name == "hybrid_qwen"
    assert isinstance(o, HybridQwenOCRBackend)
    assert o._model is None  # lazy — no download on construct
    assert "Qwen3-VL-8B-Instruct" in o.model_name.replace("\\", "/")



def test_get_ocr_backend_alias():
    o = get_ocr_backend("mock")
    assert o.name == "mock"


def test_crop_region_bbox():
    img = np.zeros((100, 80, 3), dtype=np.uint8)
    img[10:40, 20:50] = 255
    crop = paddle_crop(img, {"bbox": [20, 10, 50, 40]}, pad_px=0)
    assert crop is not None
    assert crop.shape == (30, 30, 3)
    assert int(crop.mean()) == 255


def test_crop_region_polygon_and_pad():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    region = {"polygon": [[10, 10], [30, 10], [30, 30], [10, 30]]}
    crop = qwen_crop(img, region, pad_px=2)
    assert crop is not None
    # 20x20 box + 2px pad each side → 24x24
    assert crop.shape[0] == 24
    assert crop.shape[1] == 24


def test_crop_region_empty():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert paddle_crop(img, {}, pad_px=0) is None
    assert paddle_crop(img, {"bbox": [5, 5, 5, 5]}, pad_px=0) is None


def test_paddle_recognize_crops_not_full_page():
    """Paddle OCR must run on the region crop, not the full page."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    region = {"bbox": [10, 20, 40, 50]}

    backend = PaddleOCRBackend(lang="en", use_gpu=False, pad_px=0)
    backend._api = "v2"
    backend._ocr = MagicMock()

    captured = {}

    def fake_run(crop):
        captured["shape"] = crop.shape
        return ["hello"], [0.9]

    with patch.object(backend, "_ensure_model"), patch.object(
        backend, "_run_ocr", side_effect=fake_run
    ):
        out = backend.recognize(img, region)

    assert out["text"] == "hello"
    assert captured["shape"] == (30, 30, 3)


def test_hybrid_qwen_recognize_uses_crop():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    region = {"bbox": [5, 5, 35, 25]}
    backend = HybridQwenOCRBackend(lang="en", pad_px=0)
    backend._model = object()  # skip load
    backend._processor = object()
    backend._torch_device = "cpu"

    with patch.object(backend, "_ensure_model"), patch.object(
        backend, "_generate", return_value="Hello world"
    ) as gen:
        out = backend.recognize(img, region)

    assert out["text"] == "Hello world"
    assert out["confidence"] > 0
    crop_arg = gen.call_args[0][0]
    assert crop_arg.shape == (20, 30, 3)


def test_unknown_ocr_still_raises():
    with pytest.raises(ValueError):
        create_ocr("not_a_real_backend")
