"""Smoke tests for yolo_comic factory wiring (no GPU / no weights required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from manga_ai.detection import create_detector
from manga_ai.detection.yolo_comic import YOLOComicDetector, _region_type


def test_create_yolo_comic_lazy():
    d = create_detector(
        "yolo_comic",
        model="models/comic-yolo.pt",
        variant="textseg",
        auto_download=False,
        use_gpu=False,
    )
    assert isinstance(d, YOLOComicDetector)
    assert d._model is None
    assert d.variant == "textseg"


def test_region_type_from_names():
    assert _region_type(0, {0: "text_comic"}) == "text"
    assert _region_type(0, {0: "text_bubble"}) == "bubble"
    assert _region_type(1, {1: "text_free"}) == "text"


def test_yolo_detect_mocked():
    det = YOLOComicDetector(
        model="models/does-not-exist.pt",
        use_gpu=False,
        auto_download=False,
        variant="textseg",
    )
    box = MagicMock()
    box.xyxy = __import__("torch").tensor([[10.0, 20.0, 110.0, 80.0]])
    box.conf = __import__("torch").tensor([0.91])
    box.cls = __import__("torch").tensor([0.0])
    box.__len__ = lambda self: 1

    result = MagicMock()
    result.boxes = box

    model = MagicMock()
    model.names = {0: "text_comic"}
    model.predict.return_value = [result]

    det._model = model
    det._device = "cpu"
    det._names = {0: "text_comic"}

    img = np.zeros((200, 150, 3), dtype=np.uint8)
    regions = det.detect(img)
    assert len(regions) == 1
    assert regions[0]["bbox"] == [10, 20, 110, 80]
    assert regions[0]["region_type"] == "text"
    assert regions[0]["label"] == "text_comic"
