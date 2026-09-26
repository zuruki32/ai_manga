"""Mock OCR backend."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from manga_ai.ocr.base import OCRBackend

# Simple deterministic fake texts
_FAKE_TEXTS = [
    "안녕하세요",
    "무슨 일이야?",
    "빨리 와!",
    "정말?",
    "알겠어.",
    "고마워.",
    "왜 그래?",
    "괜찮아.",
    "다음에 봐.",
    "이거 뭐야?",
]


class MockOCR(OCRBackend):
    name = "mock"

    def __init__(self):
        self._counter = 0

    def recognize(self, image: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        text = _FAKE_TEXTS[self._counter % len(_FAKE_TEXTS)]
        self._counter += 1
        return {
            "text": text,
            "confidence": 0.92,
            "language": "ko",
        }
