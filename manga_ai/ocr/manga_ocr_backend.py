"""Manga-OCR backend for Japanese (optional)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from manga_ai.ocr.base import OCRBackend
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.ocr.manga_ocr")


class MangaOCRBackend(OCRBackend):
    name = "manga_ocr"

    def __init__(self):
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from manga_ocr import MangaOcr
        except ImportError as e:
            raise ImportError(
                "manga-ocr required. pip install manga-ocr"
            ) from e
        self._model = MangaOcr()

    def recognize(self, image: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_model()
        from PIL import Image
        bbox = region.get("bbox")
        if not bbox or len(bbox) != 4:
            return {"text": "", "confidence": 0.0, "language": "ja"}
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return {"text": "", "confidence": 0.0, "language": "ja"}
        pil = Image.fromarray(crop)
        text = self._model(pil)
        return {
            "text": (text or "").strip(),
            "confidence": 0.9 if text else 0.0,
            "language": "ja",
        }

    def unload(self) -> None:
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
