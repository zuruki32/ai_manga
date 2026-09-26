"""OCR router: pick backend by language."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from manga_ai.ocr.base import OCRBackend
from manga_ai.ocr.mock import MockOCR
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.ocr.router")


class OCRRouter(OCRBackend):
    """Routes to Manga-OCR for Japanese, Paddle for others, Mock as fallback."""

    name = "router"

    def __init__(
        self,
        language_map: Optional[Dict[str, str]] = None,
        default_lang: str = "ko",
        use_gpu: bool = True,
    ):
        self.language_map = language_map or {
            "ja": "manga_ocr",
            "ko": "paddle",
            "zh": "paddle",
            "en": "paddle",
        }
        self.default_lang = default_lang
        self.use_gpu = use_gpu
        self._backends: Dict[str, OCRBackend] = {}

    def _get_backend(self, lang: str) -> OCRBackend:
        key = self.language_map.get(lang, self.language_map.get(self.default_lang, "mock"))
        if key in self._backends:
            return self._backends[key]
        backend: OCRBackend
        try:
            if key == "manga_ocr":
                from manga_ai.ocr.manga_ocr_backend import MangaOCRBackend
                backend = MangaOCRBackend()
            elif key == "paddle":
                from manga_ai.ocr.paddle import PaddleOCRBackend
                backend = PaddleOCRBackend(lang=lang, use_gpu=self.use_gpu)
            else:
                backend = MockOCR()
        except ImportError as e:
            logger.warning(f"OCR backend '{key}' unavailable ({e}), using mock")
            backend = MockOCR()
        self._backends[key] = backend
        return backend

    def recognize(self, image: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        lang = region.get("source_language") or region.get("language") or self.default_lang
        return self._get_backend(lang).recognize(image, region)

    def unload(self) -> None:
        for b in self._backends.values():
            b.unload()
        self._backends.clear()
