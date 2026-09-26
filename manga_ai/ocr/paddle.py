"""PaddleOCR recognition — PaddleOCR 2.x and 3.x compatible."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from manga_ai.logging import get_logger
from manga_ai.ocr.base import OCRBackend

logger = get_logger("manga_ai.ocr.paddle")


class PaddleOCRBackend(OCRBackend):
    name = "paddle"

    def __init__(self, lang: str = "en", use_gpu: bool = True, **kwargs: Any):
        self.lang = "en" if lang in ("korean", "ko", None) else lang
        self.use_gpu = use_gpu
        self._ocr = None
        self._api = None

    def _ensure_model(self):
        if self._ocr is not None:
            return
        from paddleocr import PaddleOCR

        logger.info(f"Loading PaddleOCR rec lang={self.lang}")
        for kwargs in (
            dict(lang=self.lang, device="gpu" if self.use_gpu else "cpu"),
            dict(lang=self.lang, use_gpu=self.use_gpu, use_angle_cls=True, show_log=False),
            dict(lang=self.lang, use_angle_cls=True),
            dict(lang=self.lang),
        ):
            try:
                self._ocr = PaddleOCR(**kwargs)
                break
            except TypeError:
                continue
        if self._ocr is None:
            self._ocr = PaddleOCR(lang=self.lang)
        self._api = "v3" if hasattr(self._ocr, "predict") else "v2"

    def recognize(self, image: np.ndarray, language: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_model()
        import cv2

        if image is None or image.size == 0:
            return {"text": "", "confidence": 0.0, "language": self.lang}
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        texts, confs = [], []
        if self._api == "v3":
            try:
                result = self._ocr.predict(bgr)
            except Exception:
                result = self._ocr.predict(input=bgr)
            pages = result if isinstance(result, list) else [result]
            for page in pages:
                if isinstance(page, dict):
                    ts = page.get("rec_texts") or []
                    ss = page.get("rec_scores") or []
                elif hasattr(page, "rec_texts"):
                    ts = list(getattr(page, "rec_texts") or [])
                    ss = list(getattr(page, "rec_scores", None) or [])
                else:
                    continue
                texts.extend([str(t) for t in ts])
                confs.extend([float(s) for s in ss])
        else:
            try:
                result = self._ocr.ocr(bgr, cls=True)
            except TypeError:
                result = self._ocr.ocr(bgr)
            if result and result[0]:
                for item in result[0]:
                    if item and len(item) > 1 and isinstance(item[1], (list, tuple)):
                        texts.append(str(item[1][0]))
                        confs.append(float(item[1][1]))

        text = " ".join(texts).strip()
        conf = float(sum(confs) / len(confs)) if confs else 0.0
        return {"text": text, "confidence": conf, "language": self.lang}

    def unload(self) -> None:
        self._ocr = None
