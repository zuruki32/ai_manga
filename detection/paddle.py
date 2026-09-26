"""PaddleOCR-based text detector (English / comics friendly)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from manga_ai.detection.base import Detector
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.detection.paddle")


class PaddleDetector(Detector):
    """Full-page text detection via PaddleOCR (det + optional rec for confidence)."""

    name = "paddle"

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        confidence_threshold: float = 0.5,
        **kwargs: Any,
    ):
        self.lang = lang if lang not in ("korean", "ko") else "en"
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self._ocr = None

    def _ensure_model(self):
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                "paddleocr required.\n"
                "  pip install paddlepaddle paddleocr\n"
                "  # GPU: install paddlepaddle-gpu matching your CUDA"
            ) from e

        logger.info(f"Loading PaddleOCR detector lang={self.lang} gpu={self.use_gpu}")
        # Newer paddleocr API varies; try modern then fallback
        try:
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
            )
        except TypeError:
            self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        self._ensure_model()
        # Paddle expects BGR or path; RGB numpy often works
        import cv2

        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            result = self._ocr.ocr(bgr, cls=True)
        except TypeError:
            result = self._ocr.ocr(bgr)

        regions: List[Dict[str, Any]] = []
        if not result:
            return regions

        # result is list per page; first page
        lines = result[0] if result and isinstance(result[0], list) else result
        if not lines:
            return regions

        h, w = image.shape[:2]
        for item in lines:
            if not item:
                continue
            # item: [box, (text, conf)] or similar
            box = item[0]
            conf = 1.0
            preview = ""
            if len(item) > 1 and item[1] is not None:
                if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                    preview = str(item[1][0])
                    conf = float(item[1][1])
                elif isinstance(item[1], (int, float)):
                    conf = float(item[1])
            if conf < self.confidence_threshold:
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
            x2, y2 = min(w, int(max(xs))), min(h, int(max(ys)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            regions.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "polygon": [[int(p[0]), int(p[1])] for p in box],
                    "confidence": conf,
                    "region_type": "text",
                    "preview_text": preview,
                }
            )
        regions.sort(key=lambda r: (r["bbox"][1] // 20, r["bbox"][0]))
        return regions

    def unload(self) -> None:
        self._ocr = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Paddle detector unloaded")
