"""EasyOCR full-page text detector (real boxes, not mock).

Uses EasyOCR's CRAFT-based detector via readtext().
Requires: pip install easyocr
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from manga_ai.detection.base import Detector
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.detection.easyocr")


class EasyOCRDetector(Detector):
    """Full-page text detection using EasyOCR."""

    name = "easyocr"

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = True,
        confidence_threshold: float = 0.3,
        min_size: int = 10,
        paragraph: bool = False,
    ):
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.min_size = min_size
        self.paragraph = paragraph
        self._reader = None

    def _ensure_model(self):
        if self._reader is not None:
            return
        try:
            import easyocr
        except ImportError as e:
            raise ImportError(
                "easyocr is required for EasyOCRDetector.\n"
                "  pip install easyocr"
            ) from e

        # GPU if available
        gpu = bool(self.use_gpu)
        if gpu:
            try:
                import torch
                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False

        logger.info(f"Loading EasyOCR detector languages={self.languages} gpu={gpu}")
        self._reader = easyocr.Reader(self.languages, gpu=gpu)

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        self._ensure_model()
        # EasyOCR expects RGB or BGR numpy; RGB is fine
        if image.ndim == 2:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # detail=1 → list of (bbox, text, conf)
        results = self._reader.readtext(
            image,
            detail=1,
            paragraph=self.paragraph,
        )

        regions: List[Dict[str, Any]] = []
        h, w = image.shape[:2]

        for item in results:
            if not item or len(item) < 3:
                continue
            box, text, conf = item[0], item[1], float(item[2])
            if conf < self.confidence_threshold:
                continue
            # box: 4 points [[x,y], ...]
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
            x2, y2 = min(w, int(max(xs))), min(h, int(max(ys)))
            if (x2 - x1) < self.min_size or (y2 - y1) < self.min_size:
                continue

            polygon = [[int(p[0]), int(p[1])] for p in box]
            regions.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "polygon": polygon,
                    "confidence": conf,
                    "region_type": "text",
                    # optional hint for OCR stage (pipeline may ignore)
                    "preview_text": str(text) if text else "",
                }
            )

        # Sort top-to-bottom, left-to-right (reading order)
        regions.sort(key=lambda r: (r["bbox"][1] // 20, r["bbox"][0]))
        return regions

    def unload(self) -> None:
        self._reader = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("EasyOCR detector unloaded")
