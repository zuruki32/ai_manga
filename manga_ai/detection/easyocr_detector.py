"""EasyOCR full-page text detector tuned for 6GB GPUs.

- Runs CRAFT detect via readtext
- Optional long-side limit to save VRAM during detection
- Scales boxes back to original resolution
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from manga_ai.detection.base import Detector
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.detection.easyocr")


class EasyOCRDetector(Detector):
    name = "easyocr"

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = True,
        confidence_threshold: float = 0.22,
        min_size: int = 10,
        paragraph: bool = False,
        detect_max_dimension: int = 1600,
        **kwargs: Any,
    ):
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.min_size = min_size
        self.paragraph = paragraph
        self.detect_max_dimension = detect_max_dimension
        self._reader = None

    def _ensure_model(self):
        if self._reader is not None:
            return
        try:
            import easyocr
        except ImportError as e:
            raise ImportError(
                "easyocr is required for EasyOCRDetector.\n  pip install easyocr"
            ) from e

        gpu = bool(self.use_gpu)
        if gpu:
            try:
                import torch
                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False

        logger.info(f"Loading EasyOCR detector languages={self.languages} gpu={gpu}")
        self._reader = easyocr.Reader(self.languages, gpu=gpu)

    def _resize_for_detect(self, image: np.ndarray):
        """Downscale long side for VRAM; return image, scale_x, scale_y."""
        import cv2

        h, w = image.shape[:2]
        long_side = max(h, w)
        if self.detect_max_dimension and long_side > self.detect_max_dimension:
            scale = self.detect_max_dimension / float(long_side)
            nw, nh = int(w * scale), int(h * scale)
            small = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
            return small, w / nw, h / nh
        return image, 1.0, 1.0

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        self._ensure_model()
        import cv2

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h0, w0 = image.shape[:2]
        work, sx, sy = self._resize_for_detect(image)

        results = self._reader.readtext(
            work,
            detail=1,
            paragraph=self.paragraph,
            # slightly more sensitive for manhwa fonts
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.3,
            canvas_size=min(2560, max(work.shape[:2])),
            mag_ratio=1.0,
        )

        regions: List[Dict[str, Any]] = []
        for item in results:
            if not item or len(item) < 3:
                continue
            box, text, conf = item[0], item[1], float(item[2])
            if conf < self.confidence_threshold:
                continue

            # scale box back to original resolution
            pts = [[float(p[0]) * sx, float(p[1]) * sy] for p in box]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
            x2 = min(w0, int(max(xs)))
            y2 = min(h0, int(max(ys)))
            if (x2 - x1) < self.min_size or (y2 - y1) < self.min_size:
                continue

            polygon = [[int(p[0]), int(p[1])] for p in pts]
            regions.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "polygon": polygon,
                    "confidence": conf,
                    "region_type": "text",
                    "preview_text": str(text) if text else "",
                }
            )

        regions.sort(key=lambda r: (r["bbox"][1] // 24, r["bbox"][0]))
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
