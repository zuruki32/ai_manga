"""PaddleOCR detector — supports PaddleOCR 2.x (.ocr) and 3.x (.predict)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from manga_ai.detection.base import Detector
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.detection.paddle")


class PaddleDetector(Detector):
    name = "paddle"

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        confidence_threshold: float = 0.4,
        **kwargs: Any,
    ):
        self.lang = "en" if lang in ("korean", "ko", None) else lang
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self._ocr = None
        self._api = None  # "v3" | "v2"

    def _ensure_model(self):
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                "paddleocr required: pip install paddlepaddle paddleocr"
            ) from e

        logger.info(f"Loading PaddleOCR lang={self.lang} gpu={self.use_gpu}")
        # Try 3.x style init first
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

        if hasattr(self._ocr, "predict"):
            self._api = "v3"
        else:
            self._api = "v2"
        logger.info(f"PaddleOCR API mode={self._api}")

    def _parse_v3(self, result, w: int, h: int) -> List[Dict[str, Any]]:
        regions = []
        if not result:
            return regions
        pages = result if isinstance(result, list) else [result]
        for page in pages:
            if page is None:
                continue
            # dict-like
            if isinstance(page, dict):
                texts = page.get("rec_texts") or page.get("texts") or []
                boxes = page.get("rec_boxes") or page.get("dt_polys") or page.get("rec_polys") or []
                scores = page.get("rec_scores") or page.get("scores") or [1.0] * len(texts)
            elif hasattr(page, "rec_texts"):
                texts = list(getattr(page, "rec_texts") or [])
                boxes = list(getattr(page, "rec_boxes", None) or getattr(page, "dt_polys", None) or [])
                scores = list(getattr(page, "rec_scores", None) or [1.0] * len(texts))
            else:
                continue
            for i, box in enumerate(boxes):
                conf = float(scores[i]) if i < len(scores) else 1.0
                if conf < self.confidence_threshold:
                    continue
                preview = str(texts[i]) if i < len(texts) else ""
                pts = np.array(box, dtype=float).reshape(-1, 2)
                xs, ys = pts[:, 0], pts[:, 1]
                x1, y1 = max(0, int(xs.min())), max(0, int(ys.min()))
                x2, y2 = min(w, int(xs.max())), min(h, int(ys.max()))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                regions.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "polygon": [[int(p[0]), int(p[1])] for p in pts],
                        "confidence": conf,
                        "region_type": "text",
                        "preview_text": preview,
                    }
                )
        return regions

    def _parse_v2(self, result, w: int, h: int) -> List[Dict[str, Any]]:
        regions = []
        if not result:
            return regions
        lines = result[0] if result and isinstance(result[0], list) else result
        if not lines:
            return regions
        for item in lines:
            if not item:
                continue
            box = item[0]
            conf, preview = 1.0, ""
            if len(item) > 1 and item[1] is not None:
                if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                    preview, conf = str(item[1][0]), float(item[1][1])
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
        return regions

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        self._ensure_model()
        import cv2

        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]

        if self._api == "v3":
            try:
                result = self._ocr.predict(bgr)
            except Exception:
                result = self._ocr.predict(input=bgr)
            regions = self._parse_v3(result, w, h)
        else:
            try:
                result = self._ocr.ocr(bgr, cls=True)
            except TypeError:
                result = self._ocr.ocr(bgr)
            regions = self._parse_v2(result, w, h)

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
