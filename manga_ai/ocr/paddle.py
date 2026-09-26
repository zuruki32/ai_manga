"""PaddleOCR recognition — PaddleOCR 2.x and 3.x compatible.

Recognizes text on each detection crop (bbox / polygon), matching other OCR backends.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from manga_ai.logging import get_logger
from manga_ai.ocr.base import OCRBackend

logger = get_logger("manga_ai.ocr.paddle")


def _crop_region(
    image: np.ndarray, region: Dict[str, Any], pad_px: int = 2
) -> Optional[np.ndarray]:
    """Crop a detection region from a full-page RGB image."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    bbox = region.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    else:
        polygon = region.get("polygon") or []
        if not polygon:
            return None
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(w, x2 + pad_px)
    y2 = min(h, y2 + pad_px)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


class PaddleOCRBackend(OCRBackend):
    name = "paddle"

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        pad_px: int = 2,
        **kwargs: Any,
    ):
        self.lang = "en" if lang in ("korean", "ko", None) else lang
        self.use_gpu = use_gpu
        self.pad_px = int(pad_px)
        self._ocr = None
        self._api = None

    def _ensure_model(self):
        if self._ocr is not None:
            return
        from manga_ai.utils.paddle_ocr import create_paddle_ocr

        logger.info(f"Loading PaddleOCR rec lang={self.lang}")
        # enable_mkldnn=False avoids Windows CPU oneDNN/PIR crash
        self._ocr = create_paddle_ocr(
            lang=self.lang, use_gpu=self.use_gpu, enable_mkldnn=False
        )
        self._api = "v3" if hasattr(self._ocr, "predict") else "v2"

    def _run_ocr(self, crop: np.ndarray) -> Tuple[list, list]:
        """Run PaddleOCR on a single RGB crop; return (texts, confs)."""
        import cv2

        if crop.ndim == 2:
            bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

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
        return texts, confs

    def recognize(self, image: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_model()
        crop = _crop_region(image, region, pad_px=self.pad_px)
        if crop is None:
            return {"text": "", "confidence": 0.0, "language": self.lang}

        texts, confs = self._run_ocr(crop)
        text = " ".join(texts).strip()
        conf = float(sum(confs) / len(confs)) if confs else 0.0
        return {"text": text, "confidence": conf, "language": self.lang}

    def unload(self) -> None:
        self._ocr = None
