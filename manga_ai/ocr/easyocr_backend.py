"""EasyOCR local backend (fully offline after model download)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from manga_ai.ocr.base import OCRBackend
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.ocr.easyocr")

# EasyOCR language codes
_LANG = {
    "en": "en",
    "ko": "ko",
    "ja": "ja",
    "zh": "ch_sim",
    "fa": "fa",
    "persian": "fa",
    "korean": "ko",
    "japan": "ja",
    "chinese": "ch_sim",
}


class EasyOCRBackend(OCRBackend):
    name = "easyocr"

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: bool = True,
        default_lang: str = "en",
    ):
        self.languages = languages or [default_lang]
        # Normalize
        self.languages = [_LANG.get(l, l) for l in self.languages]
        self.use_gpu = use_gpu
        self._reader = None

    def _ensure_model(self):
        if self._reader is not None:
            return
        try:
            import easyocr
        except ImportError as e:
            raise ImportError(
                "easyocr required. pip install easyocr"
            ) from e
        logger.info(f"Loading EasyOCR languages={self.languages} gpu={self.use_gpu}")
        self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)

    def recognize(self, image: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_model()
        bbox = region.get("bbox")
        if not bbox or len(bbox) != 4:
            return {"text": "", "confidence": 0.0, "language": self.languages[0]}
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return {"text": "", "confidence": 0.0, "language": self.languages[0]}

        # EasyOCR expects RGB or BGR numpy
        results = self._reader.readtext(crop, detail=1, paragraph=False)
        texts = []
        confs = []
        for item in results:
            # item: (bbox, text, conf)
            if len(item) >= 3:
                texts.append(str(item[1]))
                confs.append(float(item[2]))
        text = " ".join(texts).strip()
        conf = float(sum(confs) / len(confs)) if confs else 0.0
        return {
            "text": text,
            "confidence": conf,
            "language": self.languages[0],
        }

    def unload(self) -> None:
        self._reader = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
