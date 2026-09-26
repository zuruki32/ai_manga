"""YOLO / Ultralytics comic bubble + text detector.

Recommended models (download once):
  - ogkalu/comic-text-segmenter-yolov8m  (YOLOv8m, manga/webtoon)
  - any ultralytics .pt trained on speech bubbles

Usage:
  detection:
    backend: yolo_comic
    model: models/comic-yolo.pt   # local path
    confidence_threshold: 0.25
    imgsz: 1280
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from manga_ai.detection.base import Detector
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.detection.yolo_comic")


class YOLOComicDetector(Detector):
    name = "yolo_comic"

    def __init__(
        self,
        model: str = "models/comic-yolo.pt",
        use_gpu: bool = True,
        confidence_threshold: float = 0.25,
        imgsz: int = 1280,
        classes: Optional[List[int]] = None,
        **kwargs: Any,
    ):
        self.model_path = model
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.imgsz = int(imgsz)
        self.classes = classes  # e.g. [0,1] for bubble+text
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics required for yolo_comic detector.\n"
                "  pip install ultralytics"
            ) from e

        path = Path(self.model_path)
        if not path.exists():
            # try HF-style id via ultralytics if user passed repo
            logger.warning(
                f"Model path not found: {path}. "
                f"Place a YOLOv8 .pt at this path "
                f"(e.g. from ogkalu/comic-text-segmenter-yolov8m)."
            )
        device = 0 if self.use_gpu else "cpu"
        logger.info(f"Loading YOLO comic detector model={self.model_path} device={device}")
        self._model = YOLO(str(self.model_path))
        self._device = device

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        self._ensure_model()
        h, w = image.shape[:2]

        kwargs = dict(
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            device=self._device,
            verbose=False,
        )
        if self.classes is not None:
            kwargs["classes"] = self.classes

        results = self._model.predict(image, **kwargs)
        regions: List[Dict[str, Any]] = []
        if not results:
            return regions

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return regions

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            conf = float(confs[i])
            cls_id = int(clss[i])
            # Prefer polygon as bbox corners
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            regions.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "polygon": poly,
                    "confidence": conf,
                    "region_type": "bubble" if cls_id == 0 else "text",
                    "class_id": cls_id,
                }
            )

        regions.sort(key=lambda r: (r["bbox"][1] // 20, r["bbox"][0]))
        return regions

    def unload(self) -> None:
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("YOLO comic detector unloaded")
