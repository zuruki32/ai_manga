"""YOLO / Ultralytics comic text detector (ogkalu / AnimeText).

Recommended weights (download once)::

    python scripts/download_comic_yolo.py --variant textseg
    # → models/comic-yolo.pt   (ogkalu/comic-text-segmenter-yolov8m)

Other variants: ``bubble``, ``animetext``.

Config::

    detection:
      backend: yolo_comic
      model: models/comic-yolo.pt
      confidence_threshold: 0.25
      imgsz: 1024
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from manga_ai.detection.base import Detector
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.detection.yolo_comic")

# HF defaults used when local weights are missing
_DEFAULT_HF = {
    "textseg": ("ogkalu/comic-text-segmenter-yolov8m", "comic-text-segmenter.pt"),
    "bubble": ("ogkalu/comic-speech-bubble-detector-yolov8m", "comic-speech-bubble-detector.pt"),
    "animetext": ("Library-Mutsumi/AnimeText_yolo", "yolo12n_animetext/model.pt"),
}


def _region_type(cls_id: int, names: Optional[Dict[int, str]]) -> str:
    label = ""
    if names:
        label = str(names.get(cls_id, "")).lower()
    if "bubble" in label:
        return "bubble"
    if "free" in label or "text" in label or "comic" in label or "block" in label:
        return "text"
    return "text" if cls_id != 0 or not label else "bubble"


class YOLOComicDetector(Detector):
    name = "yolo_comic"

    def __init__(
        self,
        model: str = "models/comic-yolo.pt",
        use_gpu: bool = True,
        confidence_threshold: float = 0.25,
        imgsz: int = 1024,
        classes: Optional[List[int]] = None,
        variant: str = "textseg",
        auto_download: bool = True,
        **kwargs: Any,
    ):
        self.model_path = model
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.imgsz = int(imgsz)
        self.classes = classes
        self.variant = (variant or "textseg").lower()
        self.auto_download = auto_download
        self._model = None
        self._device: Any = "cpu"
        self._names: Dict[int, str] = {}

    def _download_default(self, dest: Path) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub required to auto-download YOLO weights.\n"
                "  pip install huggingface_hub\n"
                "Or: python scripts/download_comic_yolo.py --variant textseg"
            ) from e

        repo, filename = _DEFAULT_HF.get(self.variant, _DEFAULT_HF["textseg"])
        logger.info(f"Downloading {repo}/{filename} → {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(repo, filename)
        import shutil

        shutil.copy2(cached, dest)
        return dest

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
            if self.auto_download:
                path = self._download_default(path)
            else:
                raise FileNotFoundError(
                    f"YOLO weights not found: {path}\n"
                    f"Run: python scripts/download_comic_yolo.py --variant {self.variant}"
                )

        # Prefer CUDA when available; fall back to CPU cleanly
        device: Any = "cpu"
        if self.use_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    device = 0
            except Exception:
                device = "cpu"

        logger.info(
            f"Loading YOLO comic detector model={path} device={device} "
            f"variant={self.variant} imgsz={self.imgsz}"
        )
        self._model = YOLO(str(path))
        self._device = device
        names = getattr(self._model, "names", None) or {}
        if isinstance(names, dict):
            self._names = {int(k): str(v) for k, v in names.items()}
        else:
            self._names = {i: str(n) for i, n in enumerate(names)}

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        self._ensure_model()
        h, w = image.shape[:2]

        kwargs: Dict[str, Any] = dict(
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
            label = self._names.get(cls_id, str(cls_id))
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            regions.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "polygon": poly,
                    "confidence": conf,
                    "region_type": _region_type(cls_id, self._names),
                    "class_id": cls_id,
                    "label": label,
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
