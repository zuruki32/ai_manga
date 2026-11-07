"""Comic-text-detector (ctd) text segmentation mask.

Your YOLO/bubble detector only finds dialogue text, so stylized/colored sound
effects and free text over artwork are never masked and survive cleaning. This
runs dmMaze's comic-text-detector (the same model PanelCleaner and
manga-image-translator use) to get a pixel-precise mask of ALL comic text, which
the masking stage unions into the cleaning mask.

Runs through OpenCV's cv2.dnn on the ONNX model — no PyTorch, no extra deps
(opencv is already required), CPU is fine (~0.5-1 s/page at 1024px).
"""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from manga_ai.logging import get_logger

logger = get_logger("manga_ai.masking.ctd")

CTD_URL = os.environ.get(
    "CTD_MODEL_URL",
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt.onnx",
)


def _default_cache() -> Path:
    env = os.environ.get("MANGA_AI_MODEL_DIR")
    if env:
        return Path(env)
    local = Path.cwd() / "models"
    if local.is_dir():
        return local
    return Path.home() / ".cache" / "manga_ai"


class CTDTextMasker:
    """Pixel-precise text mask for the whole page (bubbles, SFX, free text)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: int = 1024,
        thresh: float = 0.3,
        dilate: int = 3,
    ) -> None:
        self.input_size = int(input_size)
        self.thresh = float(thresh)
        self.dilate = int(dilate)
        self.model_path = Path(model_path) if model_path else (_default_cache() / "comictextdetector.pt.onnx")
        self._net = None

    # ------------------------------------------------------------------
    def _ensure_downloaded(self) -> Path:
        p = self.model_path
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        last: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                tmp.unlink(missing_ok=True)
                logger.info("Downloading comic-text-detector (attempt %d) from %s", attempt, CTD_URL)
                urllib.request.urlretrieve(CTD_URL, tmp)
                if tmp.stat().st_size < 1_000_000:
                    raise RuntimeError("downloaded file too small")
                tmp.replace(p)
                return p
            except Exception as e:  # pragma: no cover - network dependent
                last = e
                logger.warning("ctd download attempt %d failed: %s", attempt, e)
                time.sleep(2 ** attempt)
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download comic-text-detector: {last}. "
            f"Download it manually from {CTD_URL} and place it at {p}."
        )

    def _ensure_net(self) -> None:
        if self._net is not None:
            return
        path = self._ensure_downloaded()
        self._net = cv2.dnn.readNetFromONNX(str(path))
        logger.info("Loaded comic-text-detector (cv2.dnn) from %s", path)

    # ------------------------------------------------------------------
    def text_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a uint8 mask (H, W), 255 = text, for RGB input."""
        self._ensure_net()
        h, w = image_rgb.shape[:2]
        s = self.input_size
        r = s / max(h, w)
        nh, nw = max(1, round(h * r)), max(1, round(w * r))
        resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((s, s, 3), 114, np.uint8)
        canvas[:nh, :nw] = resized
        # canvas is already RGB (pipeline arrays are RGB); don't swap channels
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (s, s), swapRB=False)
        self._net.setInput(blob)
        seg = self._net.forward("seg")
        segm = seg[0, 0] if seg.ndim == 4 else np.squeeze(seg)
        segm = segm[:nh, :nw]
        mask = cv2.resize(
            (segm > self.thresh).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST
        )
        if self.dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate * 2 + 1, self.dilate * 2 + 1))
            mask = cv2.dilate(mask, k)
        return mask

    def unload(self) -> None:
        self._net = None
