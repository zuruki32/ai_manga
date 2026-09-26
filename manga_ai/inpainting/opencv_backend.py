"""OpenCV Telea / NS inpainting fallback."""

from __future__ import annotations

import cv2
import numpy as np

from manga_ai.inpainting.base import Inpainter


class OpenCVInpainter(Inpainter):
    name = "opencv"

    def __init__(self, radius: int = 5, method: str = "telea"):
        self.radius = radius
        self.method = method

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.max() == 0:
            return image.copy()
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        flags = cv2.INPAINT_TELEA if self.method == "telea" else cv2.INPAINT_NS
        return cv2.inpaint(image, mask, self.radius, flags)

    def unload(self) -> None:
        pass
