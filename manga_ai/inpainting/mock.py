"""Mock inpainter – fills masked areas with a soft average color."""

from __future__ import annotations

import cv2
import numpy as np

from manga_ai.inpainting.base import Inpainter


class MockInpainter(Inpainter):
    name = "mock"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Simple OpenCV Telea as a lightweight "mock" that still produces real output
        if mask.max() == 0:
            return image.copy()
        # Ensure mask is single channel uint8
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        result = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return result
