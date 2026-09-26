"""Abstract Inpainter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Inpainter(ABC):
    """Abstract inpainting backend."""

    name: str = "base"

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remove text regions indicated by the mask.

        Args:
            image: RGB uint8 array (H, W, 3)
            mask: single-channel uint8 (H, W), 255 = area to inpaint

        Returns:
            Cleaned RGB image (same shape as input)
        """
        ...

    def unload(self) -> None:
        pass
