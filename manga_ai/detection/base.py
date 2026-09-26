"""Abstract Detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class Detector(ABC):
    """Abstract text detector."""

    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text regions in an image.

        Args:
            image: RGB numpy array (H, W, 3)

        Returns:
            List of dicts with keys:
                bbox: [x1, y1, x2, y2]
                polygon: [[x, y], ...]
                confidence: float
                region_type: str
        """
        ...

    def unload(self) -> None:
        """Release GPU / model resources. Override in real backends."""
        pass
