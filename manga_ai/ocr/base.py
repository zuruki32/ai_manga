"""Abstract OCR interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class OCRBackend(ABC):
    """Abstract OCR backend."""

    name: str = "base"

    @abstractmethod
    def recognize(self, image: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recognize text in a region of the image.

        Args:
            image: Full page RGB array
            region: Detection dict with bbox / polygon

        Returns:
            {
                "text": str,
                "confidence": float,
                "language": str
            }
        """
        ...

    def unload(self) -> None:
        pass
