"""Mock detector for pipeline testing without ML models."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from manga_ai.detection.base import Detector


class MockDetector(Detector):
    """Returns a few synthetic text regions based on image size."""

    name = "mock"

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        h, w = image.shape[:2]
        # Place 2–3 fake boxes in typical bubble locations
        regions = [
            {
                "bbox": [int(w * 0.1), int(h * 0.1), int(w * 0.45), int(h * 0.25)],
                "polygon": [
                    [int(w * 0.1), int(h * 0.1)],
                    [int(w * 0.45), int(h * 0.1)],
                    [int(w * 0.45), int(h * 0.25)],
                    [int(w * 0.1), int(h * 0.25)],
                ],
                "confidence": 0.95,
                "region_type": "text",
            },
            {
                "bbox": [int(w * 0.5), int(h * 0.55), int(w * 0.9), int(h * 0.7)],
                "polygon": [
                    [int(w * 0.5), int(h * 0.55)],
                    [int(w * 0.9), int(h * 0.55)],
                    [int(w * 0.9), int(h * 0.7)],
                    [int(w * 0.5), int(h * 0.7)],
                ],
                "confidence": 0.88,
                "region_type": "text",
            },
        ]
        if h > 800:
            regions.append(
                {
                    "bbox": [int(w * 0.2), int(h * 0.8), int(w * 0.75), int(h * 0.92)],
                    "polygon": [
                        [int(w * 0.2), int(h * 0.8)],
                        [int(w * 0.75), int(h * 0.8)],
                        [int(w * 0.75), int(h * 0.92)],
                        [int(w * 0.2), int(h * 0.92)],
                    ],
                    "confidence": 0.91,
                    "region_type": "text",
                }
            )
        return regions
