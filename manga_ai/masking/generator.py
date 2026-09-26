"""Generate text masks from polygons / bboxes (+ optional residual fill)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class MaskGenerator:
    """Creates binary masks for inpainting from detected text regions."""

    def __init__(
        self,
        dilation_px: int = 12,
        blur_radius: int = 2,
        use_polygon: bool = True,
        residual_expand: bool = True,
        residual_thresh: int = 140,
    ):
        self.dilation_px = dilation_px
        self.blur_radius = blur_radius
        self.use_polygon = use_polygon
        self.residual_expand = residual_expand
        self.residual_thresh = residual_thresh

    def generate(
        self,
        image_shape: Tuple[int, int],
        regions: List[Dict[str, Any]],
        image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create uint8 mask (255 = remove).

        If residual_expand and image given, expand mask over nearby dark
        pixels (leftover letter strokes outside the box).
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for region in regions:
            if self.use_polygon and region.get("polygon"):
                pts = np.array(region["polygon"], dtype=np.int32)
                if pts.ndim == 2 and pts.shape[0] >= 3:
                    cv2.fillPoly(mask, [pts], 255)
                    continue
            bbox = region.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = 255

        if self.dilation_px > 0:
            k = self.dilation_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Grow mask over dark ink near detected text (helps full-page clean)
        if self.residual_expand and image is not None and mask.any():
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            dark = (gray < self.residual_thresh).astype(np.uint8) * 255
            # only near existing mask
            near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
            extra = cv2.bitwise_and(dark, near)
            mask = cv2.bitwise_or(mask, extra)
            mask = cv2.dilate(
                mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )

        if self.blur_radius > 0:
            k = self.blur_radius * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        return mask
