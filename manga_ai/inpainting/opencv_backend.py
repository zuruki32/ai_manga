"""Improved OpenCV inpainting for manhwa speech bubbles / captions.

Strategy (manga-image-translator style heuristics without heavy models):
1. Soft-expand mask
2. Estimate local background (bubble white / panel color) via masked median
3. Fill mask with that background (kills ink cleanly)
4. Edge-blend + TELEA/NS pass to remove seams
"""

from __future__ import annotations

import cv2
import numpy as np

from manga_ai.inpainting.base import Inpainter


class OpenCVInpainter(Inpainter):
    name = "opencv"

    def __init__(
        self,
        radius: int = 7,
        method: str = "telea",
        passes: int = 2,
        soft_edge: int = 3,
        bg_fill: bool = True,
    ):
        self.radius = max(1, int(radius))
        self.method = (method or "telea").lower()
        self.passes = max(1, int(passes))
        self.soft_edge = max(0, int(soft_edge))
        self.bg_fill = bool(bg_fill)

    def _flags(self) -> int:
        return cv2.INPAINT_TELEA if self.method != "ns" else cv2.INPAINT_NS

    def _prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask = (mask > 0).astype(np.uint8) * 255
        if self.soft_edge > 0:
            k = self.soft_edge * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)
            # close small holes inside text strokes
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _local_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Per-pixel background estimate from neighborhood outside mask."""
        inv = cv2.bitwise_not(mask)
        # Prefer bright pixels (speech-bubble white) when present near mask
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        bright = ((gray > 200) & (inv > 0)).astype(np.uint8) * 255
        sample = bright if bright.sum() > 500 else inv

        # Large blur of valid pixels → soft plate color under text
        img_f = image.astype(np.float32)
        if image.ndim == 3:
            sample3 = np.stack([sample] * 3, axis=-1).astype(np.float32) / 255.0
            blurred = cv2.GaussianBlur(img_f * sample3, (0, 0), sigmaX=12)
            weights = cv2.GaussianBlur(sample3, (0, 0), sigmaX=12) + 1e-6
            bg = blurred / weights
        else:
            sample_f = sample.astype(np.float32) / 255.0
            blurred = cv2.GaussianBlur(img_f * sample_f, (0, 0), sigmaX=12)
            weights = cv2.GaussianBlur(sample_f, (0, 0), sigmaX=12) + 1e-6
            bg = blurred / weights
        return np.clip(bg, 0, 255).astype(np.uint8)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask is None or np.max(mask) == 0:
            return image.copy()

        work = image.copy()
        m = self._prepare_mask(mask)

        if self.bg_fill:
            bg = self._local_background(work, m)
            if work.ndim == 3:
                sel = m > 0
                work[sel] = bg[sel]
            else:
                work[m > 0] = bg[m > 0]

        # Feathered second pass: shrink core, inpaint ring for seamless edge
        flags = self._flags()
        core = m
        for i in range(self.passes):
            r = self.radius + i * 2
            work = cv2.inpaint(work, core, r, flags)
            # alternate method on last pass for residual strokes
            if i == self.passes - 1 and self.method == "telea":
                work = cv2.inpaint(work, core, max(3, self.radius - 1), cv2.INPAINT_NS)

        # Soft edge blend between original and inpainted using distance mask
        if self.soft_edge > 0:
            dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            edge = np.clip(dist / max(1.0, float(self.soft_edge + 2)), 0.0, 1.0)
            if work.ndim == 3:
                edge = edge[:, :, None]
            # only blend near boundary (not deep inside)
            blend = np.clip(edge * 1.25, 0.0, 1.0)
            out = image.astype(np.float32) * (1.0 - blend) + work.astype(np.float32) * blend
            # deep inside mask keep fully cleaned
            deep = (m > 0) & (dist > self.soft_edge)
            if work.ndim == 3:
                out[deep] = work[deep]
            else:
                out[deep] = work[deep]
            return np.clip(out, 0, 255).astype(np.uint8)

        return work

    def unload(self) -> None:
        pass
