"""LaMa inpainting backend via simple-lama-inpainting (optional)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from manga_ai.inpainting.base import Inpainter
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.inpainting.lama")


class LaMaInpainter(Inpainter):
    name = "lama"

    def __init__(
        self,
        device: str = "cuda",
        tile_size: int = 768,
        overlap: int = 64,
        precision: str = "fp16",
    ):
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.precision = precision
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError as e:
            raise ImportError(
                "simple-lama-inpainting required. "
                "pip install simple-lama-inpainting"
            ) from e
        device = self.device
        try:
            import torch
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
                logger.warning("CUDA not available, LaMa falling back to CPU")
        except ImportError:
            device = "cpu"
        self._model = SimpleLama(device=device)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._ensure_model()
        from PIL import Image

        if mask.max() == 0:
            return image.copy()

        h, w = image.shape[:2]
        # Tiled processing for large images / low VRAM
        if max(h, w) <= self.tile_size + self.overlap:
            return self._inpaint_full(image, mask)

        return self._inpaint_tiled(image, mask)

    def _inpaint_full(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from PIL import Image
        pil_img = Image.fromarray(image)
        pil_mask = Image.fromarray(mask).convert("L")
        result = self._model(pil_img, pil_mask)
        if isinstance(result, Image.Image):
            return np.array(result.convert("RGB"))
        return np.array(result)

    def _inpaint_tiled(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Simple overlapping tile inpainting for 6GB VRAM."""
        h, w = image.shape[:2]
        out = image.copy().astype(np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        step = max(1, self.tile_size - self.overlap)

        ys = list(range(0, h, step))
        xs = list(range(0, w, step))
        if ys[-1] + self.tile_size < h:
            ys.append(max(0, h - self.tile_size))
        if xs[-1] + self.tile_size < w:
            xs.append(max(0, w - self.tile_size))

        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + self.tile_size, h)
                x1 = min(x0 + self.tile_size, w)
                y0c, x0c = y1 - (y1 - y0), x1 - (x1 - x0)
                # ensure tile size
                tile_img = image[y0:y1, x0:x1]
                tile_mask = mask[y0:y1, x0:x1]
                if tile_mask.max() == 0:
                    continue
                try:
                    cleaned = self._inpaint_full(tile_img, tile_mask)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning("OOM on tile, clearing cache and retrying smaller")
                        self._clear_cuda()
                        # fall back to OpenCV for this tile
                        import cv2
                        cleaned = cv2.inpaint(
                            tile_img, tile_mask, 5, cv2.INPAINT_TELEA
                        )
                    else:
                        raise
                th, tw = cleaned.shape[:2]
                out[y0 : y0 + th, x0 : x0 + tw] += cleaned.astype(np.float32)
                weight[y0 : y0 + th, x0 : x0 + tw] += 1.0

        weight = np.maximum(weight, 1.0)
        out = out / weight[:, :, None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def _clear_cuda(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload(self) -> None:
        self._model = None
        self._clear_cuda()
