"""IOPaint inpainting backend.

Wraps IOPaint's model manager so the pipeline can use the manga/anime
finetuned LaMa model (``lama_large_512px``, published as the ``manga`` model
in IOPaint) for high-quality text removal that reconstructs the art and
screentone behind the text.

Why this over ``simple-lama-inpainting``:
    - ``simple_lama_inpainting`` loads the *generic* big-LaMa trained on
      natural photos (Places2). It smears line art and screentone.
    - IOPaint's ``manga`` model is LaMa finetuned on ~300k manga/anime pages,
      which is what we actually want for cleaning.

VRAM: LaMa-family models are small (~200 MB) and run comfortably on 6 GB.
With ``HDStrategy.CROP`` the model only ever sees crops around masked text
(plus a margin), so even very tall webtoon pages stay cheap and the rest of
the page is copied through untouched.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from manga_ai.inpainting.base import Inpainter
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.inpainting.iopaint")


class IOPaintInpainter(Inpainter):
    """High-quality manga text removal via IOPaint.

    Args:
        model_name: IOPaint model id. ``"manga"`` is the anime/manga finetuned
            LaMa (recommended). ``"lama"`` is the generic big-LaMa. Other erase
            models such as ``"mat"`` or ``"migan"`` also work.
        device: ``"cuda"``, ``"cuda:0"`` or ``"cpu"``. Falls back to CPU when
            CUDA is unavailable.
        crop_trigger_size: Use the CROP strategy when the longer image side
            exceeds this. Lower = less VRAM, more (smaller) forward passes.
        crop_margin: Context pixels kept around each masked region in a crop.
            More margin = better blending, more VRAM per crop.
        resize_limit: Longer side limit used by the RESIZE strategy (only when
            ``hd_strategy="resize"``).
        hd_strategy: ``"crop"`` (default, best for webtoons/low VRAM),
            ``"resize"`` (downscale whole page), or ``"original"`` (full page,
            highest VRAM).
        result_is_bgr: IOPaint erase models return BGR; keep ``True`` so colors
            are correct. Flip to ``False`` only if output looks blue-tinted.
    """

    name = "iopaint"

    def __init__(
        self,
        model_name: str = "manga",
        device: str = "cuda",
        crop_trigger_size: int = 1024,
        crop_margin: int = 128,
        resize_limit: int = 2048,
        hd_strategy: str = "crop",
        result_is_bgr: bool = True,
        **_: Any,
    ) -> None:
        self.model_name = model_name or "manga"
        self.device = device or "cuda"
        self.crop_trigger_size = int(crop_trigger_size)
        self.crop_margin = int(crop_margin)
        self.resize_limit = int(resize_limit)
        self.hd_strategy = (hd_strategy or "crop").lower()
        self.result_is_bgr = bool(result_is_bgr)
        self._model = None
        self._config = None
        self._resolved_device: Optional[str] = None

    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from iopaint.model_manager import ModelManager
            from iopaint.schema import HDStrategy, InpaintRequest
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "IOPaint is required for the 'iopaint' backend. Install it with:\n"
                "    pip install 'manga-ai[iopaint]'\n"
                "    # or: pip install iopaint\n"
                f"(original error: {e})"
            ) from e

        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, IOPaint falling back to CPU")
            device = "cpu"
        self._resolved_device = device

        logger.info(
            "Loading IOPaint model %r on %s (strategy=%s, crop_trigger=%d, margin=%d)",
            self.model_name,
            device,
            self.hd_strategy,
            self.crop_trigger_size,
            self.crop_margin,
        )
        self._model = ModelManager(name=self.model_name, device=torch.device(device))

        strategy = {
            "crop": HDStrategy.CROP,
            "resize": HDStrategy.RESIZE,
            "original": HDStrategy.ORIGINAL,
        }.get(self.hd_strategy, HDStrategy.CROP)

        self._config = InpaintRequest(
            hd_strategy=strategy,
            hd_strategy_crop_trigger_size=self.crop_trigger_size,
            hd_strategy_crop_margin=self.crop_margin,
            hd_strategy_resize_limit=self.resize_limit,
        )

    # ------------------------------------------------------------------
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.size == 0 or mask.max() == 0:
            return image.copy()

        self._ensure_model()

        # IOPaint wants a single-channel uint8 mask, 255 = inpaint.
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = np.ascontiguousarray(mask.astype(np.uint8))
        img = np.ascontiguousarray(image.astype(np.uint8))

        result = self._model(img, mask, self._config)
        result = np.asarray(result)
        if self.result_is_bgr and result.ndim == 3 and result.shape[2] == 3:
            result = result[:, :, ::-1]
        return np.ascontiguousarray(result.astype(np.uint8))

    # ------------------------------------------------------------------
    def unload(self) -> None:
        self._model = None
        self._config = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
