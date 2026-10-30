"""Inpainting backends."""

from __future__ import annotations

from typing import Any

from manga_ai.inpainting.base import Inpainter
from manga_ai.inpainting.mock import MockInpainter
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.inpainting")

__all__ = [
    "Inpainter",
    "MockInpainter",
    "create_inpainter",
    "lama_available",
    "iopaint_available",
]


def lama_available() -> bool:
    try:
        import simple_lama_inpainting  # noqa: F401
        return True
    except ImportError:
        return False


def iopaint_available() -> bool:
    # Catch any import-time failure (e.g. a huggingface-hub version clash),
    # not just ImportError, so a broken install degrades gracefully to OpenCV
    # instead of crashing the pipeline.
    try:
        import iopaint  # noqa: F401
        return True
    except Exception as e:  # pragma: no cover - env dependent
        logger.warning("IOPaint import failed (%s) — cleaning will use OpenCV", e)
        return False


def _opencv_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "radius": int(kwargs.get("radius", 7)),
        "method": kwargs.get("method", "telea"),
        "passes": int(kwargs.get("passes", 2)),
        "soft_edge": int(kwargs.get("soft_edge", 3)),
        "bg_fill": bool(kwargs.get("bg_fill", True)),
    }


def create_inpainter(backend: str = "mock", **kwargs) -> Inpainter:
    backend = (backend or "mock").lower()
    if backend == "mock":
        return MockInpainter()
    if backend == "opencv":
        from manga_ai.inpainting.opencv_backend import OpenCVInpainter
        return OpenCVInpainter(**_opencv_kwargs(kwargs))
    if backend in ("anime-lama", "lama-anime", "animelama", "anime_lama"):
        # Self-contained color anime/manga LaMa — no iopaint/diffusers/hub deps.
        from manga_ai.inpainting.anime_lama import AnimeLaMaInpainter
        return AnimeLaMaInpainter(
            device=kwargs.get("device", "cuda"),
            crop_margin=int(kwargs.get("crop_margin", 128)),
            model_path=kwargs.get("model_path"),
        )
    if backend in ("iopaint", "manga"):
        if not iopaint_available():
            logger.warning(
                "IOPaint requested but not installed — using OpenCV cleaner "
                "(low quality on manga). Install with: "
                "pip install 'manga-ai[iopaint]'   # or: pip install iopaint"
            )
            from manga_ai.inpainting.opencv_backend import OpenCVInpainter
            return OpenCVInpainter(**_opencv_kwargs(kwargs))
        from manga_ai.inpainting.iopaint_backend import IOPaintInpainter
        # backend "manga" is shorthand for iopaint + the manga model
        default_model = "manga" if backend == "manga" else kwargs.get("model", "manga")
        return IOPaintInpainter(
            model_name=kwargs.get("model", default_model),
            device=kwargs.get("device", "cuda"),
            crop_trigger_size=int(kwargs.get("crop_trigger_size", 1024)),
            crop_margin=int(kwargs.get("crop_margin", 128)),
            resize_limit=int(kwargs.get("resize_limit", 2048)),
            hd_strategy=kwargs.get("hd_strategy", "crop"),
            result_is_bgr=bool(kwargs.get("result_is_bgr", True)),
        )
    if backend == "lama":
        if not lama_available():
            logger.warning(
                "LaMa requested but simple-lama-inpainting is not installed — "
                "using OpenCV cleaner. Install with: "
                "pip install 'manga-ai[lama]'   # or: pip install simple-lama-inpainting --no-deps"
            )
            from manga_ai.inpainting.opencv_backend import OpenCVInpainter
            return OpenCVInpainter(**_opencv_kwargs(kwargs))
        from manga_ai.inpainting.lama import LaMaInpainter
        return LaMaInpainter(
            device=kwargs.get("device", "cuda"),
            tile_size=kwargs.get("tile_size", 768),
            overlap=kwargs.get("overlap", 64),
            precision=kwargs.get("precision", "fp16"),
        )
    raise ValueError(f"Unknown inpainter backend: {backend}")
