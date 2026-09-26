"""Inpainting backends."""

from __future__ import annotations

from typing import Any

from manga_ai.inpainting.base import Inpainter
from manga_ai.inpainting.mock import MockInpainter
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.inpainting")

__all__ = ["Inpainter", "MockInpainter", "create_inpainter", "lama_available"]


def lama_available() -> bool:
    try:
        import simple_lama_inpainting  # noqa: F401
        return True
    except ImportError:
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
