"""Inpainting backends."""

from manga_ai.inpainting.base import Inpainter
from manga_ai.inpainting.mock import MockInpainter

__all__ = ["Inpainter", "MockInpainter", "create_inpainter"]


def create_inpainter(backend: str = "mock", **kwargs) -> Inpainter:
    backend = (backend or "mock").lower()
    if backend == "mock":
        return MockInpainter()
    if backend == "opencv":
        from manga_ai.inpainting.opencv_backend import OpenCVInpainter
        return OpenCVInpainter(
            radius=kwargs.get("radius", 5),
            method=kwargs.get("method", "telea"),
        )
    if backend == "lama":
        from manga_ai.inpainting.lama import LaMaInpainter
        return LaMaInpainter(
            device=kwargs.get("device", "cuda"),
            tile_size=kwargs.get("tile_size", 768),
            overlap=kwargs.get("overlap", 64),
            precision=kwargs.get("precision", "fp16"),
        )
    raise ValueError(f"Unknown inpainter backend: {backend}")
