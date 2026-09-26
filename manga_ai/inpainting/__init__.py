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
            radius=int(kwargs.get("radius", 7)),
            method=kwargs.get("method", "telea"),
            passes=int(kwargs.get("passes", 2)),
            soft_edge=int(kwargs.get("soft_edge", 3)),
            bg_fill=bool(kwargs.get("bg_fill", True)),
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
