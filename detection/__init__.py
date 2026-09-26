"""Text detection backends."""

from manga_ai.detection.base import Detector
from manga_ai.detection.mock import MockDetector

__all__ = ["Detector", "MockDetector", "create_detector"]


def create_detector(backend: str = "mock", **kwargs) -> Detector:
    backend = (backend or "mock").lower()
    if backend == "mock":
        return MockDetector()
    if backend in ("easyocr", "easy"):
        from manga_ai.detection.easyocr_detector import EasyOCRDetector
        langs = kwargs.get("languages") or kwargs.get("lang")
        if isinstance(langs, str):
            langs = [langs]
        return EasyOCRDetector(
            languages=langs or ["en"],
            use_gpu=kwargs.get("use_gpu", True),
            confidence_threshold=float(kwargs.get("confidence_threshold", 0.3)),
            min_size=int(kwargs.get("min_size", 10)),
            paragraph=bool(kwargs.get("paragraph", False)),
        )
    if backend == "paddle":
        from manga_ai.detection.paddle import PaddleDetector
        return PaddleDetector(
            lang=kwargs.get("lang", "en"),
            use_gpu=kwargs.get("use_gpu", True),
            confidence_threshold=kwargs.get("confidence_threshold", 0.5),
        )
    raise ValueError(
        f"Unknown detector backend: {backend}. "
        f"Use: mock | easyocr | paddle"
    )
