"""OCR backends."""

from manga_ai.ocr.base import OCRBackend
from manga_ai.ocr.mock import MockOCR

__all__ = ["OCRBackend", "MockOCR", "create_ocr"]


def create_ocr(backend: str = "mock", **kwargs) -> OCRBackend:
    backend = (backend or "mock").lower()
    if backend == "mock":
        return MockOCR()
    if backend == "paddle":
        from manga_ai.ocr.paddle import PaddleOCRBackend
        return PaddleOCRBackend(
            lang=kwargs.get("lang", "korean"),
            use_gpu=kwargs.get("use_gpu", True),
        )
    if backend in ("manga_ocr", "manga-ocr"):
        from manga_ai.ocr.manga_ocr_backend import MangaOCRBackend
        return MangaOCRBackend()
    if backend == "easyocr":
        from manga_ai.ocr.easyocr_backend import EasyOCRBackend
        langs = kwargs.get("languages")
        if langs is None:
            lang = kwargs.get("lang") or kwargs.get("default_lang") or "en"
            langs = [lang]
        return EasyOCRBackend(
            languages=langs,
            use_gpu=kwargs.get("use_gpu", True),
            default_lang=kwargs.get("default_lang", "en"),
        )
    if backend == "router":
        from manga_ai.ocr.router import OCRRouter
        return OCRRouter(
            language_map=kwargs.get("language_map"),
            default_lang=kwargs.get("default_lang", "ko"),
            use_gpu=kwargs.get("use_gpu", True),
        )
    raise ValueError(f"Unknown OCR backend: {backend}")
