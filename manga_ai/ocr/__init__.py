"""OCR backends."""

from manga_ai.ocr.base import OCRBackend
from manga_ai.ocr.mock import MockOCR

__all__ = ["OCRBackend", "MockOCR", "create_ocr", "get_ocr_backend"]


def create_ocr(backend: str = "mock", **kwargs) -> OCRBackend:
    backend = (backend or "mock").lower().replace("-", "_")
    if backend == "mock":
        return MockOCR()
    if backend == "paddle":
        from manga_ai.ocr.paddle import PaddleOCRBackend
        return PaddleOCRBackend(
            lang=kwargs.get("lang", "korean"),
            use_gpu=kwargs.get("use_gpu", True),
            pad_px=kwargs.get("pad_px", 2),
        )
    if backend in ("manga_ocr", "mangaocr"):
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
    if backend in ("hybrid_qwen", "qwen", "qwen_vl", "qwen2_vl"):
        from manga_ai.ocr.hybrid_qwen import HybridQwenOCRBackend
        return HybridQwenOCRBackend(
            model=kwargs.get("model"),
            lang=kwargs.get("lang") or kwargs.get("default_lang") or "en",
            default_lang=kwargs.get("default_lang"),
            use_gpu=kwargs.get("use_gpu", True),
            device=kwargs.get("device"),
            prompt=kwargs.get("prompt"),
            max_new_tokens=kwargs.get("max_new_tokens", 128),
            load_in_4bit=kwargs.get("load_in_4bit", False),
            load_in_8bit=kwargs.get("load_in_8bit", False),
            pad_px=kwargs.get("pad_px", 4),
            min_pixels=kwargs.get("min_pixels"),
            max_pixels=kwargs.get("max_pixels"),
        )
    if backend == "router":
        from manga_ai.ocr.router import OCRRouter
        return OCRRouter(
            language_map=kwargs.get("language_map"),
            default_lang=kwargs.get("default_lang", "ko"),
            use_gpu=kwargs.get("use_gpu", True),
        )
    raise ValueError(f"Unknown OCR backend: {backend}")


def get_ocr_backend(backend: str = "mock", **kwargs) -> OCRBackend:
    """Alias for create_ocr (used by scripts/benchmark.py)."""
    return create_ocr(backend, **kwargs)
