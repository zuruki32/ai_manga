"""Translation backends."""

from __future__ import annotations

from pathlib import Path

from manga_ai.logging import get_logger
from manga_ai.translation.base import Translator
from manga_ai.translation.mock import MockTranslator

logger = get_logger("manga_ai.translation")

__all__ = ["Translator", "MockTranslator", "create_translator"]


def _looks_like_gguf(model: str | None) -> bool:
    if not model:
        return False
    low = str(model).lower().replace("\\", "/")
    return low.endswith(".gguf") or "gguf" in Path(low).name or low.rstrip("/").endswith("-gguf")


def _hf_sibling_of_gguf(model: str) -> str | None:
    """models/gemma-3-4b-persian-gguf → models/gemma-3-4b-persian if that folder exists."""
    p = Path(model)
    name = p.name
    if name.lower().endswith("-gguf"):
        sibling = p.with_name(name[: -len("-gguf")])
        if sibling.is_dir():
            return str(sibling)
    if name.lower().endswith(".gguf"):
        parent = p.parent
        if parent.name.lower().endswith("-gguf"):
            sibling = parent.with_name(parent.name[: -len("-gguf")])
            if sibling.is_dir():
                return str(sibling)
    return None


def create_translator(backend: str = "mock", **kwargs) -> Translator:
    backend = (backend or "mock").lower()
    model = kwargs.get("model")

    # Never silently run broken GGUF when config asked for HuggingFace Gemma
    if backend in ("huggingface", "hf", "local", "transformers", "gemma") and _looks_like_gguf(
        model
    ):
        sibling = _hf_sibling_of_gguf(str(model))
        if sibling:
            logger.warning(
                f"translation.model looks like GGUF ({model}) but backend={backend}; "
                f"using HF folder instead: {sibling}"
            )
            model = sibling
            kwargs = {**kwargs, "model": sibling}
        else:
            raise ValueError(
                f"translation.backend={backend} but model is GGUF path: {model}\n"
                "Use the complete HF folder, e.g.:\n"
                "  translation:\n"
                "    backend: huggingface\n"
                "    model: models/gemma-3-4b-persian\n"
                "Or set backend: llama_cpp explicitly (not recommended on Windows)."
            )

    if backend == "mock":
        return MockTranslator()
    if backend in ("openai_compatible", "openai", "http"):
        from manga_ai.translation.openai_compatible import OpenAICompatibleTranslator
        return OpenAICompatibleTranslator(
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model"),
            temperature=kwargs.get("temperature", 0.2),
            max_retries=kwargs.get("max_retries", 3),
        )
    if backend in ("huggingface", "hf", "local", "transformers", "gemma"):
        from manga_ai.translation.huggingface_local import HuggingFaceTranslator
        logger.info(f"Translator backend=huggingface model={model}")
        return HuggingFaceTranslator(
            model=model,
            device=kwargs.get("device", "cuda"),
            max_length=kwargs.get("max_length", 256),
            batch_size=kwargs.get("batch_size", 8),
            source_language=kwargs.get("source_language", "en"),
            target_language=kwargs.get("target_language", "fa"),
            pivot_model=kwargs.get("pivot_model"),
            load_in_4bit=bool(kwargs.get("load_in_4bit", False)),
            load_in_8bit=bool(kwargs.get("load_in_8bit", False)),
            max_new_tokens=int(kwargs.get("max_new_tokens", 128)),
            local_files_only=bool(kwargs.get("local_files_only", False)),
            validate=bool(kwargs.get("validate", True)),
            max_retries=int(kwargs.get("max_retries", 1)),
            style=str(kwargs.get("style", "colloquial_fa")),
            extra_instructions=str(kwargs.get("extra_instructions") or ""),
        )
    if backend in ("llama_cpp", "llamacpp", "gguf"):
        from manga_ai.translation.llama_cpp_backend import LlamaCppTranslator
        logger.warning(
            f"Translator backend=llama_cpp model={model} — "
            "prefer backend=huggingface + models/gemma-3-4b-persian on Windows"
        )
        return LlamaCppTranslator(
            model=model,
            n_gpu_layers=int(kwargs.get("n_gpu_layers", -1)),
            n_ctx=int(kwargs.get("n_ctx", 2048)),
            max_tokens=int(kwargs.get("max_tokens", kwargs.get("max_new_tokens", 128))),
            n_threads=kwargs.get("n_threads"),
            temperature=float(kwargs.get("temperature", 0.1)),
            chat_format=kwargs.get("chat_format"),
            source_language=kwargs.get("source_language", "en"),
            target_language=kwargs.get("target_language", "fa"),
            verbose=bool(kwargs.get("verbose", False)),
        )
    raise ValueError(f"Unknown translator backend: {backend}")
