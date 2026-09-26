"""Translation backends."""

from manga_ai.translation.base import Translator
from manga_ai.translation.mock import MockTranslator

__all__ = ["Translator", "MockTranslator", "create_translator"]


def create_translator(backend: str = "mock", **kwargs) -> Translator:
    backend = (backend or "mock").lower()
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
        return HuggingFaceTranslator(
            model=kwargs.get("model"),
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
        return LlamaCppTranslator(
            model=kwargs.get("model"),
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
