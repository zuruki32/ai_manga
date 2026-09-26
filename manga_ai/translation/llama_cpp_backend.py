"""GGUF translation via llama-cpp-python (GPU offload, no Ollama required).

Install:
  pip install llama-cpp-python
  # CUDA wheel (Windows/Linux), see https://github.com/abetlen/llama-cpp-python#installation-with-cuda

Example config:
  translation:
    backend: llama_cpp
    model: models/gemma-3-4b-persian-q4.gguf   # path to .gguf file
    n_gpu_layers: -1    # -1 = all layers on GPU
    n_ctx: 2048
    max_tokens: 128
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from manga_ai.translation.base import Translator
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.translation.llama_cpp")


class LlamaCppTranslator(Translator):
    name = "llama_cpp"

    def __init__(
        self,
        model: Optional[str] = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        max_tokens: int = 128,
        n_threads: Optional[int] = None,
        temperature: float = 0.1,
        chat_format: Optional[str] = None,
        source_language: str = "en",
        target_language: str = "fa",
        verbose: bool = False,
    ):
        self.model_path = model
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.n_threads = n_threads
        self.temperature = temperature
        self.chat_format = chat_format
        self.source_language = source_language
        self.target_language = target_language
        self.verbose = verbose
        self._llm = None

    def _resolve_gguf_path(self) -> Path:
        """Accept a .gguf file or a folder containing one."""
        if not self.model_path:
            raise ValueError(
                "translation.model must be a path to a .gguf file or folder "
                "(e.g. models/gemma-3-4b-persian-gguf)"
            )
        path = Path(self.model_path).expanduser()
        if path.is_file() and path.suffix.lower() == ".gguf":
            return path.resolve()
        if path.is_dir():
            ggufs = sorted(path.glob("*.gguf")) + sorted(path.glob("**/*.gguf"))
            # de-dupe while keeping order
            seen = set()
            uniq = []
            for g in ggufs:
                key = str(g.resolve())
                if key not in seen:
                    seen.add(key)
                    uniq.append(g)
            if not uniq:
                raise FileNotFoundError(f"No .gguf files in {path.resolve()}")
            # Prefer smaller quantized variants for 6GB GPUs
            def score(p: Path) -> tuple:
                n = p.name.lower()
                pref = 50
                for i, tag in enumerate(
                    ("q4_k_m", "q4_k_s", "q4_0", "q5_k_m", "q5_0", "q8_0", "q6_k", "f16", "f32")
                ):
                    if tag in n:
                        pref = i
                        break
                return (pref, p.stat().st_size, p.name)

            chosen = sorted(uniq, key=score)[0]
            logger.info(f"Resolved GGUF folder {path} → {chosen.name}")
            return chosen.resolve()
        raise FileNotFoundError(f"GGUF not found: {path.resolve()}")

    def _ensure_model(self):
        if self._llm is not None:
            return
        path = self._resolve_gguf_path()

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for GGUF backend.\n"
                "  pip install llama-cpp-python\n"
                "  # CUDA: see https://github.com/abetlen/llama-cpp-python#installation-with-cuda"
            ) from e

        kwargs: Dict[str, Any] = {
            "model_path": str(path),
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": self.verbose,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads
        if self.chat_format:
            kwargs["chat_format"] = self.chat_format

        logger.info(
            f"Loading GGUF: {path.name} (n_gpu_layers={self.n_gpu_layers}, n_ctx={self.n_ctx})"
        )
        self._llm = Llama(**kwargs)

    def _lang_name(self, code: str) -> str:
        return {
            "en": "English",
            "fa": "Persian",
            "pe": "Persian",
            "ko": "Korean",
            "ja": "Japanese",
            "zh": "Chinese",
        }.get(code[:2].lower(), code)

    def _build_prompt(self, text: str, src: str, tgt: str) -> str:
        src_n = self._lang_name(src)
        tgt_n = self._lang_name(tgt)
        return (
            f"You are a professional manhwa/comic translator.\n"
            f"Translate the following {src_n} dialogue to natural colloquial {tgt_n}.\n"
            f"Keep character names unchanged unless a glossary replacement is already applied.\n"
            f"Return only the translation, nothing else.\n\n"
            f"{text}"
        )

    def _translate_one(self, text: str, src: str, tgt: str) -> str:
        if not text.strip():
            return ""
        self._ensure_model()
        user = self._build_prompt(text, src, tgt)

        # Prefer chat API when available
        try:
            out = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": user}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return (out["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            prompt = f"<start_of_turn>user\n{user}\n<end_of_turn>\n<start_of_turn>model\n"
            out = self._llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["<end_of_turn>", "<start_of_turn>"],
            )
            return (out["choices"][0]["text"] or "").strip()

    def translate_chapter(
        self,
        regions: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        src = (context.get("source_language") or self.source_language or "en").lower()
        tgt = (context.get("target_language") or self.target_language or "fa").lower()
        glossary = context.get("glossary") or {}

        results = []
        for r in regions:
            text = r.get("source_text") or ""
            # glossary placeholders
            mapping = {}
            for i, (term, _) in enumerate(glossary.items()):
                if term and term in text:
                    ph = f"__GLOSS_{i}__"
                    text = text.replace(term, ph)
                    mapping[ph] = term
            try:
                translated = self._translate_one(text, src, tgt)
            except Exception as e:
                logger.error(f"GGUF translate failed for {r.get('region_id')}: {e}")
                translated = r.get("source_text") or ""
            for ph, term in mapping.items():
                translated = translated.replace(ph, term)
            for term, fa in glossary.items():
                if term and term in translated:
                    translated = translated.replace(term, fa)
            results.append({"region_id": r["region_id"], "text": translated.strip()})
        return results

    def unload(self) -> None:
        self._llm = None
        logger.info("llama_cpp translator unloaded")
