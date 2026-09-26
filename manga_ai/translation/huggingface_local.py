"""Local Hugging Face translation (MarianMT, M2M100, Gemma, Llama).

Does NOT rely on pipeline("translation") — newer transformers dropped that task.
Uses AutoModelForSeq2SeqLM / AutoModelForCausalLM / Gemma3 directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from manga_ai.translation.base import Translator
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.translation.huggingface")

DEFAULT_MODELS = {
    ("en", "fa"): "Helsinki-NLP/opus-mt-en-fa",
    ("en", "per"): "Helsinki-NLP/opus-mt-en-fa",
    ("ko", "en"): "Helsinki-NLP/opus-mt-ko-en",
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
}

CAUSAL_PREFIXES = ("gemma", "llama", "mistral", "qwen", "phi", "gpt")


def _is_causal(model_name: str) -> bool:
    low = (model_name or "").lower().replace("\\", "/")
    # path folders
    if any(p in low for p in CAUSAL_PREFIXES):
        return True
    return False


def _is_gemma3(model_name: str) -> bool:
    low = (model_name or "").lower().replace("\\", "/")
    return "gemma-3" in low or "gemma3" in low or "gemma_3" in low


def _has_accelerate() -> bool:
    try:
        import accelerate  # noqa: F401
        return True
    except Exception:
        return False


def _pick_causal_class(model_id: str) -> Type[Any]:
    """Prefer Gemma3 class when available for local gemma-3 folders."""
    if _is_gemma3(model_id):
        try:
            from transformers import Gemma3ForCausalLM

            return Gemma3ForCausalLM
        except ImportError:
            pass
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def _load_gemma3_fallback(model_id: str, kwargs: Dict[str, Any]):
    """Last-resort loaders for multimodal Gemma-3 configs."""
    try:
        from transformers import Gemma3ForConditionalGeneration

        logger.info("Falling back to Gemma3ForConditionalGeneration")
        return Gemma3ForConditionalGeneration.from_pretrained(model_id, **kwargs)
    except Exception as e:
        logger.warning(f"Gemma3ForConditionalGeneration failed: {e}")
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_id, **kwargs)


class HuggingFaceTranslator(Translator):
    name = "huggingface"

    def __init__(
        self,
        model: Optional[str] = None,
        device: str = "cuda",
        max_length: int = 256,
        batch_size: int = 8,
        source_language: str = "en",
        target_language: str = "fa",
        pivot_model: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_new_tokens: int = 128,
        local_files_only: bool = False,
        **kwargs: Any,
    ):
        self.model_name = model
        self.device = device
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.source_language = source_language
        self.target_language = target_language
        self.pivot_model_name = pivot_model
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.max_new_tokens = max_new_tokens
        # Auto: if model path exists on disk, stay offline
        if local_files_only:
            self.local_files_only = True
        elif model and Path(model).expanduser().exists():
            self.local_files_only = True
        else:
            self.local_files_only = False

        self._model = None
        self._tokenizer = None
        self._pivot_model = None
        self._pivot_tokenizer = None
        self._mode = "seq2seq"  # or "causal"
        self._is_m2m = False

    def _resolve_model(self, src: str, tgt: str) -> str:
        if self.model_name:
            return self.model_name
        key = (src[:2].lower(), tgt[:2].lower())
        return DEFAULT_MODELS.get(key, "Helsinki-NLP/opus-mt-en-fa")

    def _torch_device(self):
        import torch
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _load_seq2seq(self, model_id: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info(f"Loading seq2seq model: {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id)
        kwargs: Dict[str, Any] = {}
        if self.load_in_4bit or self.load_in_8bit:
            if not _has_accelerate():
                raise ImportError("pip install accelerate bitsandbytes  # required for 4/8-bit")
            from transformers import BitsAndBytesConfig
            if self.load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        else:
            if _has_accelerate() and self.device.startswith("cuda"):
                kwargs["device_map"] = "auto"
                kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.float16 if self.device.startswith("cuda") else torch.float32

        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)
        if "device_map" not in kwargs:
            model = model.to(self._torch_device())
        model.eval()

        low = model_id.lower().replace("\\", "/")
        is_m2m = "m2m100" in low or "m2m_100" in low
        return model, tok, is_m2m

    def _load_causal(self, model_id: str):
        import torch
        from transformers import AutoTokenizer

        logger.info(
            f"Loading causal model: {model_id} "
            f"(4bit={self.load_in_4bit}, 8bit={self.load_in_8bit}, "
            f"local_files_only={self.local_files_only})"
        )
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        use_device_map = False

        if self.load_in_4bit or self.load_in_8bit:
            if not _has_accelerate():
                raise ImportError(
                    "pip install accelerate bitsandbytes\n"
                    "Required for 4-bit/8-bit loading."
                )
            from transformers import BitsAndBytesConfig
            if self.load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
            use_device_map = True
        else:
            if _has_accelerate() and self.device.startswith("cuda"):
                kwargs["device_map"] = "auto"
                kwargs["torch_dtype"] = torch.bfloat16 if _is_gemma3(model_id) else torch.float16
                use_device_map = True
            else:
                # no accelerate: load then .to(device)
                kwargs["torch_dtype"] = (
                    torch.bfloat16
                    if _is_gemma3(model_id)
                    else (torch.float16 if self.device.startswith("cuda") else torch.float32)
                )

        model_cls = _pick_causal_class(model_id)
        try:
            model = model_cls.from_pretrained(model_id, **kwargs)
        except Exception as e1:
            logger.warning(f"{model_cls.__name__} failed ({e1}), trying AutoModelForCausalLM")
            from transformers import AutoModelForCausalLM

            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            except Exception as e2:
                logger.warning(f"AutoModelForCausalLM failed ({e2})")
                if _is_gemma3(model_id):
                    model = _load_gemma3_fallback(model_id, kwargs)
                else:
                    from transformers import AutoModel

                    model = AutoModel.from_pretrained(model_id, **kwargs)

        if not use_device_map:
            model = model.to(self._torch_device())
        model.eval()
        return model, tok

    def _ensure_pipeline(self, src: str, tgt: str) -> None:
        if self._model is not None:
            return
        model_id = self._resolve_model(src, tgt)
        if _is_causal(model_id):
            self._model, self._tokenizer = self._load_causal(model_id)
            self._mode = "causal"
            self._is_m2m = False
        else:
            self._model, self._tokenizer, self._is_m2m = self._load_seq2seq(model_id)
            self._mode = "seq2seq"

        if self.pivot_model_name and self._mode == "seq2seq":
            self._pivot_model, self._pivot_tokenizer, _ = self._load_seq2seq(
                self.pivot_model_name
            )

    def _seq2seq_generate(self, texts: List[str], model, tok, is_m2m: bool, src: str, tgt: str) -> List[str]:
        import torch

        device = next(model.parameters()).device
        # M2M100 language codes
        if is_m2m:
            src_lang = {"en": "en", "fa": "fa", "pe": "fa"}.get(src[:2], src[:2])
            tgt_lang = {"en": "en", "fa": "fa", "pe": "fa"}.get(tgt[:2], tgt[:2])
            if hasattr(tok, "src_lang"):
                tok.src_lang = src_lang
            forced_bos = None
            if hasattr(tok, "get_lang_id"):
                try:
                    forced_bos = tok.get_lang_id(tgt_lang)
                except Exception:
                    forced_bos = None
            elif hasattr(tok, "lang_code_to_id") and tgt_lang in getattr(tok, "lang_code_to_id", {}):
                forced_bos = tok.lang_code_to_id[tgt_lang]
        else:
            forced_bos = None

        inputs = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_length": self.max_length,
            "num_beams": 4,
        }
        if forced_bos is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        return tok.batch_decode(out, skip_special_tokens=True)

    def _translate_causal_one(self, text: str, src: str, tgt: str) -> str:
        if not text.strip():
            return ""
        import torch

        lang_names = {
            "en": "English",
            "fa": "Persian",
            "pe": "Persian",
            "ko": "Korean",
            "ja": "Japanese",
            "zh": "Chinese",
        }
        src_name = lang_names.get(src[:2], src)
        tgt_name = lang_names.get(tgt[:2], tgt)
        tok = self._tokenizer
        model_l = (self.model_name or "").lower().replace("\\", "/")

        # Sheikhaei dedicated EN-FA format
        if "english-persian" in model_l or "en-fa" in model_l or "llama-en-fa" in model_l or "llama-3.2-1b" in model_l:
            prompt = f"### English:\n{text}\n### Persian:\n"
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        else:
            user_msg = (
                f"You are a professional manhwa/comic translator.\n"
                f"Translate the following {src_name} dialogue to natural colloquial {tgt_name}.\n"
                f"Keep character names unchanged unless a glossary replacement is already applied.\n"
                f"Return only the translation, nothing else.\n\n{text}"
            )
            try:
                messages = [{"role": "user", "content": user_msg}]
                prompt = tok.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
            except Exception:
                prompt = (
                    f"<start_of_turn>user\n{user_msg}\n"
                    f"<end_of_turn>\n<start_of_turn>model\n"
                )
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)

        try:
            dev = next(self._model.parameters()).device
            inputs = {k: v.to(dev) for k, v in inputs.items()}
        except Exception:
            pass

        pad_id = tok.eos_token_id or getattr(tok, "pad_token_id", None)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return tok.decode(gen, skip_special_tokens=True).strip()

    def translate_chapter(
        self,
        regions: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        src = (context.get("source_language") or self.source_language or "en").lower()
        tgt = (context.get("target_language") or self.target_language or "fa").lower()
        glossary = context.get("glossary") or {}

        need_pivot = False
        if src[:2] not in ("en",) and tgt[:2] in ("fa", "pe") and src[:2] != "fa":
            # optional pivot for ko/ja → en → fa if pivot set
            if self.pivot_model_name:
                need_pivot = True

        self._ensure_pipeline(src if not need_pivot else "en", tgt)

        texts = [r.get("source_text") or "" for r in regions]
        placeholders: List[Dict[str, str]] = []
        protected: List[str] = []
        for text in texts:
            mapping = {}
            t = text
            for i, (term, _) in enumerate(glossary.items()):
                if term and term in t:
                    ph = f"__GLOSS_{i}__"
                    t = t.replace(term, ph)
                    mapping[ph] = term
            placeholders.append(mapping)
            protected.append(t)

        translated = [""] * len(protected)
        non_empty_idx = [i for i, t in enumerate(protected) if t.strip()]
        non_empty_texts = [protected[i] for i in non_empty_idx]
        results_text: List[str] = []

        if non_empty_texts:
            if self._mode == "causal":
                for t in non_empty_texts:
                    try:
                        results_text.append(self._translate_causal_one(t, src, tgt))
                    except Exception as e:
                        logger.error(f"Causal translation failed: {e}")
                        results_text.append(t)
            else:
                for start in range(0, len(non_empty_texts), self.batch_size):
                    batch = non_empty_texts[start : start + self.batch_size]
                    try:
                        if need_pivot and self._pivot_model is not None:
                            mid = self._seq2seq_generate(
                                batch, self._pivot_model, self._pivot_tokenizer,
                                False, src, "en",
                            )
                            out = self._seq2seq_generate(
                                mid, self._model, self._tokenizer,
                                self._is_m2m, "en", tgt,
                            )
                        else:
                            out = self._seq2seq_generate(
                                batch, self._model, self._tokenizer,
                                self._is_m2m, src, tgt,
                            )
                        results_text.extend(out)
                    except Exception as e:
                        logger.error(f"HF translation batch failed: {e}")
                        results_text.extend(batch)

            for j, idx in enumerate(non_empty_idx):
                translated[idx] = (
                    results_text[j] if j < len(results_text) else protected[idx]
                )

        for i, text in enumerate(translated):
            for ph, term in placeholders[i].items():
                text = text.replace(ph, term)
            for term, fa in glossary.items():
                if term and term in text:
                    text = text.replace(term, fa)
            translated[i] = text.strip()

        return [
            {"region_id": regions[i]["region_id"], "text": translated[i]}
            for i in range(len(regions))
        ]

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._pivot_model = None
        self._pivot_tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("HF translator unloaded")
