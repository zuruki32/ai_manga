"""Local Hugging Face translation (MarianMT, M2M100, Gemma, Llama).

Does NOT rely on pipeline("translation") — newer transformers dropped that task.
Uses AutoModelForSeq2SeqLM / AutoModelForCausalLM / Gemma3 directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

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


def _read_config(model_id: str) -> Dict[str, Any]:
    path = Path(model_id).expanduser()
    cfg_path = path / "config.json" if path.is_dir() else None
    if cfg_path and cfg_path.is_file():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not read {cfg_path}: {e}")
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    except Exception:
        return {}


def _architectures(cfg: Dict[str, Any]) -> List[str]:
    arch = cfg.get("architectures") or []
    if isinstance(arch, str):
        return [arch]
    return [str(a) for a in arch]


def _is_multimodal_gemma3(model_id: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    cfg = cfg or _read_config(model_id)
    arch = " ".join(_architectures(cfg)).lower()
    if "conditionalgeneration" in arch or "gemma3forconditional" in arch:
        return True
    if cfg.get("vision_config") or cfg.get("text_config"):
        # Gemma-3 4B/12B/27B IT ship as multimodal configs
        return True
    return False


def _pick_model_class(model_id: str) -> Type[Any]:
    """Pick the correct HF class for a local/remote checkpoint."""
    cfg = _read_config(model_id)
    archs = _architectures(cfg)
    arch_l = " ".join(archs).lower()

    if _is_gemma3(model_id) or "gemma3" in arch_l:
        if _is_multimodal_gemma3(model_id, cfg) or "conditional" in arch_l:
            try:
                from transformers import Gemma3ForConditionalGeneration

                logger.info(
                    f"Using Gemma3ForConditionalGeneration "
                    f"(architectures={archs or 'multimodal config'})"
                )
                return Gemma3ForConditionalGeneration
            except ImportError as e:
                raise ImportError(
                    "Gemma3ForConditionalGeneration requires a recent transformers.\n"
                    "  pip install -U 'transformers>=4.57.0'"
                ) from e
        try:
            from transformers import Gemma3ForCausalLM

            logger.info(f"Using Gemma3ForCausalLM (architectures={archs})")
            return Gemma3ForCausalLM
        except ImportError:
            pass

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def _assert_weights_look_loaded(model: Any, model_id: str) -> None:
    """Fail fast if we loaded the wrong architecture (empty / random heads)."""
    import torch

    # Collect a few key tensors that must exist for a usable text model
    state = model.state_dict()
    keys = list(state.keys())
    has_lang = any(k.startswith("language_model.") for k in keys)
    has_text = any(
        k.startswith("model.embed_tokens")
        or k.startswith("model.layers")
        or k.endswith("lm_head.weight")
        for k in keys
    )
    if not has_lang and not has_text:
        raise RuntimeError(
            f"Loaded model has no language weights (keys={len(keys)}). "
            f"Wrong class for {model_id}?"
        )

    # Spot-check that some weight is non-zero / finite (not freshly init empty shell)
    sample_key = None
    for cand in (
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "lm_head.weight",
    ):
        if cand in state:
            sample_key = cand
            break
    if sample_key is None:
        # any 2d float tensor
        for k, v in state.items():
            if getattr(v, "ndim", 0) == 2 and v.numel() > 1000:
                sample_key = k
                break
    if sample_key is None:
        raise RuntimeError(f"Could not find embed/lm_head weights in {model_id}")

    t = state[sample_key].detach()
    if t.is_meta:
        raise RuntimeError(f"Weight {sample_key} is still on meta device — load failed")
    # dequantized peek for bitsandbytes can be awkward; check finite + non-all-zero on float
    try:
        flat = t.float().reshape(-1)[:4096]
        if not torch.isfinite(flat).all():
            raise RuntimeError(f"Weight {sample_key} has non-finite values")
        if float(flat.abs().sum()) == 0.0:
            raise RuntimeError(
                f"Weight {sample_key} is all zeros — checkpoint did not load. "
                f"Your folder may be incomplete or the wrong class was used for {model_id}."
            )
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Could not validate weight {sample_key}: {e}")

    logger.info(f"Weight check OK ({sample_key}, total_tensors={len(keys)})")


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
        self._is_gemma3_mm = False

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

    def _load_kwargs(self, model_id: str) -> Dict[str, Any]:
        import torch

        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        # Gemma3 prefers bf16 compute
        compute_dtype = torch.bfloat16 if _is_gemma3(model_id) else torch.float16

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
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = compute_dtype
            if _has_accelerate() and self.device.startswith("cuda"):
                kwargs["device_map"] = "auto"
        return kwargs

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
        from transformers import AutoTokenizer

        cfg = _read_config(model_id)
        self._is_gemma3_mm = _is_multimodal_gemma3(model_id, cfg)
        model_cls = _pick_model_class(model_id)

        logger.info(
            f"Loading causal/VLM model: {model_id} via {model_cls.__name__} "
            f"(4bit={self.load_in_4bit}, 8bit={self.load_in_8bit}, "
            f"local_files_only={self.local_files_only}, multimodal={self._is_gemma3_mm})"
        )
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        kwargs = self._load_kwargs(model_id)
        use_device_map = "device_map" in kwargs

        try:
            model = model_cls.from_pretrained(model_id, **kwargs)
        except Exception as e1:
            # Wrong class guess — try the other Gemma3 class once
            logger.warning(f"{model_cls.__name__} failed ({e1})")
            alt = None
            if "CausalLM" in model_cls.__name__:
                try:
                    from transformers import Gemma3ForConditionalGeneration as alt
                except ImportError:
                    alt = None
            elif "Conditional" in model_cls.__name__:
                try:
                    from transformers import Gemma3ForCausalLM as alt
                except ImportError:
                    alt = None
            if alt is None:
                raise
            logger.info(f"Retrying with {alt.__name__}")
            model = alt.from_pretrained(model_id, **kwargs)
            self._is_gemma3_mm = "Conditional" in alt.__name__

        if not use_device_map:
            model = model.to(self._torch_device())
        model.eval()
        _assert_weights_look_loaded(model, model_id)

        # Smoke test: one short generate so FP4/device errors surface before the chapter loop
        try:
            self._smoke_generate(model, tok)
        except Exception as e:
            raise RuntimeError(
                f"Model loaded but generate() failed for {model_id}: {e}\n"
                "Common fixes:\n"
                "  - pip install -U 'transformers>=4.57.0' accelerate bitsandbytes\n"
                "  - Ensure models/gemma-3-4b-persian is a complete HF snapshot "
                "(config.json + safetensors), not a partial download"
            ) from e
        return model, tok

    def _smoke_generate(self, model: Any, tok: Any) -> None:
        import torch

        prompt = "Translate to Persian: Hi"
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            text = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
        inputs = tok(text, return_tensors="pt")
        try:
            dev = next(model.parameters()).device
            inputs = {k: v.to(dev) for k, v in inputs.items()}
        except Exception:
            pass
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tok.eos_token_id or tok.pad_token_id,
            )
        _ = tok.decode(out[0], skip_special_tokens=True)
        logger.info("Generate smoke test OK")

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

    def _build_causal_messages(self, text: str, src_name: str, tgt_name: str) -> Any:
        user_msg = (
            f"You are a professional manhwa/comic translator.\n"
            f"Translate the following {src_name} dialogue to natural colloquial {tgt_name}.\n"
            f"Keep character names unchanged unless a glossary replacement is already applied.\n"
            f"Return only the translation, nothing else.\n\n{text}"
        )
        # Gemma3 multimodal chat templates often want typed content blocks
        if self._is_gemma3_mm:
            return [{"role": "user", "content": [{"type": "text", "text": user_msg}]}]
        return [{"role": "user", "content": user_msg}]

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

        if "english-persian" in model_l or "en-fa" in model_l or "llama-en-fa" in model_l or "llama-3.2-1b" in model_l:
            prompt = f"### English:\n{text}\n### Persian:\n"
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        else:
            messages = self._build_causal_messages(text, src_name, tgt_name)
            try:
                prompt = tok.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
            except Exception:
                # typed content failed — fall back to plain string message
                if isinstance(messages[0]["content"], list):
                    plain = messages[0]["content"][0]["text"]
                    messages = [{"role": "user", "content": plain}]
                    try:
                        prompt = tok.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False
                        )
                    except Exception:
                        prompt = (
                            f"<start_of_turn>user\n{plain}\n"
                            f"<end_of_turn>\n<start_of_turn>model\n"
                        )
                else:
                    prompt = (
                        f"<start_of_turn>user\n{messages[0]['content']}\n"
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
                    # Fail fast — do not silently dump English for the whole chapter
                    results_text.append(self._translate_causal_one(t, src, tgt))
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
                        raise

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
