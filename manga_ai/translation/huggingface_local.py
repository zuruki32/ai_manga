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

CAUSAL_PREFIXES = (
    "gemma", "llama", "mistral", "qwen", "phi", "gpt",
    "hunyuan", "aya", "cohere", "seed-x", "tower",
)


def _is_causal(model_name: str) -> bool:
    low = (model_name or "").lower().replace("\\", "/")
    if any(p in low for p in CAUSAL_PREFIXES):
        return True
    return False


def _is_hunyuan_mt(model_name: str) -> bool:
    """Tencent Hunyuan-MT: dedicated translator with a fixed prompt template."""
    low = (model_name or "").lower().replace("\\", "/")
    return "hunyuan-mt" in low or "hunyuan_mt" in low


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
        validate: bool = True,
        max_retries: int = 1,
        style: str = "colloquial_fa",
        extra_instructions: str = "",
        context_window: int = 1,
        mt_target_label: str = "Persian",
        **kwargs: Any,
    ):
        self.model_name = model
        self.context_window = max(0, int(context_window))
        self.mt_target_label = (mt_target_label or "Persian").strip()
        self.device = device
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.source_language = source_language
        self.target_language = target_language
        self.pivot_model_name = pivot_model
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.max_new_tokens = max_new_tokens
        self.validate = validate
        self.max_retries = max(0, int(max_retries))
        self.style = (style or "colloquial_fa").lower()
        self.extra_instructions = (extra_instructions or "").strip()
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
        # Gemma3 / Hunyuan are trained in bf16; fall back to fp16 on GPUs without bf16
        wants_bf16 = _is_gemma3(model_id) or _is_hunyuan_mt(model_id)
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if (wants_bf16 and bf16_ok) else torch.float16

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

    def _build_causal_messages(
        self,
        text: str,
        src_name: str,
        tgt_name: str,
        *,
        keep_names: Optional[List[str]] = None,
        strict: bool = False,
        prev_text: str = "",
        next_text: str = "",
    ) -> Any:
        name_line = ""
        if keep_names:
            uniq = []
            seen = set()
            for n in keep_names:
                key = n.lower()
                if n and key not in seen:
                    seen.add(key)
                    uniq.append(n)
            if uniq:
                name_line = (
                    "توکن‌های داخل ⟦ ⟧ رو عیناً تو خروجی نگه دار و ترجمه/حذف نکن "
                    f"(بعداً با اسم جایگزین می‌شن): {', '.join(uniq)}\n"
                )
        strict_line = ""
        if strict:
            strict_line = (
                "هشدار: ترجمه قبلی ناقص بود یا کلمه انگلیسی داشت. "
                "این بار همه‌ی معنی رو کامل و فقط با حروف فارسی برگردون.\n"
            )

        context_block = ""
        if prev_text or next_text:
            context_block = "بافت گفتگو (فقط برای فهم معنی — اینا رو ترجمه نکن):\n"
            if prev_text:
                context_block += f"- حباب قبلی: {prev_text}\n"
            if next_text:
                context_block += f"- حباب بعدی: {next_text}\n"
            context_block += "\n"

        if self.style in ("colloquial_fa", "scanlation", "manhwa_fa", "fa_colloquial"):
            # Don't list sample filler words here — Gemma sprinkles them into every line
            user_msg = (
                "نقش تو: مترجم دیالوگ مانهوا از انگلیسی به فارسی محاوره‌ای ایرانی.\n"
                f"{strict_line}"
                "فقط همون یک جمله‌ی «متن انگلیسی» رو ترجمه کن؛ معنی دقیق، لحن طبیعی و گفتاری، نه کتابی.\n"
                "هر حرف ندا یا کلمه‌ی پرکننده فقط وقتی بیاد که معادلش تو متن انگلیسی باشه.\n"
                "هیچ کلمه‌ی انگلیسی تو خروجی نذار (جز توکن‌های ⟦ ⟧).\n"
                f"{name_line}"
            )
            tips = self._safe_extra_instructions()
            if tips:
                user_msg += f"دستور اضافه (لحن فقط، نمونه دیالوگ نده):\n{tips}\n"
            user_msg += (
                "ممنوع:\n"
                "- ساختن دیالوگ جدید یا کپی از حافظه/نمونه\n"
                "- آوردن پیشوند EN/FA یا توضیح\n"
                "- خلاصه‌کردن یا حذف معنی\n"
                "- ترجمه کردن توکن‌های داخل ⟦ ⟧\n\n"
                f"{context_block}"
                f"متن انگلیسی:\n{text}\n\n"
                "فقط ترجمه فارسی:"
            )
        else:
            user_msg = (
                f"You are a professional manhwa/comic dialogue translator.\n"
                f"{strict_line}"
                f"Translate the following {src_name} dialogue into natural colloquial {tgt_name}.\n"
                f"{name_line}"
                f"Rules:\n"
                f"- Translate the FULL meaning; do NOT summarize or shorten.\n"
                f"- Return ONLY the translation text, nothing else.\n"
                f"- Keep punctuation and speaker tone (hesitation, shouting).\n"
                f"- If a word looks like OCR garbage, skip that word only.\n\n"
                + (
                    "Context (do NOT translate):\n"
                    + (f"- previous: {prev_text}\n" if prev_text else "")
                    + (f"- next: {next_text}\n" if next_text else "")
                    + "\n"
                    if (prev_text or next_text)
                    else ""
                )
                + f"Line to translate:\n{text}"
            )
        if self._is_gemma3_mm:
            return [{"role": "user", "content": [{"type": "text", "text": user_msg}]}]
        return [{"role": "user", "content": user_msg}]

    def _names_in_text(self, text: str, glossary: Dict[str, str]) -> List[str]:
        found = []
        for term in glossary.keys():
            if term and term in text:
                found.append(term)
        return found

    def _fa_has_name(self, fa: str, en_name: str, fa_name: str) -> bool:
        if not fa:
            return False
        if en_name and en_name in fa:
            return True
        if fa_name and fa_name in fa:
            return True
        # case-insensitive latin check
        if en_name and en_name.lower() in fa.lower():
            return True
        return False

    def _protect_names(
        self, text: str, glossary: Dict[str, str]
    ) -> tuple[str, Dict[str, str], List[str]]:
        """Replace glossary EN names with stable tokens the model must keep."""
        mapping: Dict[str, str] = {}  # placeholder -> FA (preferred) or EN
        keep_tokens: List[str] = []
        if not text or not glossary:
            return text, mapping, keep_tokens
        # longest keys first
        items = sorted(
            ((k, v) for k, v in glossary.items() if k and k in text),
            key=lambda kv: -len(kv[0]),
        )
        # de-dupe overlapping (same span): keep first longest only by sequential replace
        out = text
        i = 0
        for en, fa in items:
            if en not in out:
                continue
            ph = f"⟦{i}⟧"
            out = out.replace(en, ph)
            mapping[ph] = fa or en
            keep_tokens.append(ph)
            i += 1
        return out, mapping, keep_tokens

    def _restore_names(self, text: str, mapping: Dict[str, str]) -> str:
        out = text or ""
        for ph, fa in mapping.items():
            out = out.replace(ph, fa)
            # model sometimes strips brackets
            bare = ph.strip("⟦⟧")
            out = out.replace(f"[{bare}]", fa).replace(f"({bare})", fa)
        return out

    # Stock FA from old few-shots / model regurgitation (must never appear unless EN matches)
    _LEAKED_FA = (
        "شیطونای کوچولو",
        "چطوری می‌تونستم بخوابم اونم وقتی داشتی این همه درد می‌کشیدی",
        "خیلی خوب از پسش براومدی، جوول",
        "وای خدای من، شما شیطونای",
        "واقعاً دردسرساز شدی",
        "واقعاً دردسرساز هستی",
        "خیلی متاسفم، من نمی‌خواستم اشکات رو بریزم",
        "تو بهترینشی، میدونی",
        "وقتی عصبانی میشی خیلی قیافه خوبی داری",
    )
    _LEAKED_EN = (
        "you're a real pain in the neck",
        "i'm so sorry, i didn't mean to make you cry",
        "you're the best, you know",
        "you're so cute when you're angry",
        "you did great, jewel",
        "how could i sleep when you were in so much pain",
        "oh my god, you little rascals",
    )
    _FILLER_FA = ("چیه؟", "چیه", "وای خدای من", "وای خدای من!")
    _HARD_QA = frozenset(
        {"empty_fa", "no_persian", "identical", "hallucinated_example", "latin_leftover"}
    )

    def _strip_added_fillers(self, fa: str, en: str) -> str:
        """Drop وای/خب/مگه نه that the model added without an English source for them."""
        import re

        t = (fa or "").strip()
        if not t:
            return t
        en_s = en or ""
        has_interj = re.search(
            r"\b(wow|oh|ah+|uh|um|huh|hmm+|whoa|ugh|eh|hey|well|ooh|oops|geez|gosh|yikes|wah)\b",
            en_s,
            re.I,
        )
        has_tag_q = re.search(
            r"(right\s*\?|isn'?t (it|he|she|that)|don'?t you|didn'?t (she|he|you|it|they|we)|"
            r"aren'?t (you|they|we)|won'?t you|wasn'?t it|doesn'?t it|huh\s*\?)",
            en_s,
            re.I,
        )
        out = t
        if not has_interj:
            out = re.sub(r"(^|(?<=[.!?؟]\s))(?:وای|خب|آخ|اوه)،\s*", r"\1", out)
        if not has_tag_q:
            end = "؟" if "?" in en_s else "."
            out = re.sub(r"،\s*(?:مگه نه|نه)\s*[؟?]", end, out)
        out = re.sub(r"\s{2,}", " ", out).strip()
        return out if re.search(r"[\u0600-\u06FF]", out) else t

    def _inline_names(self, text: str, glossary: Dict[str, str]) -> str:
        """For dedicated MT models: swap EN names for their FA form before translating."""
        out = text or ""
        for en, fa in sorted(glossary.items(), key=lambda kv: -len(kv[0] or "")):
            if en and fa and en in out:
                out = out.replace(en, fa)
        return out

    def _safe_extra_instructions(self) -> str:
        """Strip EN:/FA: few-shot pairs from project tips so they can't poison the prompt."""
        import re

        raw = (self.extra_instructions or "").strip()
        if not raw:
            return ""
        # Drop explicit parallel examples
        cleaned = re.sub(r"(?im)^\s*EN\s*:.*$", "", raw)
        cleaned = re.sub(r"(?im)^\s*FA\s*:.*$", "", cleaned)
        cleaned = re.sub(r"(?i)\bEN\s*:.*?(?=FA\s*:|$)", "", cleaned)
        cleaned = re.sub(r"(?i)\bFA\s*:", "", cleaned)
        for leak in self._LEAKED_FA:
            cleaned = cleaned.replace(leak, "")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _en_related_to_leak(self, en: str, leak: str) -> bool:
        en_l = (en or "").lower()
        leak_keys = {
            "شیطونای": ("rascal", "little devil", "naughty"),
            "خوابم": ("sleep", "pain"),
            "جوول": ("jewel", "did great"),
            "دردسرساز": ("pain in the neck", "troublesome", "nuisance"),
            "اشکات": ("cry", "sorry", "tear"),
            "بهترینشی": ("you're the best", "the best"),
            "عصبانی": ("angry", "cute when"),
        }
        for needle, keys in leak_keys.items():
            if needle in leak and any(k in en_l for k in keys):
                return True
        return False

    def _sanitize_fa_output(self, fa: str, en: str) -> str:
        import re

        t = (fa or "").strip()
        if not t:
            return ""
        # Drop accidental "EN: ... FA: ..." junk the model invents
        if re.search(r"(?i)^EN\s*:", t):
            m = re.search(r"(?i)FA\s*:\s*(.+)$", t, re.S)
            if m:
                t = m.group(1).strip()
            else:
                t = ""
        t = re.sub(r"(?i)\bEN\s*:.*?(?=FA\s*:|$)", "", t, flags=re.S).strip()
        t = re.sub(r"(?i)^\s*FA\s*:\s*", "", t).strip()
        t = re.sub(r"(?i)\bFA\s*:\s*", "", t).strip()
        # Strip leaked English example lines embedded in FA
        for leak_en in self._LEAKED_EN:
            if leak_en in t.lower():
                t = re.sub(re.escape(leak_en), "", t, flags=re.I).strip(" ،,.!?")
        # Latin leftover that isn't a placeholder / known name → drop those clauses
        if re.search(r"[A-Za-z]{4,}", t) and "⟦" not in t:
            # keep short latin (OCR names) only if also in source
            en_l = (en or "").lower()
            def _keep_latin(m: re.Match) -> str:
                w = m.group(0)
                return w if w.lower() in en_l or w.startswith("⟦") else ""
            t = re.sub(r"[A-Za-z][A-Za-z'’\-]{3,}", _keep_latin, t)
        removed_leak = False
        for leak in self._LEAKED_FA:
            if leak in t and not self._en_related_to_leak(en, leak):
                removed_leak = True
                # Drop the whole clause/sentence that contains the leak
                parts = re.split(r"(?<=[!.?؟\n])\s*", t)
                kept = []
                for p in parts:
                    if leak in p:
                        continue
                    # Drop filler-only clauses glued after a stock line
                    compact = p.strip(" !?؟.,،~")
                    if compact in self._FILLER_FA:
                        continue
                    kept.append(p)
                t = " ".join(kept).strip(" ،,.!?؟")
                t = t.replace(leak, "").strip(" ،,.!?؟")
        t = re.sub(r"\s{2,}", " ", t).strip(" ،")
        # Remnant of "وای خدای من، شما شیطونای کوچولو" after partial strip
        if re.search(r"وای خدای من،?\s*شما\s*$", t) and "rascal" not in (en or "").lower():
            t = ""
        # Trailing lone filler after a real clause (keep standalone "چیه؟" for short EN)
        t = re.sub(r"(?<=\S)\s+(?:چیه|چیه؟|وای خدای من!?)\s*$", "", t).strip(" ،")
        en_words = len((en or "").split())
        fa_compact = t.strip(" !?؟.,،~")
        # After killing a stock phrase, leftover filler is still garbage
        if removed_leak and (not fa_compact or fa_compact in self._FILLER_FA):
            return ""
        # Pure filler for a multi-word English line → force retry
        if en_words >= 3 and fa_compact in self._FILLER_FA:
            return ""
        if en_words >= 6 and len(fa_compact) < 4:
            return ""
        return t

    def _translation_ok(
        self,
        en: str,
        fa: str,
        glossary: Dict[str, str],
        tgt: str,
        *,
        placeholders: Optional[Dict[str, str]] = None,
    ) -> tuple[bool, str]:
        import re

        if not en.strip():
            return True, ""
        if not fa.strip():
            return False, "empty_fa"
        if tgt[:2] in ("fa", "pe"):
            if not re.search(r"[\u0600-\u06FF]", fa):
                return False, "no_persian"
        if fa.strip() == en.strip():
            return False, "identical"
        if re.search(r"(?i)\bEN\s*:", fa) or re.search(r"(?i)\bFA\s*:", fa):
            return False, "hallucinated_example"
        fa_l = fa.lower()
        for leak_en in self._LEAKED_EN:
            if leak_en in fa_l and leak_en not in (en or "").lower():
                return False, "hallucinated_example"
        for leak in self._LEAKED_FA:
            if leak in fa and not self._en_related_to_leak(en, leak):
                return False, "hallucinated_example"
        en_words = max(1, len(en.split()))
        fa_words = max(1, len(fa.split()))
        fa_compact = fa.strip(" !?؟.,،~")
        if en_words >= 3 and fa_compact in self._FILLER_FA:
            return False, "hallucinated_example"
        latin = [
            w
            for w in re.findall(r"[A-Za-z][A-Za-z'’\-]{2,}", re.sub(r"⟦\d+⟧", "", fa))
            if w.lower() not in ("ok",)
        ]
        if latin:
            return False, "latin_leftover"
        # Colloquial FA is often shorter — extreme cuts are soft only
        if en_words >= 12 and fa_words < max(2, int(en_words * 0.2)):
            return True, "soft_too_short"
        if placeholders:
            missing_ph = [ph for ph in placeholders if ph not in fa]
            if missing_ph:
                return True, f"soft_missing_ph:{','.join(missing_ph)}"
        return True, ""

    def _apply_glossary(self, text: str, glossary: Dict[str, str]) -> str:
        # Longer keys first so "Your Highness" wins over partials
        for term, fa in sorted(glossary.items(), key=lambda kv: -len(kv[0] or "")):
            if term and fa and term in text:
                text = text.replace(term, fa)
        return text

    def _translate_causal_one(
        self,
        text: str,
        src: str,
        tgt: str,
        *,
        keep_names: Optional[List[str]] = None,
        strict: bool = False,
        prev_text: str = "",
        next_text: str = "",
        sample: bool = False,
    ) -> str:
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

        approx_out = min(512, max(self.max_new_tokens, int(len(text.split()) * 2.5) + 32))
        gen_extra: Dict[str, Any] = {}

        if _is_hunyuan_mt(self.model_name or ""):
            # Official template; extra instructions degrade this translator
            label = self.mt_target_label if tgt[:2] in ("fa", "pe") else tgt_name
            prompt_txt = (
                f"Translate the following segment into {label}, "
                f"without additional explanation.\n\n{text}"
            )
            messages = [{"role": "user", "content": prompt_txt}]
            try:
                # Hunyuan's template already opens the assistant turn
                prompt = tok.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False
                )
            except Exception:
                prompt = prompt_txt
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=768)
            gen_extra["repetition_penalty"] = 1.05
        elif "english-persian" in model_l or "en-fa" in model_l or "llama-en-fa" in model_l or "llama-3.2-1b" in model_l:
            prompt = f"### English:\n{text}\n### Persian:\n"
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=768)
        else:
            messages = self._build_causal_messages(
                text,
                src_name,
                tgt_name,
                keep_names=keep_names,
                strict=strict,
                prev_text=prev_text,
                next_text=next_text,
            )
            try:
                prompt = tok.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=768)
            except Exception:
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
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=768)

        try:
            dev = next(self._model.parameters()).device
            inputs = {k: v.to(dev) for k, v in inputs.items()}
        except Exception:
            pass

        pad_id = tok.eos_token_id or getattr(tok, "pad_token_id", None)
        if sample:
            # Hunyuan-MT's recommended sampling settings
            gen_extra.update(
                {"do_sample": True, "temperature": 0.7, "top_p": 0.6, "top_k": 20}
            )
        else:
            gen_extra["do_sample"] = False
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=approx_out,
                pad_token_id=pad_id,
                **gen_extra,
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        text_out = tok.decode(gen, skip_special_tokens=True).strip()
        if "```" in text_out:
            text_out = text_out.split("```")[0].strip()
        return text_out

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
        translated = [""] * len(texts)
        non_empty_idx = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_idx]
        results_text: List[str] = []

        if non_empty_texts:
            if self._mode == "causal":
                is_mt = _is_hunyuan_mt(self.model_name or "")
                ctx_texts = [self._inline_names(x, glossary) for x in non_empty_texts]
                for j, t in enumerate(non_empty_texts):
                    if is_mt:
                        protected, ph_map, keep_ph = self._inline_names(t, glossary), {}, []
                    else:
                        protected, ph_map, keep_ph = self._protect_names(t, glossary)
                    w = self.context_window
                    prev_text = " / ".join(ctx_texts[max(0, j - w) : j]) if w else ""
                    next_text = " / ".join(ctx_texts[j + 1 : j + 1 + w]) if w else ""

                    def attempt(strict: bool, sample: bool) -> str:
                        raw = self._translate_causal_one(
                            protected,
                            src,
                            tgt,
                            keep_names=keep_ph or None,
                            strict=strict,
                            prev_text=prev_text,
                            next_text=next_text,
                            sample=sample,
                        )
                        cleaned = self._sanitize_fa_output(raw, protected)
                        return self._strip_added_fillers(cleaned, t)

                    fa = attempt(strict=False, sample=False)
                    best = fa
                    if self.validate:
                        ok, reason = self._translation_ok(
                            protected, fa, glossary, tgt, placeholders=ph_map
                        )
                        retries = 0
                        while not ok and reason in self._HARD_QA and retries < self.max_retries:
                            logger.warning(
                                f"Translation QA failed ({reason}); retrying"
                            )
                            # Greedy MT would repeat itself; sample instead of a stricter prompt
                            fa = attempt(strict=not is_mt, sample=is_mt)
                            ok, reason = self._translation_ok(
                                protected, fa, glossary, tgt, placeholders=ph_map
                            )
                            if fa and (ok or not best):
                                best = fa
                            retries += 1
                        if ok:
                            best = fa
                        if reason.startswith("soft_"):
                            logger.debug(f"Translation soft QA: {reason}")
                        elif not ok and reason in self._HARD_QA:
                            logger.warning(
                                f"Translation QA still failing ({reason}) — keeping best attempt"
                            )
                    fa = self._restore_names(best, ph_map)
                    results_text.append(self._apply_glossary(fa, glossary))
            else:
                protected_list: List[str] = []
                placeholders: List[Dict[str, str]] = []
                for text in non_empty_texts:
                    mapping = {}
                    t = text
                    for i, (term, _) in enumerate(
                        sorted(glossary.items(), key=lambda kv: -len(kv[0] or ""))
                    ):
                        if term and term in t:
                            ph = f"⟦{i}⟧"
                            t = t.replace(term, ph)
                            mapping[ph] = glossary.get(term) or term
                    placeholders.append(mapping)
                    protected_list.append(t)

                for start in range(0, len(protected_list), self.batch_size):
                    batch = protected_list[start : start + self.batch_size]
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

                for j, text in enumerate(results_text):
                    text = self._restore_names(text, placeholders[j])
                    results_text[j] = self._apply_glossary(text, glossary)

            for j, idx in enumerate(non_empty_idx):
                translated[idx] = (
                    results_text[j] if j < len(results_text) else texts[idx]
                )

        return [
            {"region_id": regions[i]["region_id"], "text": (translated[i] or "").strip()}
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
