"""Hybrid Qwen-VL OCR: recognize text on detector crops via Qwen2/2.5/3-VL.

Pair with any detection backend (paddle, easyocr, yolo_comic). Does not replace
existing OCR backends — register as ``ocr.backend: hybrid_qwen``.

``ocr.model`` may be a Hugging Face id **or a local folder**, e.g.::

    model: "D:/projects/models/Qwen3-VL-8B-Instruct"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np

from manga_ai.logging import get_logger
from manga_ai.ocr.base import OCRBackend

logger = get_logger("manga_ai.ocr.hybrid_qwen")

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_PROMPT = (
    "Read all text in this comic/manga speech bubble or caption. "
    "Output only the transcribed text, nothing else. "
    "If there is no readable text, output an empty string."
)


def _crop_region(
    image: np.ndarray, region: Dict[str, Any], pad_px: int = 4
) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    bbox = region.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    else:
        polygon = region.get("polygon") or []
        if not polygon:
            return None
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(w, x2 + pad_px)
    y2 = min(h, y2 + pad_px)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def _has_accelerate() -> bool:
    try:
        import accelerate  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_model_id(model: Optional[str]) -> str:
    """Resolve HF id or local path; honor QWEN_VL_MODEL / MANGA_AI_QWEN_MODEL env."""
    raw = (
        (model or "").strip()
        or os.getenv("QWEN_VL_MODEL", "").strip()
        or os.getenv("MANGA_AI_QWEN_MODEL", "").strip()
        or DEFAULT_MODEL
    )
    # Expand ~ / env vars; normalize Windows paths
    expanded = os.path.expandvars(os.path.expanduser(raw))
    path = Path(expanded)
    if path.exists() and path.is_dir():
        return str(path.resolve())
    return expanded


def _pick_model_class(model_id: str) -> Type[Any]:
    """Choose the right Transformers class for Qwen2 / 2.5 / 3 VL."""
    low = model_id.replace("\\", "/").lower()
    errors = []

    if "qwen3" in low or "qwen3-vl" in low or "qwen3_vl" in low:
        try:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        except ImportError as e:
            errors.append(f"Qwen3VLForConditionalGeneration: {e}")
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        except ImportError as e:
            errors.append(f"AutoModelForImageTextToText: {e}")

    if "qwen2.5" in low or "qwen2_5" in low:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        except ImportError as e:
            errors.append(str(e))

    try:
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration
    except ImportError as e:
        errors.append(str(e))

    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError as e:
        errors.append(str(e))

    raise ImportError(
        "Need a recent transformers with Qwen2/Qwen2.5/Qwen3-VL support. "
        "For Qwen3-VL: pip install -U 'transformers>=4.57.0' "
        f"(details: {'; '.join(errors)})"
    )


class HybridQwenOCRBackend(OCRBackend):
    """Crop each detection region and OCR it with a local Qwen-VL model."""

    name = "hybrid_qwen"

    def __init__(
        self,
        model: Optional[str] = None,
        lang: str = "en",
        default_lang: Optional[str] = None,
        use_gpu: bool = True,
        device: Optional[str] = None,
        prompt: Optional[str] = None,
        max_new_tokens: int = 128,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        pad_px: int = 4,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        local_files_only: Optional[bool] = None,
        **kwargs: Any,
    ):
        self.model_name = _resolve_model_id(model)
        self.lang = default_lang or lang or "en"
        self.use_gpu = use_gpu
        self.device = device or ("cuda" if use_gpu else "cpu")
        self.prompt = prompt or DEFAULT_PROMPT
        self.max_new_tokens = int(max_new_tokens)
        self.load_in_4bit = bool(load_in_4bit)
        self.load_in_8bit = bool(load_in_8bit)
        self.pad_px = int(pad_px)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        # Auto local-only when path exists on disk (skip Hub)
        if local_files_only is None:
            self.local_files_only = Path(self.model_name).is_dir()
        else:
            self.local_files_only = bool(local_files_only)
        self._model = None
        self._processor = None
        self._torch_device = None

    def _resolve_device(self):
        import torch

        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as e:
            raise ImportError(
                "hybrid_qwen requires transformers + torch. "
                "pip install transformers accelerate torch qwen-vl-utils"
            ) from e

        model_id = self.model_name
        if self.local_files_only and not Path(model_id).is_dir():
            raise FileNotFoundError(
                f"Local Qwen model folder not found: {model_id}\n"
                f"Set ocr.model to your path, e.g. D:/projects/models/Qwen3-VL-8B-Instruct"
            )

        self._torch_device = self._resolve_device()
        logger.info(
            f"Loading Hybrid Qwen OCR model={model_id} "
            f"device={self._torch_device} 4bit={self.load_in_4bit} "
            f"local_files_only={self.local_files_only}"
        )

        model_cls = _pick_model_class(model_id)

        proc_kwargs: Dict[str, Any] = {"local_files_only": self.local_files_only}
        if self.min_pixels is not None:
            proc_kwargs["min_pixels"] = int(self.min_pixels)
        if self.max_pixels is not None:
            proc_kwargs["max_pixels"] = int(self.max_pixels)
        self._processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)

        load_kwargs: Dict[str, Any] = {"local_files_only": self.local_files_only}
        if self.load_in_4bit or self.load_in_8bit:
            if not _has_accelerate():
                raise ImportError(
                    "pip install accelerate bitsandbytes  # required for 4/8-bit"
                )
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit and not self.load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = "auto"
        else:
            dtype = torch.float16 if self._torch_device.type == "cuda" else torch.float32
            load_kwargs["torch_dtype"] = dtype

        self._model = model_cls.from_pretrained(model_id, **load_kwargs)
        if "device_map" not in load_kwargs:
            self._model = self._model.to(self._torch_device)
        self._model.eval()

    def _generate(self, crop: np.ndarray) -> str:
        from PIL import Image

        pil = Image.fromarray(crop)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = [pil]
        try:
            from qwen_vl_utils import process_vision_info

            # Qwen3-VL may want image_patch_size=16
            try:
                image_inputs, _video_inputs = process_vision_info(
                    messages, image_patch_size=16
                )
            except TypeError:
                image_inputs, _video_inputs = process_vision_info(messages)
        except Exception:
            pass

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Move tensors; some processors return BatchFeature
        moved = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                moved[k] = v.to(self._torch_device)
            else:
                moved[k] = v
        inputs = moved

        import torch

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated)
        ]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (decoded[0] if decoded else "").strip()

    def recognize(self, image: np.ndarray, region: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_model()
        crop = _crop_region(image, region, pad_px=self.pad_px)
        if crop is None:
            return {"text": "", "confidence": 0.0, "language": self.lang}

        try:
            text = self._generate(crop)
        except Exception as e:
            logger.warning(f"Hybrid Qwen OCR failed on crop: {e}")
            return {"text": "", "confidence": 0.0, "language": self.lang}

        for prefix in ("Text:", "The text is:", "The text in the image is:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()
        if text.lower() in ("none", "n/a", "no text", "empty", '""', "''"):
            text = ""

        return {
            "text": text,
            "confidence": 0.85 if text else 0.0,
            "language": self.lang,
        }

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._torch_device = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Hybrid Qwen OCR unloaded")
