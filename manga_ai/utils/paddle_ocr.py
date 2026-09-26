"""Shared PaddleOCR construction helpers (2.x / 3.x / Windows CPU)."""

from __future__ import annotations

import os
from typing import Any, Optional

from manga_ai.logging import get_logger

logger = get_logger("manga_ai.paddle_ocr")


def _disable_onednn_env() -> None:
    """Force-disable oneDNN / PIR before paddle is imported.

    PaddlePaddle 3.3.x + PaddleX still crashes on CPU with:
      ConvertPirAttribute2RuntimeAttribute not support
      [pir::ArrayAttribute<pir::DoubleAttribute>]
    even when enable_mkldnn=False is passed — PaddleX re-enables mkldnn.
    Overwrite (do not setdefault) so later paddlex code cannot rely on defaults.
    """
    for key, val in (
        ("FLAGS_use_mkldnn", "0"),
        ("FLAGS_enable_pir_api", "0"),
        ("FLAGS_enable_pir_in_executor", "0"),
        ("FLAGS_pir_apply_inplace_pass", "0"),
        ("FLAGS_onednn_exhaustive_search", "0"),
    ):
        os.environ[key] = val


def _paddle_gpu_available() -> bool:
    try:
        import paddle

        return bool(paddle.device.is_compiled_with_cuda()) and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def _patch_disable_mkldnn(ocr: Any) -> None:
    """Best-effort: force PaddleX predictors off mkldnn after construction."""
    try:
        pipeline = getattr(ocr, "paddlex_pipeline", None) or getattr(ocr, "_pipeline", None)
        if pipeline is None:
            return
        # Walk nested models / predictors
        for attr in ("paddlex_pipeline", "pipeline", "_pipeline", "model"):
            obj = getattr(pipeline, attr, pipeline)
            models = getattr(obj, "_models", None) or getattr(obj, "models", None) or []
            if isinstance(models, dict):
                models = list(models.values())
            for model in list(models) + [obj]:
                for opt_name in ("_pp_option", "pp_option", "predictor_option"):
                    opt = getattr(model, opt_name, None)
                    if opt is None:
                        continue
                    for setter in ("set_run_mode", "run_mode"):
                        try:
                            if setter == "set_run_mode" and hasattr(opt, setter):
                                opt.set_run_mode("paddle")
                            elif setter == "run_mode":
                                setattr(opt, "run_mode", "paddle")
                        except Exception:
                            pass
                    try:
                        if hasattr(opt, "enable_mkldnn"):
                            opt.enable_mkldnn = False
                    except Exception:
                        pass
    except Exception as e:
        logger.debug(f"mkldnn patch skipped: {e}")


def create_paddle_ocr(
    lang: str = "en",
    use_gpu: bool = True,
    *,
    enable_mkldnn: bool = False,
) -> Any:
    """Construct a PaddleOCR instance compatible with 2.x and 3.x."""
    _disable_onednn_env()

    gpu_ok = bool(use_gpu) and _paddle_gpu_available()
    if use_gpu and not gpu_ok:
        logger.warning(
            "Paddle GPU not available (CPU paddle build or no CUDA). "
            "Using device=cpu with mkldnn disabled. "
            "If inference still crashes, pip install paddlepaddle==3.2.2 "
            "or switch detection.backend to easyocr."
        )
    device = "gpu" if gpu_ok else "cpu"

    from paddleocr import PaddleOCR

    # lean kwargs first — skip doc-orientation / unwarping extras that also hit oneDNN
    attempts = (
        dict(
            lang=lang,
            device=device,
            enable_mkldnn=enable_mkldnn,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        ),
        dict(
            lang=lang,
            device=device,
            enable_mkldnn=enable_mkldnn,
        ),
        dict(
            lang=lang,
            device=device,
        ),
        dict(
            lang=lang,
            use_gpu=gpu_ok,
            use_angle_cls=True,
            show_log=False,
            enable_mkldnn=enable_mkldnn,
        ),
        dict(
            lang=lang,
            use_gpu=gpu_ok,
            use_angle_cls=True,
            show_log=False,
        ),
        dict(lang=lang, use_angle_cls=True, enable_mkldnn=enable_mkldnn),
        dict(lang=lang, use_angle_cls=True),
        dict(lang=lang, enable_mkldnn=enable_mkldnn),
        dict(lang=lang),
    )

    last_err: Optional[BaseException] = None
    for kwargs in attempts:
        try:
            ocr = PaddleOCR(**kwargs)
            _patch_disable_mkldnn(ocr)
            logger.info(
                f"PaddleOCR ready lang={lang} device={device} "
                f"kwargs={ {k: v for k, v in kwargs.items() if k != 'lang'} }"
            )
            return ocr
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            logger.debug(f"PaddleOCR init failed with {kwargs}: {e}")
            continue

    raise RuntimeError(f"Could not construct PaddleOCR: {last_err}")
