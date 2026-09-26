"""Shared PaddleOCR construction helpers (2.x / 3.x / Windows CPU)."""

from __future__ import annotations

import os
from typing import Any, Optional

from manga_ai.logging import get_logger

logger = get_logger("manga_ai.paddle_ocr")


def _disable_onednn_env() -> None:
    """Work around PaddlePaddle 3.3.x PIR+oneDNN crash on CPU (Windows).

    Error: ConvertPirAttribute2RuntimeAttribute not support
    [pir::ArrayAttribute<pir::DoubleAttribute>]
    """
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    os.environ.setdefault("FLAGS_pir_apply_inplace_pass", "0")


def create_paddle_ocr(
    lang: str = "en",
    use_gpu: bool = True,
    *,
    enable_mkldnn: bool = False,
) -> Any:
    """Construct a PaddleOCR instance compatible with 2.x and 3.x.

    Always prefers ``enable_mkldnn=False`` so CPU inference does not hit the
    known oneDNN / PIR bug on PaddlePaddle 3.3.x.
    """
    _disable_onednn_env()
    from paddleocr import PaddleOCR

    device = "gpu" if use_gpu else "cpu"
    attempts = (
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
            use_gpu=use_gpu,
            use_angle_cls=True,
            show_log=False,
            enable_mkldnn=enable_mkldnn,
        ),
        dict(
            lang=lang,
            use_gpu=use_gpu,
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
            logger.info(
                f"PaddleOCR ready lang={lang} device={device} "
                f"kwargs={ {k: v for k, v in kwargs.items() if k != 'lang'} }"
            )
            return ocr
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            # Some builds reject enable_mkldnn / device — try next shape
            last_err = e
            logger.debug(f"PaddleOCR init failed with {kwargs}: {e}")
            continue

    raise RuntimeError(f"Could not construct PaddleOCR: {last_err}")
