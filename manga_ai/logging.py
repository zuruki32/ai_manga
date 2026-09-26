"""Structured logging for the pipeline."""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional


_CONFIGURED = False


def setup_logging(level: str = "INFO", structured: bool = True) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    if structured:
        fmt = "[%(levelname)s] %(message)s"
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("manga_ai")
    root.setLevel(numeric)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "manga_ai") -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


def log_stage(logger: logging.Logger, stage: str, message: str, **kwargs: Any) -> None:
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    if extra:
        logger.info(f"[{stage}] {message} | {extra}")
    else:
        logger.info(f"[{stage}] {message}")


def log_error(
    logger: logging.Logger,
    stage: str,
    error: str,
    page: Optional[str] = None,
    region: Optional[str] = None,
) -> None:
    parts = [f"stage={stage}"]
    if page:
        parts.append(f"page={page}")
    if region:
        parts.append(f"region={region}")
    parts.append(f'error="{error}"')
    logger.error(" | ".join(parts))
