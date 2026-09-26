"""GPU / device helpers for 6GB VRAM safety."""

from __future__ import annotations

from typing import Optional

from manga_ai.logging import get_logger

logger = get_logger("manga_ai.device")


def get_device(preferred: str = "cuda", device_id: int = 0) -> str:
    if preferred.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return f"cuda:{device_id}"
        except ImportError:
            pass
        logger.warning("CUDA requested but unavailable, using CPU")
    return "cpu"


def peak_vram_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return None


def clear_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def log_vram(tag: str = "") -> None:
    mb = peak_vram_mb()
    if mb is not None:
        logger.info(f"VRAM peak{(' ' + tag) if tag else ''}: {mb:.0f} MB")
