"""Self-contained anime/manga-finetuned LaMa inpainter (color-preserving).

This bypasses IOPaint entirely: it downloads the same TorchScript model IOPaint
uses for its ``anime-lama`` backend (``anime-manga-big-lama.pt``, big-LaMa
finetuned on ~300k manga/anime pages) straight from GitHub releases and runs it
with ``torch.jit.load``. No ``iopaint``, no ``diffusers``, no
``huggingface_hub`` — so it is immune to the hub/diffusers version conflicts
that break IOPaint when a newer ``transformers`` (e.g. for Qwen3-VL) is
installed.

Why not the IOPaint ``manga`` model? That one is for *black & white* manga — it
works in grayscale and returns a gray image, which would strip the color from a
color webtoon. ``anime-lama`` operates in RGB and preserves color.

VRAM: LaMa is small and this runs per text-region crop (plus a margin), so even
very tall webtoon pages stay cheap on a 6 GB GPU and unmasked pixels are kept
exactly.
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

from manga_ai.inpainting.base import Inpainter
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.inpainting.anime_lama")

ANIME_LAMA_URL = os.environ.get(
    "ANIME_LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt",
)
ANIME_LAMA_MD5 = os.environ.get("ANIME_LAMA_MODEL_MD5", "29f284f36a0a510bcacf39ecf4c4d54f")
PAD_MOD = 8


def _ceil_modulo(x: int, mod: int) -> int:
    return x if x % mod == 0 else (x // mod + 1) * mod


def _norm_img(np_img: np.ndarray) -> np.ndarray:
    if np_img.ndim == 2:
        np_img = np_img[:, :, np.newaxis]
    return np.transpose(np_img, (2, 0, 1)).astype("float32") / 255.0


def _pad_to_modulo(img: np.ndarray, mod: int) -> np.ndarray:
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    h, w = img.shape[:2]
    oh, ow = _ceil_modulo(h, mod), _ceil_modulo(w, mod)
    return np.pad(img, ((0, oh - h), (0, ow - w), (0, 0)), mode="symmetric")


def _boxes_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape[:2]
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        box = np.array([x, y, x + bw, y + bh], dtype=int)
        box[::2] = np.clip(box[::2], 0, w)
        box[1::2] = np.clip(box[1::2], 0, h)
        boxes.append(box)
    return boxes


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_cache() -> Path:
    env = os.environ.get("MANGA_AI_MODEL_DIR")
    if env:
        return Path(env)
    # prefer repo-local models/ if it exists, else user cache
    local = Path.cwd() / "models"
    if local.is_dir():
        return local
    return Path.home() / ".cache" / "manga_ai"


class AnimeLaMaInpainter(Inpainter):
    """Color anime/manga big-LaMa, run per text-region crop for low VRAM."""

    name = "anime-lama"
    model_name = "anime-lama"

    def __init__(
        self,
        device: str = "cuda",
        crop_margin: int = 128,
        model_path: Optional[str] = None,
        solid_fill_std: float = 10.0,
        feather_sigma: float = 2.0,
        **_: Any,
    ) -> None:
        self.device = device or "cuda"
        self.crop_margin = int(crop_margin)
        # Below this color std in the ring just outside the text, treat the region
        # as a solid bubble/flat area and flat-fill it (crisp) instead of LaMa.
        self.solid_fill_std = float(solid_fill_std)
        self.feather_sigma = float(feather_sigma)
        self.model_path = Path(model_path) if model_path else (_default_cache() / "anime-manga-big-lama.pt")
        self._model = None
        self._torch = None

    # ------------------------------------------------------------------
    def _download_once(self, dest: Path) -> None:
        """One download attempt; prefer torch's hardened streaming downloader."""
        try:
            import torch

            torch.hub.download_url_to_file(ANIME_LAMA_URL, str(dest), progress=False)
            return
        except Exception as e:
            # fall back to urllib if torch.hub is unavailable/unhappy
            logger.info("torch.hub download failed (%s); trying urllib", e)
            urllib.request.urlretrieve(ANIME_LAMA_URL, str(dest))

    def _ensure_downloaded(self) -> Path:
        p = self.model_path
        if p.exists() and (not ANIME_LAMA_MD5 or _md5(p) == ANIME_LAMA_MD5):
            return p
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        last: Optional[Exception] = None
        # Retry: the GitHub CDN occasionally drops the TLS record mid-transfer
        # ([SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC]); a fresh connection fixes it.
        for attempt in range(1, 5):
            try:
                tmp.unlink(missing_ok=True)
                logger.info("Downloading anime-lama model (attempt %d) from %s", attempt, ANIME_LAMA_URL)
                self._download_once(tmp)
                if ANIME_LAMA_MD5 and _md5(tmp) != ANIME_LAMA_MD5:
                    raise RuntimeError(f"checksum mismatch (got {_md5(tmp)}, want {ANIME_LAMA_MD5})")
                tmp.replace(p)
                logger.info("Saved anime-lama model to %s", p)
                return p
            except Exception as e:
                last = e
                logger.warning("anime-lama download attempt %d failed: %s", attempt, e)
                time.sleep(2 ** attempt)
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download anime-lama after retries: {last}. "
            f"Download it manually from {ANIME_LAMA_URL} and place it at {p} "
            f"(or set MANGA_AI_MODEL_DIR / ANIME_LAMA_MODEL_URL)."
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch

        self._torch = torch
        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, anime-lama falling back to CPU")
            device = "cpu"
        self.device = device
        path = self._ensure_downloaded()
        logger.info("Loading anime-lama (TorchScript) on %s", device)
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        self._model = model

    # ------------------------------------------------------------------
    def _forward_crop(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run LaMa on one crop; returns RGB uint8 of the crop's original size."""
        torch = self._torch
        h, w = img_rgb.shape[:2]
        img_p = _pad_to_modulo(img_rgb, PAD_MOD)
        mask_p = _pad_to_modulo(mask, PAD_MOD)

        img_t = torch.from_numpy(_norm_img(img_p)).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(_norm_img(mask_p)).unsqueeze(0).to(self.device)
        mask_t = (mask_t > 0) * 1.0

        with torch.no_grad():
            out = self._model(img_t, mask_t)
        res = out[0].permute(1, 2, 0).detach().cpu().numpy()
        res = np.clip(res * 255, 0, 255).astype("uint8")
        return res[:h, :w]

    def _solid_fill(self, crop_img: np.ndarray, crop_mask: np.ndarray) -> Optional[np.ndarray]:
        """If the ring just outside the mask is near-uniform (a solid speech
        bubble / flat background), return a flat fill of that color — crisp and
        ghost-free. Otherwise return None so LaMa handles the textured art."""
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dil = cv2.dilate(crop_mask, k)
        ring = (dil > 0) & (crop_mask == 0)
        pix = crop_img[ring]
        if pix.shape[0] < 24:
            return None
        pix = pix.reshape(-1, crop_img.shape[2])
        std = float(pix.std(axis=0).mean())
        if std <= self.solid_fill_std:
            color = np.median(pix, axis=0)
            out = np.empty_like(crop_img)
            out[:] = color.astype(np.uint8)
            return out
        return None

    def _feather(self, orig: np.ndarray, cleaned: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        """Blend cleaned into orig with a soft alpha from the mask (no hard seams)."""
        a = cv2.GaussianBlur((crop_mask > 0).astype(np.float32), (0, 0), self.feather_sigma)
        a = np.clip(a, 0.0, 1.0)[:, :, None]
        return (orig.astype(np.float32) * (1 - a) + cleaned.astype(np.float32) * a).astype(np.uint8)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.size == 0 or mask.max() == 0:
            return image.copy()

        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = np.ascontiguousarray(mask.astype(np.uint8))
        image = np.ascontiguousarray(image.astype(np.uint8))
        H, W = image.shape[:2]

        result = image.copy()
        for box in _boxes_from_mask(mask):
            x1, y1, x2, y2 = box
            x1 = max(0, x1 - self.crop_margin)
            y1 = max(0, y1 - self.crop_margin)
            x2 = min(W, x2 + self.crop_margin)
            y2 = min(H, y2 + self.crop_margin)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_img = image[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]
            if crop_mask.max() == 0:
                continue
            m = crop_mask > 0

            # Crisp flat fill for solid bubble interiors; LaMa only for real art.
            filled = self._solid_fill(crop_img, crop_mask)
            if filled is not None:
                region = result[y1:y2, x1:x2]
                region[m] = filled[m]
                result[y1:y2, x1:x2] = region
                continue

            self._ensure_model()
            try:
                cleaned = self._forward_crop(crop_img, crop_mask)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._clear_cuda()
                raise
            result[y1:y2, x1:x2] = self._feather(crop_img, cleaned, crop_mask)
        return result

    def _clear_cuda(self) -> None:
        try:
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass

    def unload(self) -> None:
        self._model = None
        self._clear_cuda()
        self._torch = None
