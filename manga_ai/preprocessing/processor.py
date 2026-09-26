"""Image validation and light preprocessing (never mutates originals)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from manga_ai.schemas import ImageInfo
from manga_ai.utils.hashing import file_hash


class Preprocessor:
    def __init__(
        self,
        convert_to_rgb: bool = True,
        max_dimension: Optional[int] = None,
        validate: bool = True,
    ):
        self.convert_to_rgb = convert_to_rgb
        self.max_dimension = max_dimension
        self.validate = validate

    def load(self, path: str | Path) -> Tuple[np.ndarray, ImageInfo]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as img:
            # Handle EXIF orientation
            img = ImageOps.exif_transpose(img)

            if self.convert_to_rgb and img.mode != "RGB":
                img = img.convert("RGB")

            width, height = img.size
            mode = img.mode
            channels = len(img.getbands())

            # Optional downscale for processing (original stays untouched on disk)
            process_img = img
            if self.max_dimension and max(width, height) > self.max_dimension:
                scale = self.max_dimension / max(width, height)
                new_w = int(width * scale)
                new_h = int(height * scale)
                process_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                width, height = new_w, new_h

            arr = np.array(process_img)

        info = ImageInfo(
            path=str(path),
            page_id=path.stem,
            width=width,
            height=height,
            channels=channels,
            mode=mode,
            file_hash=file_hash(path),
        )

        if self.validate:
            if arr.size == 0:
                raise ValueError(f"Empty image: {path}")
            if arr.ndim not in (2, 3):
                raise ValueError(f"Unexpected image shape {arr.shape}: {path}")

        return arr, info
