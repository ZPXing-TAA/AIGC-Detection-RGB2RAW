from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_preprocessed_rgb(path: str, image_size: int, force_rgb: bool = True) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if force_rgb:
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size for {path}")

        scale = float(image_size) / float(min(width, height))
        new_w = max(image_size, int(round(width * scale)))
        new_h = max(image_size, int(round(height * scale)))
        if (new_w, new_h) != img.size:
            img = img.resize((new_w, new_h), Image.BICUBIC)

        left = max(0, (new_w - image_size) // 2)
        top = max(0, (new_h - image_size) // 2)
        img = img.crop((left, top, left + image_size, top + image_size))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.shape != (image_size, image_size, 3):
            raise ValueError(f"Preprocess failed for {Path(path)}: got {arr.shape}")
        return np.moveaxis(arr, -1, 0)
