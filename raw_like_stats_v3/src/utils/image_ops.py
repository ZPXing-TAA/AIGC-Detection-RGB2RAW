from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageOps


def _resample_bicubic():
    return getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def _resample_bilinear():
    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def load_rgb_image(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def resize_shorter_side(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w == size and h == size:
        return img
    if w < h:
        new_w = size
        new_h = int(round(h * float(size) / float(w)))
    else:
        new_h = size
        new_w = int(round(w * float(size) / float(h)))
    return img.resize((new_w, new_h), _resample_bicubic())


def center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = max(0, int(round((w - size) / 2.0)))
    top = max(0, int(round((h - size) / 2.0)))
    return img.crop((left, top, left + size, top + size))


def load_resized_center_crop(path: str, size: int) -> Image.Image:
    img = load_rgb_image(path)
    img = resize_shorter_side(img, size)
    return center_crop(img, size)


def pil_to_chw_float(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def load_rgb_chw(path: str, size: int) -> np.ndarray:
    return pil_to_chw_float(load_resized_center_crop(path, size))


def resize_chw(raw: np.ndarray, size: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3:
        raise ValueError("Expected CHW array, got shape {}".format(raw.shape))
    if raw.shape[1] == size and raw.shape[2] == size:
        return raw
    channels = []
    for c in range(raw.shape[0]):
        channel = np.clip(raw[c], 0.0, 1.0)
        img = Image.fromarray((channel * 65535.0).round().astype(np.uint16))
        img = img.resize((size, size), _resample_bilinear())
        channels.append(np.asarray(img, dtype=np.float32) / 65535.0)
    return np.stack(channels, axis=0).astype(np.float32)


def normalize_chw(x: np.ndarray, mean: Iterable[float], std: Iterable[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mean_arr = np.asarray(list(mean), dtype=np.float32).reshape(-1, 1, 1)
    std_arr = np.asarray(list(std), dtype=np.float32).reshape(-1, 1, 1)
    return (arr - mean_arr) / std_arr

