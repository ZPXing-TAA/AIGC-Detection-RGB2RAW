from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch


def save_raw_like(path: str, raw: np.ndarray, dtype: str = "uint16", layout: str = "chw") -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(raw, dtype=np.float32)
    if layout == "hwc":
        arr = np.transpose(arr, (1, 2, 0))
    elif layout != "chw":
        raise ValueError("Unsupported raw cache layout: {}".format(layout))

    arr = np.clip(arr, 0.0, 1.0)
    if dtype == "uint16":
        out = np.rint(arr * 65535.0).astype(np.uint16)
    elif dtype == "float16":
        out = arr.astype(np.float16)
    elif dtype == "float32":
        out = arr.astype(np.float32)
    else:
        raise ValueError("Unsupported raw cache dtype: {}".format(dtype))

    if path_obj.suffix == ".pt":
        torch.save(torch.from_numpy(out), str(path_obj))
    else:
        np.save(str(path_obj), out)


def load_raw_like(path: str) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix == ".pt":
        arr = torch.load(str(path_obj), map_location="cpu")
        if torch.is_tensor(arr):
            arr = arr.numpy()
    else:
        arr = np.load(str(path_obj))
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError("Expected 3D RAW-like array, got shape {}".format(arr.shape))
    if arr.shape[0] == 4:
        chw = arr
    elif arr.shape[-1] == 4:
        chw = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError("Expected 4 RAW-like channels, got shape {}".format(arr.shape))
    if chw.dtype == np.uint16:
        return chw.astype(np.float32) / 65535.0
    return chw.astype(np.float32)


def raw_cache_info(path: str) -> Dict[str, object]:
    arr = np.load(path, mmap_mode="r") if str(path).endswith(".npy") else torch.load(path, map_location="cpu")
    shape = tuple(arr.shape)
    dtype = str(arr.dtype)
    return {"shape": shape, "dtype": dtype}

