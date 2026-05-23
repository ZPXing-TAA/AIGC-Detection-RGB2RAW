from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

EPS = 1e-12


def ensure_chw(raw_like: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw_like, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D array, got {arr.shape}")
    if arr.shape[0] != 4 and arr.shape[-1] == 4:
        arr = np.moveaxis(arr, -1, 0)
    if arr.shape[0] != 4:
        raise ValueError(f"expected packed RGGB with 4 channels, got {arr.shape}")
    return arr


def raw_intensity(raw_like: np.ndarray) -> np.ndarray:
    arr = ensure_chw(raw_like)
    return np.mean(arr, axis=0, dtype=np.float32)


def finite_float(value: float) -> float:
    value = float(value)
    if np.isfinite(value):
        return value
    return 0.0


def sanitize_features(features: Dict[str, float]) -> Dict[str, float]:
    return {k: finite_float(v) for k, v in features.items()}


def quantize01(image: np.ndarray, levels: int) -> np.ndarray:
    clipped = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    q = np.floor(clipped * float(levels - 1) + 0.5).astype(np.uint8)
    return q


def moments(values: np.ndarray, stats: Iterable[str]) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        return {name: 0.0 for name in stats}
    mean = float(np.mean(x))
    var = float(np.var(x))
    std = float(np.sqrt(var + EPS))
    centered = x - mean
    out: Dict[str, float] = {}
    for name in stats:
        if name == "mean":
            out[name] = mean
        elif name == "absmean":
            out[name] = float(np.mean(np.abs(x)))
        elif name == "var":
            out[name] = var
        elif name == "std":
            out[name] = std
        elif name == "skew":
            out[name] = float(np.mean((centered / std) ** 3)) if std > EPS else 0.0
        elif name == "kurt":
            out[name] = float(np.mean((centered / std) ** 4) - 3.0) if std > EPS else 0.0
        elif name.startswith("p") and name[1:].isdigit():
            out[name] = float(np.percentile(x, int(name[1:])))
        else:
            raise ValueError(f"unknown moment stat: {name}")
    return {k: finite_float(v) for k, v in out.items()}


def feature_group_columns(columns: List[str], group: str) -> List[str]:
    prefixes = {
        "noise": ("noise_",),
        "srm": ("srm_",),
        "glcm": ("glcm_",),
        "frequency": ("freq_",),
        "all": ("noise_", "srm_", "glcm_", "freq_"),
    }
    if group not in prefixes:
        raise KeyError(f"unknown feature group: {group}")
    return [c for c in columns if c.startswith(prefixes[group])]

