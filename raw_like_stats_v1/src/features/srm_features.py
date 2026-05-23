from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage

from .srm_kernels import get_kernel_set
from .utils import ensure_chw, moments, sanitize_features


def _quantize_residual(residual: np.ndarray, q: float, truncation: float) -> np.ndarray:
    q = max(float(q), 1e-12)
    z = np.round(residual / q)
    z = np.clip(z, -float(truncation), float(truncation))
    return z.astype(np.float32, copy=False)


def extract_srm_features(raw_like: np.ndarray, config: dict) -> Dict[str, float]:
    arr = ensure_chw(raw_like)
    kernel_set = str(config.get("kernel_set", "normalized_hpf_3x3"))
    kernels = get_kernel_set(kernel_set)
    quantize = bool(config.get("quantize", True))
    q = float(config.get("q", 1.0))
    truncation = float(config.get("truncation", 3))
    scale = float(config.get("residual_scale", 1.0))
    stats = list(config.get("stats", ["absmean", "var", "skew", "kurt", "p10", "p50", "p90"]))
    features: Dict[str, float] = {}
    for k_idx, kernel in enumerate(kernels):
        for c in range(arr.shape[0]):
            residual = ndimage.convolve(arr[c], kernel, mode="reflect") * scale
            if quantize:
                residual = _quantize_residual(residual, q=q, truncation=truncation)
            vals = moments(residual, stats)
            prefix = f"srm_{kernel_set}_k{k_idx:02d}_ch{c}"
            for stat_name, value in vals.items():
                features[f"{prefix}_{stat_name}"] = value
    return sanitize_features(features)

