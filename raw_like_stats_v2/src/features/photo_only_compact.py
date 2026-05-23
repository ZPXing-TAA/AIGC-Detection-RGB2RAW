from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy import ndimage

from .noise_features import extract_noise_features
from .srm_kernels import get_kernel_set
from .utils import EPS, ensure_chw, finite_float, moments, raw_intensity, sanitize_features


NOISE20_SUFFIXES = ["slope_a", "intercept_b", "fit_r2", "residual_var"]


def extract_noise20(raw_like: np.ndarray, config: dict) -> Dict[str, float]:
    cfg = dict(config)
    cfg["include_channel_probes"] = True
    cfg["include_bin_counts"] = False
    full = extract_noise_features(raw_like, cfg)
    probes = ["intensity", "ch0", "ch1", "ch2", "ch3"]
    out: Dict[str, float] = {}
    for probe in probes:
        old_prefix = "noise_intensity" if probe == "intensity" else f"noise_{probe}"
        new_prefix = f"noise20_{probe}"
        for suffix in NOISE20_SUFFIXES:
            out[f"{new_prefix}_{suffix}"] = full.get(f"{old_prefix}_{suffix}", 0.0)
    return sanitize_features(out)


def _azimuthal_integration(image: np.ndarray, bins: int) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    x = x - float(np.mean(x))
    power = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    h, w = power.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr / (np.max(rr) + EPS)
    ids = np.minimum((rr * bins).astype(np.int64), bins - 1)
    sums = np.bincount(ids.ravel(), weights=power.ravel(), minlength=bins).astype(np.float64)
    counts = np.bincount(ids.ravel(), minlength=bins).astype(np.float64)
    return sums / (counts + EPS)


def extract_ai_spectrum(raw_like: np.ndarray, config: dict) -> Dict[str, float]:
    intensity = raw_intensity(raw_like)
    bins_list = [int(b) for b in config.get("bins", [64, 128, 300])]
    normalize = bool(config.get("normalize_by_first_bin", True))
    out: Dict[str, float] = {}
    for bins in bins_list:
        ai = _azimuthal_integration(intensity, bins)
        if normalize:
            ai = ai / (float(ai[0]) + EPS)
        prefix = f"ai{bins}"
        for k, value in enumerate(ai):
            out[f"{prefix}_bin_{k:03d}"] = finite_float(value)
        out[f"{prefix}_slope_log"] = _fit_log_slope(ai)
        out[f"{prefix}_tail_mean"] = finite_float(np.mean(ai[max(1, int(0.75 * bins)):]))
        out[f"{prefix}_tail_head_ratio"] = finite_float(np.mean(ai[max(1, int(0.75 * bins)):]) / (np.mean(ai[: max(1, int(0.10 * bins))]) + EPS))
    return sanitize_features(out)


def _fit_log_slope(values: np.ndarray) -> float:
    y = np.log(np.asarray(values, dtype=np.float64) + EPS)
    x = np.arange(len(y), dtype=np.float64)
    if len(y) < 2:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return finite_float(slope)


def _srm_mode_residual(residual: np.ndarray, mode: str, truncation: float) -> np.ndarray:
    if mode == "noq":
        return residual.astype(np.float32, copy=False)
    if mode == "q255":
        q = np.round(residual * 255.0)
        q = np.clip(q, -float(truncation), float(truncation))
        return q.astype(np.float32, copy=False)
    raise ValueError(f"unknown SRM compact mode: {mode}")


def extract_srm_compact(raw_like: np.ndarray, config: dict) -> Dict[str, float]:
    arr = ensure_chw(raw_like)
    kernel_set = str(config.get("kernel_set", "normalized_hpf_3x3"))
    kernels = get_kernel_set(kernel_set)
    stats = list(config.get("stats", ["absmean", "var", "p90"]))
    modes = list(config.get("modes", ["noq"]))
    truncation = float(config.get("truncation", 3))
    out: Dict[str, float] = {}
    for mode in modes:
        for k_idx, kernel in enumerate(kernels):
            for c in range(arr.shape[0]):
                residual = ndimage.convolve(arr[c], kernel, mode="reflect")
                values = _srm_mode_residual(residual, mode=mode, truncation=truncation)
                vals = moments(values, stats)
                prefix = f"srm_{mode}_k{k_idx:02d}_ch{c}"
                for stat_name, value in vals.items():
                    out[f"{prefix}_{stat_name}"] = value
    return sanitize_features(out)


def extract_photo_only_compact_features(raw_like: np.ndarray, config: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update(extract_noise20(raw_like, config.get("noise20", {})))
    out.update(extract_ai_spectrum(raw_like, config.get("ai_spectrum", {})))
    out.update(extract_srm_compact(raw_like, config.get("srm_compact", {})))
    return sanitize_features(out)

