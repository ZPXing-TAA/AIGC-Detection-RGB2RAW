from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .utils import EPS, ensure_chw, finite_float, raw_intensity, sanitize_features


def _patches_2d(image: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = image.shape
    h2 = (h // patch_size) * patch_size
    w2 = (w // patch_size) * patch_size
    if h2 == 0 or w2 == 0:
        return image.reshape(1, h, w)
    cropped = image[:h2, :w2]
    patches = cropped.reshape(h2 // patch_size, patch_size, w2 // patch_size, patch_size)
    patches = patches.transpose(0, 2, 1, 3).reshape(-1, patch_size, patch_size)
    return patches


def _flat_patch_scores(patches: np.ndarray, mode: str) -> np.ndarray:
    if mode == "gradient_absmean":
        gx = np.diff(patches, axis=2)
        gy = np.diff(patches, axis=1)
        return np.mean(np.abs(gx), axis=(1, 2)) + np.mean(np.abs(gy), axis=(1, 2))
    if mode == "laplacian_var":
        center = patches[:, 1:-1, 1:-1]
        lap = (
            patches[:, :-2, 1:-1]
            + patches[:, 2:, 1:-1]
            + patches[:, 1:-1, :-2]
            + patches[:, 1:-1, 2:]
            - 4.0 * center
        )
        return np.var(lap, axis=(1, 2))
    raise ValueError(f"unknown flat patch score: {mode}")


def _select_flat_patches(patches: np.ndarray, config: dict) -> np.ndarray:
    if not bool(config.get("flat_patch_selection", True)) or len(patches) == 0:
        return patches
    scores = _flat_patch_scores(patches, str(config.get("flat_patch_score", "gradient_absmean")))
    percentile = float(config.get("flat_patch_percentile", 30))
    threshold = np.percentile(scores, percentile)
    keep = scores <= threshold
    selected = patches[keep]
    return selected if len(selected) > 0 else patches


def _bin_noise_curve(
    means: np.ndarray,
    variances: np.ndarray,
    num_bins: int,
    bin_statistic: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bin_values = np.zeros(num_bins, dtype=np.float64)
    bin_centers = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.float64)
    if means.size == 0:
        return bin_centers, bin_values, bin_counts

    lo, hi = float(np.min(means)), float(np.max(means))
    if hi <= lo + EPS:
        idx = num_bins // 2
        bin_centers[idx] = lo
        bin_values[idx] = float(np.median(variances) if bin_statistic == "median" else np.mean(variances))
        bin_counts[idx] = float(means.size)
        return bin_centers, bin_values, bin_counts

    edges = np.linspace(lo, hi + EPS, num_bins + 1)
    ids = np.clip(np.digitize(means, edges) - 1, 0, num_bins - 1)
    for k in range(num_bins):
        mask = ids == k
        bin_centers[k] = 0.5 * (edges[k] + edges[k + 1])
        if np.any(mask):
            vals = variances[mask]
            bin_values[k] = float(np.median(vals) if bin_statistic == "median" else np.mean(vals))
            bin_counts[k] = float(np.sum(mask))
    return bin_centers, bin_values, bin_counts


def _fit_noise_line(x: np.ndarray, y: np.ndarray, counts: np.ndarray) -> Tuple[float, float, float, float]:
    mask = counts > 0
    if np.sum(mask) < 2:
        return 0.0, float(np.mean(y[mask])) if np.any(mask) else 0.0, 0.0, 0.0
    xx = x[mask]
    yy = y[mask]
    a, b = np.polyfit(xx, yy, 1)
    pred = a * xx + b
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + EPS)
    residual_var = float(np.var(yy - pred))
    return finite_float(a), finite_float(b), finite_float(r2), finite_float(residual_var)


def _extract_single_noise_probe(image: np.ndarray, prefix: str, config: dict) -> Dict[str, float]:
    patch_size = int(config.get("patch_size", 8))
    num_bins = int(config.get("num_bins", 16))
    bin_statistic = str(config.get("bin_statistic", "median"))
    patches = _select_flat_patches(_patches_2d(image, patch_size), config)
    means = np.mean(patches, axis=(1, 2), dtype=np.float64)
    variances = np.var(patches, axis=(1, 2), dtype=np.float64)
    centers, curve, counts = _bin_noise_curve(means, variances, num_bins, bin_statistic)
    a, b, r2, residual_var = _fit_noise_line(centers, curve, counts)
    out: Dict[str, float] = {
        f"{prefix}_slope_a": a,
        f"{prefix}_intercept_b": b,
        f"{prefix}_fit_r2": r2,
        f"{prefix}_residual_var": residual_var,
    }
    for k, value in enumerate(curve):
        out[f"{prefix}_bin_var_{k:02d}"] = finite_float(value)
    if bool(config.get("include_bin_counts", True)):
        for k, value in enumerate(counts):
            out[f"{prefix}_bin_count_{k:02d}"] = finite_float(value)
    return out


def extract_noise_features(raw_like: np.ndarray, config: dict) -> Dict[str, float]:
    arr = ensure_chw(raw_like)
    features: Dict[str, float] = {}
    features.update(_extract_single_noise_probe(raw_intensity(arr), "noise_intensity", config))
    if bool(config.get("include_channel_probes", True)):
        for c in range(arr.shape[0]):
            features.update(_extract_single_noise_probe(arr[c], f"noise_ch{c}", config))
    return sanitize_features(features)

