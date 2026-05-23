from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List

import numpy as np
from scipy import ndimage
from scipy.stats import kurtosis, skew


EPS = 1e-8


def intensity(raw: np.ndarray) -> np.ndarray:
    return np.mean(raw.astype(np.float32), axis=0)


def _safe_float(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(value)


def _moments(values: np.ndarray) -> Dict[str, float]:
    vals = values.astype(np.float64).ravel()
    if vals.size == 0:
        return {
            "absmean": 0.0,
            "var": 0.0,
            "skew": 0.0,
            "kurt": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    return {
        "absmean": _safe_float(np.mean(np.abs(vals))),
        "var": _safe_float(np.var(vals)),
        "skew": _safe_float(skew(vals, bias=False)) if vals.size > 2 else 0.0,
        "kurt": _safe_float(kurtosis(vals, bias=False)) if vals.size > 3 else 0.0,
        "p10": _safe_float(np.percentile(vals, 10)),
        "p50": _safe_float(np.percentile(vals, 50)),
        "p90": _safe_float(np.percentile(vals, 90)),
    }


def noise_features(raw: np.ndarray, num_bins: int = 16, local_window: int = 8) -> Dict[str, float]:
    img = intensity(raw)
    h = (img.shape[0] // local_window) * local_window
    w = (img.shape[1] // local_window) * local_window
    img = img[:h, :w]
    patches = img.reshape(h // local_window, local_window, w // local_window, local_window)
    patches = patches.transpose(0, 2, 1, 3).reshape(-1, local_window * local_window)
    means = patches.mean(axis=1)
    variances = patches.var(axis=1)

    out: Dict[str, float] = {}
    if means.size >= 2 and np.var(means) > EPS:
        a, b = np.polyfit(means, variances, 1)
        pred = a * means + b
        ss_res = np.sum((variances - pred) ** 2)
        ss_tot = np.sum((variances - variances.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + EPS)
        residual_var = np.var(variances - pred)
    else:
        a, b, r2, residual_var = 0.0, float(variances.mean()) if variances.size else 0.0, 0.0, 0.0

    out["noise_slope_a"] = _safe_float(a)
    out["noise_intercept_b"] = _safe_float(b)
    out["noise_fit_r2"] = _safe_float(r2)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.clip(np.digitize(means, bins, right=False) - 1, 0, num_bins - 1)
    for idx in range(num_bins):
        mask = bin_ids == idx
        out[f"noise_bin_var_{idx:02d}"] = _safe_float(np.mean(variances[mask])) if np.any(mask) else 0.0
    for idx in range(num_bins):
        out[f"noise_bin_count_{idx:02d}"] = int(np.sum(bin_ids == idx))

    out["noise_global_residual_var"] = _safe_float(residual_var)
    return out


def _filter_response(channel: np.ndarray, filter_name: str) -> np.ndarray:
    if filter_name == "laplacian":
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        return ndimage.convolve(channel, kernel, mode="reflect")
    if filter_name == "sobel_x":
        return ndimage.sobel(channel, axis=1, mode="reflect")
    if filter_name == "sobel_y":
        return ndimage.sobel(channel, axis=0, mode="reflect")
    if filter_name == "haar_h":
        kernel = np.array([[1, 1], [-1, -1]], dtype=np.float32) * 0.5
        return ndimage.convolve(channel, kernel, mode="reflect")
    if filter_name == "haar_v":
        kernel = np.array([[1, -1], [1, -1]], dtype=np.float32) * 0.5
        return ndimage.convolve(channel, kernel, mode="reflect")
    raise ValueError(f"Unknown highpass filter: {filter_name}")


def highpass_features(raw: np.ndarray, filters: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    laplacian_responses: List[np.ndarray] = []
    for filter_name in filters:
        for ch in range(raw.shape[0]):
            response = _filter_response(raw[ch], filter_name)
            if filter_name == "laplacian":
                laplacian_responses.append(response.ravel())
            stats = _moments(response)
            for stat_name, value in stats.items():
                out[f"hp_{filter_name}_ch{ch}_{stat_name}"] = value

    if len(laplacian_responses) == raw.shape[0]:
        stacked = np.vstack(laplacian_responses)
        cov = np.cov(stacked)
        for i, j in combinations(range(raw.shape[0]), 2):
            out[f"hp_residual_cov_ch{i}_ch{j}"] = _safe_float(cov[i, j])
    return out


def channel_features(raw: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    channels = raw.shape[0]
    flat = raw.reshape(channels, -1).astype(np.float64)
    means = flat.mean(axis=1)
    stds = flat.std(axis=1)
    for ch in range(channels):
        out[f"channel_mean_{ch}"] = _safe_float(means[ch])
        out[f"channel_std_{ch}"] = _safe_float(stds[ch])
    for ch in range(channels):
        if ch != 1:
            out[f"channel_ratio_{ch}_1"] = _safe_float(means[ch] / (means[1] + EPS))

    cov = np.cov(flat)
    corr = np.corrcoef(flat)
    for i, j in combinations(range(channels), 2):
        out[f"channel_cov_{i}_{j}"] = _safe_float(cov[i, j])
        out[f"channel_corr_{i}_{j}"] = _safe_float(corr[i, j])

    img = intensity(raw)
    dark_thr = np.percentile(img, 10)
    high_thr = np.percentile(img, 90)
    dark_mask = img <= dark_thr
    high_mask = img >= high_thr
    for ch in range(channels):
        out[f"channel_dark_var_{ch}"] = _safe_float(np.var(raw[ch][dark_mask])) if np.any(dark_mask) else 0.0
        if np.any(high_mask):
            out[f"channel_highlight_clip_ratio_{ch}"] = _safe_float(np.mean(raw[ch][high_mask] >= 0.98))
        else:
            out[f"channel_highlight_clip_ratio_{ch}"] = 0.0
    return out


def frequency_features(raw: np.ndarray, num_radial_bins: int = 16) -> Dict[str, float]:
    img = intensity(raw)
    img = img - np.mean(img)
    spectrum = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(spectrum) ** 2
    h, w = img.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = radius / (radius.max() + EPS)
    bins = np.linspace(0.0, 1.0, num_radial_bins + 1)
    bin_ids = np.clip(np.digitize(radius.ravel(), bins, right=False) - 1, 0, num_radial_bins - 1)
    power_flat = power.ravel()

    energies: List[float] = []
    out: Dict[str, float] = {}
    for idx in range(num_radial_bins):
        mask = bin_ids == idx
        energy = float(np.mean(power_flat[mask])) if np.any(mask) else 0.0
        energies.append(energy)
        out[f"freq_radial_energy_{idx:02d}"] = _safe_float(energy)

    low = np.mean(energies[: max(1, num_radial_bins // 4)])
    mid = np.mean(energies[num_radial_bins // 4 : max(num_radial_bins // 4 + 1, num_radial_bins // 2)])
    high = np.mean(energies[max(1, 3 * num_radial_bins // 4) :])
    out["freq_high_low_ratio"] = _safe_float(high / (low + EPS))
    out["freq_mid_low_ratio"] = _safe_float(mid / (low + EPS))

    band = 0.08
    horizontal = (np.abs(yy - cy) <= band * h) & (np.abs(xx - cx) > band * w)
    vertical = (np.abs(xx - cx) <= band * w) & (np.abs(yy - cy) > band * h)
    out["freq_horizontal_vertical_ratio"] = _safe_float(np.mean(power[horizontal]) / (np.mean(power[vertical]) + EPS))

    centers = np.arange(num_radial_bins, dtype=np.float64) + 0.5
    log_energy = np.log(np.asarray(energies, dtype=np.float64) + EPS)
    if num_radial_bins >= 2 and np.var(log_energy) > EPS:
        slope, _ = np.polyfit(centers, log_energy, 1)
    else:
        slope = 0.0
    out["freq_spectral_slope"] = _safe_float(slope)
    return out


def extract_features(raw: np.ndarray, feature_cfg: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if feature_cfg.get("noise", {}).get("enabled", True):
        out.update(
            noise_features(
                raw,
                num_bins=int(feature_cfg["noise"].get("num_intensity_bins", 16)),
                local_window=int(feature_cfg["noise"].get("local_window", 8)),
            )
        )
    if feature_cfg.get("highpass", {}).get("enabled", True):
        out.update(highpass_features(raw, feature_cfg["highpass"].get("filters", [])))
    if feature_cfg.get("channel", {}).get("enabled", True):
        out.update(channel_features(raw))
    if feature_cfg.get("frequency", {}).get("enabled", True):
        out.update(
            frequency_features(
                raw,
                num_radial_bins=int(feature_cfg["frequency"].get("num_radial_bins", 16)),
            )
        )
    return out
