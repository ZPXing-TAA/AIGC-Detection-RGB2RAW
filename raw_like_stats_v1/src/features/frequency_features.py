from __future__ import annotations

from typing import Dict

import numpy as np

from .utils import EPS, ensure_chw, finite_float, raw_intensity, sanitize_features


def _radial_energy(power: np.ndarray, num_bins: int) -> np.ndarray:
    h, w = power.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr / (np.max(rr) + EPS)
    edges = np.linspace(0.0, 1.0 + EPS, num_bins + 1)
    out = np.zeros(num_bins, dtype=np.float64)
    for k in range(num_bins):
        mask = (rr >= edges[k]) & (rr < edges[k + 1])
        out[k] = float(np.mean(power[mask])) if np.any(mask) else 0.0
    return out


def _fit_log_slope(energies: np.ndarray) -> float:
    x = np.arange(len(energies), dtype=np.float64)
    y = np.log(np.asarray(energies, dtype=np.float64) + EPS)
    if len(x) < 2:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return finite_float(slope)


def _directional_ratio(power: np.ndarray) -> float:
    h, w = power.shape
    cy = h // 2
    cx = w // 2
    band = max(1, min(h, w) // 32)
    horizontal = np.mean(power[max(0, cy - band): min(h, cy + band + 1), :])
    vertical = np.mean(power[:, max(0, cx - band): min(w, cx + band + 1)])
    return finite_float(horizontal / (vertical + EPS))


def _single_frequency_probe(image: np.ndarray, prefix: str, config: dict) -> Dict[str, float]:
    num_bins = int(config.get("num_radial_bins", 16))
    low_radius = float(config.get("low_radius", 0.15))
    mid_radius = float(config.get("mid_radius", 0.35))
    high_radius = float(config.get("high_radius", 0.65))
    x = np.asarray(image, dtype=np.float32)
    x = x - float(np.mean(x))
    spec = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(spec) ** 2
    if bool(config.get("normalize_power", False)):
        power = power / (float(np.sum(power)) + EPS)
    energies = _radial_energy(power, num_bins)
    centers = (np.arange(num_bins, dtype=np.float64) + 0.5) / float(num_bins)
    low = float(np.mean(energies[centers <= low_radius])) if np.any(centers <= low_radius) else 0.0
    mid_mask = (centers > low_radius) & (centers <= mid_radius)
    high_mask = centers >= high_radius
    mid = float(np.mean(energies[mid_mask])) if np.any(mid_mask) else 0.0
    high = float(np.mean(energies[high_mask])) if np.any(high_mask) else 0.0
    out: Dict[str, float] = {}
    for k, value in enumerate(energies):
        out[f"{prefix}_radial_energy_{k:02d}"] = finite_float(value)
    out[f"{prefix}_low_energy"] = finite_float(low)
    out[f"{prefix}_mid_energy"] = finite_float(mid)
    out[f"{prefix}_high_energy"] = finite_float(high)
    out[f"{prefix}_high_low_ratio"] = finite_float(high / (low + EPS))
    out[f"{prefix}_mid_low_ratio"] = finite_float(mid / (low + EPS))
    out[f"{prefix}_log_radial_slope"] = _fit_log_slope(energies)
    out[f"{prefix}_directional_hv_ratio"] = _directional_ratio(power)
    return out


def extract_frequency_features(raw_like: np.ndarray, config: dict) -> Dict[str, float]:
    arr = ensure_chw(raw_like)
    features = _single_frequency_probe(raw_intensity(arr), "freq_intensity", config)
    if bool(config.get("include_channel_probes", False)):
        for c in range(arr.shape[0]):
            features.update(_single_frequency_probe(arr[c], f"freq_ch{c}", config))
    return sanitize_features(features)

