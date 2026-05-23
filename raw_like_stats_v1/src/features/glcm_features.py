from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:  # scikit-image older spelling
    try:
        from skimage.feature import greycomatrix as graycomatrix
        from skimage.feature import greycoprops as graycoprops
    except ImportError:
        graycomatrix = None
        graycoprops = None

from .utils import EPS, ensure_chw, finite_float, quantize01, sanitize_features


def _glcm_entropy(glcm: np.ndarray) -> np.ndarray:
    p = glcm.astype(np.float64, copy=False)
    if p.ndim == 4:
        return -np.sum(p * np.log(p + EPS), axis=(0, 1))
    return np.array([-float(np.sum(p * np.log(p + EPS)))])


def _manual_graycomatrix(
    image: np.ndarray,
    distances: Iterable[int],
    angles: Iterable[float],
    levels: int,
    symmetric: bool,
    normed: bool,
) -> np.ndarray:
    img = np.asarray(image, dtype=np.int64)
    h, w = img.shape
    distances = list(distances)
    angles = list(angles)
    out = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.float64)
    for di, d in enumerate(distances):
        for ai, angle in enumerate(angles):
            dy = int(round(-np.sin(angle) * d))
            dx = int(round(np.cos(angle) * d))
            y0 = max(0, -dy)
            y1 = min(h, h - dy)
            x0 = max(0, -dx)
            x1 = min(w, w - dx)
            a = img[y0:y1, x0:x1].ravel()
            b = img[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
            valid = (a >= 0) & (a < levels) & (b >= 0) & (b < levels)
            idx = a[valid] * levels + b[valid]
            mat = np.bincount(idx, minlength=levels * levels).reshape(levels, levels).astype(np.float64)
            if symmetric:
                mat = mat + mat.T
            if normed:
                mat = mat / (np.sum(mat) + EPS)
            out[:, :, di, ai] = mat
    return out


def _manual_graycoprops(glcm: np.ndarray, prop: str) -> np.ndarray:
    levels = glcm.shape[0]
    ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    out = np.zeros(glcm.shape[2:], dtype=np.float64)
    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            p = glcm[:, :, d, a].astype(np.float64)
            p = p / (np.sum(p) + EPS)
            if prop == "contrast":
                out[d, a] = np.sum(((ii - jj) ** 2) * p)
            elif prop == "energy":
                out[d, a] = np.sqrt(np.sum(p ** 2))
            elif prop == "homogeneity":
                out[d, a] = np.sum(p / (1.0 + np.abs(ii - jj)))
            elif prop == "correlation":
                mu_i = float(np.sum(ii * p))
                mu_j = float(np.sum(jj * p))
                sig_i = float(np.sqrt(np.sum(((ii - mu_i) ** 2) * p) + EPS))
                sig_j = float(np.sqrt(np.sum(((jj - mu_j) ** 2) * p) + EPS))
                out[d, a] = np.sum((ii - mu_i) * (jj - mu_j) * p) / (sig_i * sig_j + EPS)
            else:
                raise ValueError(f"manual graycoprops does not support: {prop}")
    return out


def _aggregate(values: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64).ravel()
    return finite_float(np.mean(vals)), finite_float(np.std(vals))


def _manual_cross_props(matrix: np.ndarray, props: Iterable[str]) -> Dict[str, float]:
    p = matrix.astype(np.float64, copy=False)
    p = p / (np.sum(p) + EPS)
    levels = p.shape[0]
    ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    mu_i = float(np.sum(ii * p))
    mu_j = float(np.sum(jj * p))
    sig_i = float(np.sqrt(np.sum(((ii - mu_i) ** 2) * p) + EPS))
    sig_j = float(np.sqrt(np.sum(((jj - mu_j) ** 2) * p) + EPS))
    out: Dict[str, float] = {}
    for prop in props:
        if prop == "contrast":
            out[prop] = float(np.sum(((ii - jj) ** 2) * p))
        elif prop == "energy":
            out[prop] = float(np.sqrt(np.sum(p ** 2)))
        elif prop == "homogeneity":
            out[prop] = float(np.sum(p / (1.0 + np.abs(ii - jj))))
        elif prop == "entropy":
            out[prop] = -float(np.sum(p * np.log(p + EPS)))
        elif prop == "correlation":
            out[prop] = float(np.sum((ii - mu_i) * (jj - mu_j) * p) / (sig_i * sig_j + EPS))
        else:
            raise ValueError(f"unknown GLCM property: {prop}")
    return {k: finite_float(v) for k, v in out.items()}


def _cross_channel_matrix(ch_a: np.ndarray, ch_b: np.ndarray, levels: int) -> np.ndarray:
    flat_a = ch_a.ravel().astype(np.int64)
    flat_b = ch_b.ravel().astype(np.int64)
    idx = flat_a * levels + flat_b
    hist = np.bincount(idx, minlength=levels * levels)
    return hist.reshape(levels, levels)


def extract_glcm_features(raw_like: np.ndarray, config: dict) -> Dict[str, float]:
    arr = ensure_chw(raw_like)
    levels = int(config.get("levels", 32))
    distances = [int(d) for d in config.get("distances", [1, 2])]
    angles = [np.deg2rad(float(a)) for a in config.get("angles_degrees", [0, 45, 90, 135])]
    symmetric = bool(config.get("symmetric", True))
    normed = bool(config.get("normed", True))
    props = list(config.get("channel_props", ["contrast", "energy", "homogeneity", "entropy", "correlation"]))
    features: Dict[str, float] = {}
    q_channels = [quantize01(arr[c], levels=levels) for c in range(arr.shape[0])]

    for c, q in enumerate(q_channels):
        if graycomatrix is None:
            mat = _manual_graycomatrix(q, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)
        else:
            mat = graycomatrix(q, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)
        for prop in props:
            if prop == "entropy":
                values = _glcm_entropy(mat)
            elif graycoprops is None:
                values = _manual_graycoprops(mat, prop)
            else:
                values = graycoprops(mat, prop)
            mean, std = _aggregate(values)
            features[f"glcm_ch{c}_{prop}_mean"] = mean
            features[f"glcm_ch{c}_{prop}_std"] = std

    if bool(config.get("cross_channel", True)):
        pairs = config.get("cross_channel_pairs", [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        for c1, c2 in pairs:
            matrix = _cross_channel_matrix(q_channels[int(c1)], q_channels[int(c2)], levels=levels)
            vals = _manual_cross_props(matrix, props)
            for prop, value in vals.items():
                features[f"glcm_cross_ch{int(c1)}_ch{int(c2)}_{prop}"] = value
    return sanitize_features(features)
