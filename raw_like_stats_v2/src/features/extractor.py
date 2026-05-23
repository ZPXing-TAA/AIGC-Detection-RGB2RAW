from __future__ import annotations

from typing import Dict

import numpy as np

from .frequency_features import extract_frequency_features
from .glcm_features import extract_glcm_features
from .noise_features import extract_noise_features
from .srm_features import extract_srm_features
from .utils import sanitize_features


def extract_all_features(raw_like: np.ndarray, config: dict) -> Dict[str, float]:
    groups = list(config.get("groups", ["noise", "srm", "glcm", "frequency"]))
    out: Dict[str, float] = {}
    if "noise" in groups:
        out.update(extract_noise_features(raw_like, config.get("noise", {})))
    if "srm" in groups:
        out.update(extract_srm_features(raw_like, config.get("srm", {})))
    if "glcm" in groups:
        out.update(extract_glcm_features(raw_like, config.get("glcm", {})))
    if "frequency" in groups:
        out.update(extract_frequency_features(raw_like, config.get("frequency", {})))
    return sanitize_features(out)

