from __future__ import annotations

import numpy as np

from src.features.photo_only_compact import extract_photo_only_compact_features


def test_photo_only_compact_feature_schema_is_finite():
    raw = np.random.RandomState(7).rand(4, 64, 64).astype(np.float32)
    cfg = {
        "noise20": {"patch_size": 8, "num_bins": 4},
        "ai_spectrum": {"bins": [8, 16], "normalize_by_first_bin": True},
        "srm_compact": {
            "kernel_set": "normalized_hpf_3x3",
            "stats": ["absmean", "var", "p90"],
            "modes": ["noq", "q255"],
        },
    }
    feats = extract_photo_only_compact_features(raw, cfg)
    assert len([k for k in feats if k.startswith("noise20_")]) == 20
    assert any(k.startswith("ai8_") for k in feats)
    assert any(k.startswith("ai16_") for k in feats)
    assert any(k.startswith("srm_noq_") for k in feats)
    assert any(k.startswith("srm_q255_") for k in feats)
    assert all(np.isfinite(v) for v in feats.values())

