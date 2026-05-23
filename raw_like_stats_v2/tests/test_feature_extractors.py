from __future__ import annotations

import numpy as np
import torch

from src.features.extractor import extract_all_features
from src.features.srm_kernels import get_kernel_set
from src.models.mlp import TabularMLP
from src.training.tabular_data import FeaturePreprocessor


def test_feature_extractors_return_finite_values():
    rng = np.random.RandomState(42)
    raw = rng.rand(4, 64, 64).astype(np.float32)
    cfg = {
        "groups": ["noise", "srm", "glcm", "frequency"],
        "noise": {"patch_size": 8, "num_bins": 4, "include_channel_probes": True},
        "srm": {"kernel_set": "normalized_hpf_3x3", "stats": ["absmean", "var"]},
        "glcm": {"levels": 8, "distances": [1], "angles_degrees": [0, 90], "cross_channel": True},
        "frequency": {"num_radial_bins": 4},
    }
    feats = extract_all_features(raw, cfg)
    assert feats
    assert all(np.isfinite(v) for v in feats.values())
    assert any(k.startswith("noise_") for k in feats)
    assert any(k.startswith("srm_") for k in feats)
    assert any(k.startswith("glcm_") for k in feats)
    assert any(k.startswith("freq_") for k in feats)


def test_srm_kernel_bank_loads():
    kernels = get_kernel_set("normalized_hpf_3x3")
    assert len(kernels) > 0
    assert all(k.ndim == 2 for k in kernels)


def test_mlp_forward_pass():
    model = TabularMLP(input_dim=12, hidden_dims=[8, 4], dropout=0.0, batch_norm=False)
    y = model(torch.randn(3, 12))
    assert tuple(y.shape) == (3,)


def test_preprocessor_fit_transform_train_only_shape():
    import pandas as pd

    df = pd.DataFrame(
        {
            "noise_a": [1.0, 2.0, np.nan, 4.0],
            "srm_b": [0.0, 1.0, 2.0, 3.0],
            "label": [0, 1, 0, 1],
            "split": ["train", "train", "val", "test"],
        }
    )
    prep = FeaturePreprocessor.fit(df[df["split"] == "train"], ["noise_a", "srm_b"])
    x = prep.transform(df)
    assert x.shape == (4, 2)
    assert np.isfinite(x).all()

