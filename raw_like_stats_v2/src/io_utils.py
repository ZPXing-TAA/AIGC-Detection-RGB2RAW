from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


METADATA_COLUMNS = [
    "image_id",
    "dataset",
    "split",
    "subset",
    "generator",
    "generator_family",
    "label",
    "source",
    "path",
    "orig_ext",
]


def read_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("manifest must contain label column")
    if "image_id" not in df.columns:
        raise ValueError("manifest must contain image_id column")
    return df


def raw_like_path(row: pd.Series, config: Dict[str, Any]) -> str:
    cache = config["paths"]["raw_like_cache"]
    source = str(row.get("source", ""))
    if source == "photo":
        root = cache["photo_dir"]
    elif source == "gen":
        root = cache["gen_dir"]
    else:
        root = cache["photo_dir"] if int(row["label"]) == 0 else cache["gen_dir"]
    return os.path.join(root, f"{row['image_id']}.npy")


def load_raw_like(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D packed RAW-like array, got shape {arr.shape}")
    if arr.shape[0] != 4 and arr.shape[-1] == 4:
        arr = np.moveaxis(arr, -1, 0)
    if arr.shape[0] != 4:
        raise ValueError(f"expected packed RGGB shape (4,H,W), got {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    return arr


def metadata_from_row(row: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in METADATA_COLUMNS:
        if col in row:
            value = row[col]
            if pd.isna(value):
                value = ""
            if col == "label":
                value = int(value)
            out[col] = value
    return out


def stable_feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        c
        for c in df.columns
        if c.startswith(("noise_", "srm_", "glcm_", "freq_"))
    ]


def sample_manifest(
    df: pd.DataFrame,
    limit_per_split_label: Optional[int],
    seed: int,
) -> pd.DataFrame:
    if not limit_per_split_label:
        return df
    parts = []
    group_cols = ["split", "label"]
    if "subset" in df.columns:
        group_cols = ["split", "subset", "label"]
    for _, part in df.groupby(group_cols, sort=False):
        n = min(int(limit_per_split_label), len(part))
        parts.append(part.sample(n=n, random_state=seed) if len(part) > n else part)
    return pd.concat(parts, ignore_index=True)

