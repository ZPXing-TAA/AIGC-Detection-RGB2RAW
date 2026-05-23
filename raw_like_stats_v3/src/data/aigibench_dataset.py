from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.raw_cache import load_raw_like
from src.utils.image_ops import load_rgb_chw, normalize_chw, resize_chw


class AIGIBenchRgbRawDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        image_size: int,
        rgb_mean: Iterable[float],
        rgb_std: Iterable[float],
        raw_mean: Iterable[float],
        raw_std: Iterable[float],
        raw_size: int = None,
        use_rgb: bool = True,
        use_raw: bool = True,
        limit: int = None,
    ):
        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split].copy()
        df = df.sort_values("image_id").reset_index(drop=True)
        if limit is not None:
            df = df.iloc[: int(limit)].copy()
        self.df = df
        self.image_size = int(image_size)
        self.raw_size = int(raw_size if raw_size is not None else image_size)
        self.rgb_mean = list(rgb_mean)
        self.rgb_std = list(rgb_std)
        self.raw_mean = list(raw_mean)
        self.raw_std = list(raw_std)
        self.use_rgb = bool(use_rgb)
        self.use_raw = bool(use_raw)

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]
        sample: Dict[str, Any] = {
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "path": str(row["path"]),
            "raw_like_path": str(row["raw_like_path"]),
            "image_id": str(row["image_id"]),
            "dataset": str(row.get("dataset", "AIGIBench")),
            "split": str(row["split"]),
            "subset": str(row["subset"]),
            "generator": str(row["generator"]),
            "generator_family": str(row["generator_family"]),
            "source": str(row["source"]),
        }
        if self.use_rgb:
            rgb = load_rgb_chw(str(row["path"]), self.image_size)
            rgb = normalize_chw(rgb, self.rgb_mean, self.rgb_std)
            sample["rgb"] = torch.from_numpy(rgb.astype(np.float32))
        if self.use_raw:
            raw = load_raw_like(str(row["raw_like_path"]))
            raw = resize_chw(raw, self.raw_size)
            raw = normalize_chw(raw, self.raw_mean, self.raw_std)
            sample["raw_like"] = torch.from_numpy(raw.astype(np.float32))
        return sample

