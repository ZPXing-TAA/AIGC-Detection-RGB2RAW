from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .io_utils import load_raw_like, raw_like_path


def limit_per_class(rows: Sequence[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for source in ("photo", "gen"):
        out.extend([row for row in rows if row["source"] == source][:limit])
    return out


def validate_raw_outputs(cfg: dict, rows: Sequence[Dict[str, str]]) -> None:
    missing = [str(raw_like_path(cfg, row)) for row in rows if not raw_like_path(cfg, row).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing RAW-like inputs for learned training. "
            "Run scripts/run_cycleisp_rgb2raw.py first. First missing: {}".format(missing[0])
        )


def split_rows_stratified(
    rows: Sequence[Dict[str, str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List[Dict[str, str]]]:
    by_label = {0: [], 1: []}
    for row in rows:
        by_label[int(row["label"])].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for label_rows in by_label.values():
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        n = len(label_rows)
        if n == 0:
            continue
        if n < 3:
            n_train = max(1, n - 1)
            n_val = 0
            n_test = n - n_train
        else:
            n_val = max(1, int(round(n * val_ratio)))
            n_test = max(1, int(round(n * test_ratio)))
            n_train = n - n_val - n_test
            while n_train < 1 and n_val > 0:
                n_val -= 1
                n_train += 1
            while n_train < 1 and n_test > 0:
                n_test -= 1
                n_train += 1
        splits["train"].extend(label_rows[:n_train])
        splits["val"].extend(label_rows[n_train : n_train + n_val])
        splits["test"].extend(label_rows[n_train + n_val : n_train + n_val + n_test])

    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


class RawLikeDataset(Dataset):
    def __init__(self, cfg: dict, rows: Sequence[Dict[str, str]], augment_hflip: bool = False):
        self.cfg = cfg
        self.rows = list(rows)
        self.augment_hflip = augment_hflip

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        raw = load_raw_like(raw_like_path(self.cfg, row))
        if self.augment_hflip and random.random() < 0.5:
            raw = raw[:, :, ::-1].copy()
        x = torch.from_numpy(np.asarray(raw, dtype=np.float32))
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y, row["image_id"]


def rows_to_split_records(splits: Dict[str, Sequence[Dict[str, str]]]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for split_name, rows in splits.items():
        for row in rows:
            payload = dict(row)
            payload["split"] = split_name
            records.append(payload)
    return records
