from __future__ import annotations

import csv
import hashlib
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

from .config_utils import ensure_dir, resolve_path


IMAGE_ID_SAFE_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
DEFAULT_MANIFEST_COLUMNS = [
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
FAILED_IMAGE_COLUMNS = ["stage", "image_id", "split", "subset", "label", "source", "path", "error"]


def list_images(root: str, extensions: Iterable[str]) -> List[str]:
    root_path = Path(root)
    exts = {e.lower() for e in extensions}
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_path):
        for name in filenames:
            if Path(name).suffix.lower() in exts:
                paths.append(str(Path(dirpath) / name))
    return sorted(paths)


def stable_id(source: str, index: int, path: str) -> str:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]
    return f"{source}_{index:06d}_{digest}"


def safe_token(value: str) -> str:
    return "".join(ch if ch in IMAGE_ID_SAFE_CHARS else "_" for ch in value)


def stable_aigibench_id(split: str, subset: str, source: str, index: int, path: str) -> str:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    return "aigibench_{}_{}_{}_{:07d}_{}".format(
        safe_token(split),
        safe_token(subset),
        safe_token(source),
        index,
        digest,
    )


def build_bfree_manifest(
    real_dir: str,
    fake_dir: str,
    extensions: Iterable[str],
    seed: int,
    balance_classes: bool = True,
    max_per_class: Optional[int] = None,
) -> List[Dict[str, str]]:
    real_paths = list_images(real_dir, extensions)
    fake_paths = list_images(fake_dir, extensions)
    rng = random.Random(seed)
    rng.shuffle(real_paths)
    rng.shuffle(fake_paths)

    if balance_classes:
        limit = min(len(real_paths), len(fake_paths))
        if max_per_class is not None:
            limit = min(limit, int(max_per_class))
        real_paths = real_paths[:limit]
        fake_paths = fake_paths[:limit]
    elif max_per_class is not None:
        real_paths = real_paths[: int(max_per_class)]
        fake_paths = fake_paths[: int(max_per_class)]

    rows: List[Dict[str, str]] = []
    for idx, path in enumerate(sorted(real_paths)):
        rows.append(
            {
                "image_id": stable_id("photo", idx, path),
                "dataset": "bfree_online",
                "split": "",
                "subset": os.path.basename(os.path.dirname(path)),
                "generator": "",
                "generator_family": "",
                "label": "0",
                "source": "photo",
                "path": path,
                "orig_ext": Path(path).suffix.lower(),
            }
        )
    for idx, path in enumerate(sorted(fake_paths)):
        rows.append(
            {
                "image_id": stable_id("gen", idx, path),
                "dataset": "bfree_online",
                "split": "",
                "subset": os.path.basename(os.path.dirname(path)),
                "generator": "",
                "generator_family": "",
                "label": "1",
                "source": "gen",
                "path": path,
                "orig_ext": Path(path).suffix.lower(),
            }
        )
    return rows


def _row_sort_key(row: Dict[str, str]):
    return (
        row.get("split", ""),
        row.get("subset", ""),
        row.get("label", ""),
        row.get("path", ""),
    )


def _label_from_dir(name: str) -> Optional[str]:
    if name == "0_real":
        return "0"
    if name == "1_fake":
        return "1"
    return None


def _source_from_label(label: str) -> str:
    return "photo" if str(label) == "0" else "gen"


def _family_for_subset(subset: str, family_map: Dict[str, str]) -> str:
    if subset in family_map:
        return family_map[subset]
    normalized = subset.replace("_", "-")
    if normalized in family_map:
        return family_map[normalized]
    return "unknown"


def _scan_aigibench_split(
    split_root: Path,
    split: str,
    extensions: Iterable[str],
    allowed_subsets: Optional[Sequence[str]],
    family_map: Dict[str, str],
) -> List[Dict[str, str]]:
    exts = {e.lower() for e in extensions}
    allowed = set(allowed_subsets) if allowed_subsets else None
    rows: List[Dict[str, str]] = []
    if not split_root.exists():
        return rows

    for dirpath, _, filenames in os.walk(split_root):
        label = _label_from_dir(Path(dirpath).name)
        if label is None:
            continue
        rel_parts = Path(dirpath).relative_to(split_root).parts
        if not rel_parts:
            continue
        subset = rel_parts[0]
        if allowed is not None and subset not in allowed:
            continue
        image_paths = []
        for name in filenames:
            if Path(name).suffix.lower() in exts:
                image_paths.append(str(Path(dirpath) / name))
        for path in sorted(image_paths):
            rows.append(
                {
                    "dataset": "aigibench",
                    "split": split,
                    "subset": subset,
                    "generator": subset,
                    "generator_family": _family_for_subset(subset, family_map),
                    "label": label,
                    "source": _source_from_label(label),
                    "path": path,
                    "orig_ext": Path(path).suffix.lower(),
                }
            )
    return rows


def _limit_per_label_per_subset(
    rows: Sequence[Dict[str, str]],
    limit: Optional[int],
    seed: int,
    split_limits: Optional[Dict[str, Optional[int]]] = None,
) -> List[Dict[str, str]]:
    split_limits = split_limits or {}
    if limit is None and not any(value is not None for value in split_limits.values()):
        return list(rows)
    buckets: Dict[tuple, List[Dict[str, str]]] = {}
    for row in rows:
        key = (row.get("split", ""), row.get("subset", ""), row.get("label", ""))
        buckets.setdefault(key, []).append(row)
    rng = random.Random(seed)
    limited: List[Dict[str, str]] = []
    for key in sorted(buckets):
        split = key[0]
        bucket_limit = split_limits.get(split, limit)
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        if bucket_limit is None:
            selected = bucket
        else:
            selected = bucket[: int(bucket_limit)]
        limited.extend(sorted(selected, key=_row_sort_key))
    return sorted(limited, key=_row_sort_key)


def build_aigibench_manifest(
    root: str,
    extensions: Iterable[str],
    seed: int,
    train_subsets: Optional[Sequence[str]] = None,
    val_subsets: Optional[Sequence[str]] = None,
    test_subsets: Optional[Sequence[str]] = None,
    generator_families: Optional[Dict[str, str]] = None,
    max_per_label_per_subset: Optional[int] = None,
    max_per_label_per_subset_by_split: Optional[Dict[str, Optional[int]]] = None,
) -> List[Dict[str, str]]:
    root_path = Path(root)
    family_map = generator_families or {}
    rows: List[Dict[str, str]] = []
    split_specs = [
        ("train", train_subsets),
        ("val", val_subsets),
        ("test", test_subsets),
    ]
    for split, allowed_subsets in split_specs:
        rows.extend(
            _scan_aigibench_split(
                root_path / split,
                split=split,
                extensions=extensions,
                allowed_subsets=allowed_subsets,
                family_map=family_map,
            )
        )
    rows = _limit_per_label_per_subset(
        rows,
        max_per_label_per_subset,
        seed,
        split_limits=max_per_label_per_subset_by_split,
    )
    rows = sorted(rows, key=_row_sort_key)
    for idx, row in enumerate(rows):
        row["image_id"] = stable_aigibench_id(row["split"], row["subset"], row["source"], idx, row["path"])
    return rows


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    ensure_dir(path.parent)
    discovered = []
    for row in rows:
        for key in row.keys():
            if key not in discovered:
                discovered.append(key)
    fieldnames = [key for key in DEFAULT_MANIFEST_COLUMNS if key in discovered]
    fieldnames.extend([key for key in discovered if key not in fieldnames])
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_rows(rows: List[Dict[str, str]], split: str) -> List[Dict[str, str]]:
    if split not in {"photo", "gen", "all", "train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if split == "all":
        return rows
    if split in {"photo", "gen"}:
        return [row for row in rows if row["source"] == split]
    return [row for row in rows if row.get("split") == split]


def manifest_path_from_config(cfg: dict) -> Path:
    return resolve_path(cfg, cfg["data"]["manifest_path"])


def raw_output_dir(cfg: dict, source: str) -> Path:
    key = "photo_dir" if source == "photo" else "gen_dir"
    return resolve_path(cfg, cfg["raw_like_output"][key])


def raw_like_path(cfg: dict, row: Dict[str, str]) -> Path:
    return raw_output_dir(cfg, row["source"]) / f"{row['image_id']}.npy"


def preview_path(cfg: dict, row: Dict[str, str]) -> Path:
    preview_root = resolve_path(cfg, cfg["raw_like_output"]["preview_dir"])
    return preview_root / row["source"] / f"{row['image_id']}_preview.png"


def manifest_summary_path(cfg: dict) -> Path:
    data_cfg = cfg.get("data", {})
    if data_cfg.get("manifest_summary_path"):
        return resolve_path(cfg, data_cfg["manifest_summary_path"])
    return resolve_path(cfg, "outputs") / "manifest_summary.json"


def failed_images_path(cfg: dict) -> Path:
    data_cfg = cfg.get("data", {})
    if data_cfg.get("failed_images_path"):
        return resolve_path(cfg, data_cfg["failed_images_path"])
    return resolve_path(cfg, "outputs") / "failed_images.csv"


def manifest_summary(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    counts: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}
    invalid_subsets: List[Dict[str, object]] = []
    subsets: Dict[tuple, Dict[str, int]] = {}
    for row in rows:
        split = row.get("split", "") or "unspecified"
        subset = row.get("subset", "") or row.get("source", "unspecified")
        label = str(row.get("label", ""))
        counts.setdefault(split, {}).setdefault(subset, {}).setdefault(label, 0)
        counts[split][subset][label] += 1
        subsets.setdefault((split, subset), {"0": 0, "1": 0})
        if label in {"0", "1"}:
            subsets[(split, subset)][label] += 1

    for (split, subset), label_counts in sorted(subsets.items()):
        if label_counts.get("0", 0) == 0 or label_counts.get("1", 0) == 0:
            invalid_subsets.append(
                {
                    "split": split,
                    "subset": subset,
                    "real_count": int(label_counts.get("0", 0)),
                    "fake_count": int(label_counts.get("1", 0)),
                    "reason": "missing_real_or_fake",
                }
            )

    return {
        "num_rows": int(len(rows)),
        "counts_by_split_subset_label": counts,
        "invalid_or_skipped_subsets": invalid_subsets,
    }


def write_failed_images(path: Path, rows: Sequence[Dict[str, str]], stage: Optional[str] = None) -> None:
    ensure_dir(path.parent)
    existing: List[Dict[str, str]] = []
    if path.exists():
        with open(path, "r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    if stage is not None:
        existing = [row for row in existing if row.get("stage") != stage]
    payload = existing + [dict(row) for row in rows]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FAILED_IMAGE_COLUMNS)
        writer.writeheader()
        for row in payload:
            writer.writerow({key: row.get(key, "") for key in FAILED_IMAGE_COLUMNS})


def failure_record(stage: str, row: Dict[str, str], error: str) -> Dict[str, str]:
    return {
        "stage": stage,
        "image_id": row.get("image_id", ""),
        "split": row.get("split", ""),
        "subset": row.get("subset", ""),
        "label": row.get("label", ""),
        "source": row.get("source", ""),
        "path": row.get("path", ""),
        "error": error,
    }


def save_raw_like(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, array.astype(np.float32, copy=False))


def load_raw_like(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D raw-like array, got {arr.shape} at {path}")
    if arr.shape[0] in {3, 4}:
        return arr
    if arr.shape[-1] in {3, 4}:
        return np.moveaxis(arr, -1, 0)
    raise ValueError(f"Cannot infer channel axis for {arr.shape} at {path}")


def save_bayer_preview(path: Path, packed_rggb: np.ndarray) -> None:
    ensure_dir(path.parent)
    arr = np.asarray(packed_rggb, dtype=np.float32)
    if arr.shape[0] != 4:
        preview = np.mean(arr, axis=0)
    else:
        _, h, w = arr.shape
        preview = np.zeros((h * 2, w * 2), dtype=np.float32)
        preview[0::2, 0::2] = arr[0]
        preview[0::2, 1::2] = arr[1]
        preview[1::2, 0::2] = arr[2]
        preview[1::2, 1::2] = arr[3]
    preview = np.clip(preview, 0.0, 1.0)
    img = Image.fromarray((preview * 255.0 + 0.5).astype(np.uint8), mode="L")
    img.save(path)
