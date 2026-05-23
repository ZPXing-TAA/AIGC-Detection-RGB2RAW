#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import ensure_dir, load_config


LABEL_MAP = {
    "0_real": (0, "photo"),
    "1_fake": (1, "gen"),
    "1_false": (1, "gen"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full official AIGIBench manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_extensions(cfg: Dict) -> List[str]:
    exts = cfg.get("data", {}).get("image_extensions", [".jpg", ".jpeg", ".png", ".webp"])
    return [str(x).lower() for x in exts]


def raw_like_path(cfg: Dict, rel_path: Path) -> str:
    root = Path(cfg["data"]["raw_like_cache_root"])
    ext = str(cfg["data"].get("raw_cache_ext", ".npy"))
    return str((root / rel_path).with_suffix(ext))


def iter_label_dirs(split_dir: Path) -> Iterable[Path]:
    for path in split_dir.rglob("*"):
        if path.is_dir() and path.name in LABEL_MAP:
            yield path


def generator_family_for(split: str, subset: str, generator: str) -> str:
    # Keep this conservative for v3. It can be refined later without changing labels.
    return str(subset or generator)


def build_rows(cfg: Dict) -> List[Dict[str, object]]:
    dataset_root = Path(cfg["data"]["dataset_root"])
    dataset_name = cfg["data"].get("dataset_name", "AIGIBench")
    exts = set(image_extensions(cfg))
    rows: List[Dict[str, object]] = []
    for split in ["train", "val", "test"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError("Missing split directory: {}".format(split_dir))
        for label_dir in sorted(iter_label_dirs(split_dir)):
            rel_label = label_dir.relative_to(split_dir)
            parts = rel_label.parts
            if len(parts) < 2:
                continue
            subset = parts[0]
            # Per project spec, initially use the folder directly under split for both.
            generator = parts[0]
            label, source = LABEL_MAP[label_dir.name]
            for img_path in sorted(label_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in exts:
                    continue
                rel_path = img_path.relative_to(dataset_root)
                digest = hashlib.sha1(str(rel_path).encode("utf-8")).hexdigest()[:16]
                rows.append(
                    {
                        "image_id": "aigibench_{}_{}".format(split, digest),
                        "dataset": dataset_name,
                        "split": split,
                        "subset": subset,
                        "generator": generator,
                        "generator_family": generator_family_for(split, subset, generator),
                        "label": int(label),
                        "source": source,
                        "path": str(img_path),
                        "orig_ext": img_path.suffix.lower(),
                        "raw_like_path": raw_like_path(cfg, rel_path),
                    }
                )
    return rows


def make_summary(df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total": int(len(df)),
        "by_split": df.groupby("split").size().astype(int).to_dict(),
        "by_split_source": df.groupby(["split", "source"]).size().astype(int).to_dict(),
        "by_split_label": df.groupby(["split", "label"]).size().astype(int).to_dict(),
        "contains_val_sdv1.4_1_false_as_fake": bool(
            (
                (df["split"] == "val")
                & (df["subset"] == "sdv1.4")
                & (df["path"].str.contains("/1_false/"))
                & (df["label"] == 1)
                & (df["source"] == "gen")
            ).any()
        ),
    }
    return summary


def stringify_tuple_keys(obj):
    if isinstance(obj, dict):
        return {str(k): stringify_tuple_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [stringify_tuple_keys(x) for x in obj]
    return obj


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_path = Path(cfg["data"]["manifest_path"])
    if out_path.exists() and not args.overwrite:
        print("Manifest exists, not overwriting: {}".format(out_path))
        return
    rows = build_rows(cfg)
    df = pd.DataFrame(rows)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    summary = stringify_tuple_keys(make_summary(df))
    summary_path = Path(cfg["data"].get("manifest_summary_path", str(out_path.with_suffix(".summary.json"))))
    ensure_dir(summary_path.parent)
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("Wrote manifest: {}".format(out_path))
    print("Wrote summary: {}".format(summary_path))
    print("Counts: {}".format(summary["by_split"]))
    if not summary["contains_val_sdv1.4_1_false_as_fake"]:
        raise RuntimeError("val/sdv1.4/sdv1.4/1_false was not included as fake")


if __name__ == "__main__":
    main()

