#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import ensure_dir, load_config, set_seed, write_json
from src.features.utils import feature_group_columns
from src.io_utils import stable_feature_columns
from src.training.metrics import (
    binary_metrics,
    confusion_by_generator_family,
    per_generator_metrics,
    select_global_threshold,
    summarize_per_generator,
)
from src.training.tabular_data import FeaturePreprocessor


def read_features(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def train_one_lr(
    df: pd.DataFrame,
    feature_columns: List[str],
    output_dir: str,
    seed: int,
    threshold_metric: str,
    max_iter: int,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("features must contain train/val/test splits")

    preprocessor = FeaturePreprocessor.fit(train_df, feature_columns=feature_columns)
    x_train = preprocessor.transform(train_df)
    x_val = preprocessor.transform(val_df)
    x_test = preprocessor.transform(test_df)
    y_train = train_df["label"].astype(int).values
    y_val = val_df["label"].astype(int).values
    y_test = test_df["label"].astype(int).values

    clf = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    clf.fit(x_train, y_train)

    val_score = clf.predict_proba(x_val)[:, 1].astype(float)
    test_score = clf.predict_proba(x_test)[:, 1].astype(float)
    threshold, threshold_info = select_global_threshold(y_val, val_score, metric=threshold_metric)
    val_metrics = binary_metrics(y_val, val_score, threshold)
    test_metrics = binary_metrics(y_test, test_score, threshold)
    per_gen = per_generator_metrics(test_df, test_score, threshold, split="test")
    family_confusion = confusion_by_generator_family(per_gen)
    summary = summarize_per_generator(per_gen)

    metrics: Dict[str, Any] = {
        "model": "logistic_regression",
        "input_dim": int(x_train.shape[1]),
        "feature_columns": feature_columns,
        "val": val_metrics,
        "test": test_metrics,
        "per_generator_summary": summary,
        "protocol": {
            "preprocessor_fit": "train only",
            "classifier_fit": "train only",
            "threshold_selection": "pooled val global threshold",
            "test": "frozen preprocessor/classifier/threshold",
            "metadata_used_as_features": False,
        },
    }

    if save_artifacts:
        preprocessor.save(os.path.join(output_dir, "preprocessor.joblib"))
        joblib.dump(clf, os.path.join(output_dir, "model.joblib"))
        write_json(metrics, os.path.join(output_dir, "metrics.json"))
        write_json(
            {
                "threshold": float(threshold),
                "selection_split": "val",
                "selection_metric": threshold_metric,
                "val_metrics_at_threshold": threshold_info,
            },
            os.path.join(output_dir, "thresholds.json"),
        )
        per_gen.to_csv(os.path.join(output_dir, "per_generator_metrics.csv"), index=False)
        per_gen.to_json(os.path.join(output_dir, "per_generator_metrics.json"), orient="records", indent=2)
        if not family_confusion.empty:
            family_confusion.to_csv(os.path.join(output_dir, "confusion_by_generator_family.csv"), index=False)
            family_confusion.to_json(os.path.join(output_dir, "confusion_by_generator_family.json"), orient="records", indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--group-ablation", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    out_root = config["paths"]["output_dir"]
    features_path = args.features or os.path.join(out_root, "features.parquet")
    df = read_features(features_path)
    all_feature_cols = stable_feature_columns(df)
    threshold_metric = str(config.get("mlp", {}).get("threshold_metric", "balanced_accuracy"))

    lr_out = ensure_dir(os.path.join(out_root, "lr"))
    train_one_lr(
        df=df,
        feature_columns=all_feature_cols,
        output_dir=lr_out,
        seed=seed,
        threshold_metric=threshold_metric,
        max_iter=args.max_iter,
        save_artifacts=True,
    )
    print(f"Wrote LR artifacts: {lr_out}")

    if args.group_ablation:
        analysis_dir = ensure_dir(os.path.join(out_root, "analysis"))
        rows = []
        for group in ["noise", "srm", "glcm", "frequency", "all"]:
            cols = feature_group_columns(all_feature_cols, group)
            if not cols:
                continue
            metrics = train_one_lr(
                df=df,
                feature_columns=cols,
                output_dir=os.path.join(lr_out, "group_ablation", group),
                seed=seed,
                threshold_metric=threshold_metric,
                max_iter=args.max_iter,
                save_artifacts=False,
            )
            rows.append(
                {
                    "group": group,
                    "n_features": len(cols),
                    "test_accuracy": metrics["test"]["accuracy"],
                    "test_balanced_accuracy": metrics["test"]["balanced_accuracy"],
                    "test_AUROC": metrics["test"]["AUROC"],
                    "test_AP": metrics["test"]["AP"],
                    "macro_balanced_accuracy": metrics["per_generator_summary"].get("macro_balanced_accuracy"),
                    "macro_AUROC": metrics["per_generator_summary"].get("macro_AUROC"),
                    "macro_AP": metrics["per_generator_summary"].get("macro_AP"),
                    "worst_generator": metrics["per_generator_summary"].get("worst_generator_by_balanced_accuracy"),
                    "worst_balanced_accuracy": metrics["per_generator_summary"].get("worst_balanced_accuracy"),
                }
            )
        ablation = pd.DataFrame(rows)
        ablation.to_csv(os.path.join(analysis_dir, "group_ablation_lr.csv"), index=False)
        ablation.to_json(os.path.join(analysis_dir, "group_ablation_lr.json"), orient="records", indent=2)


if __name__ == "__main__":
    main()

