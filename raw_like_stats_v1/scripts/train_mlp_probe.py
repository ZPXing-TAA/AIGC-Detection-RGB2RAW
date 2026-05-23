#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import ensure_dir, load_config, output_path, set_seed, write_json
from src.features.utils import feature_group_columns
from src.io_utils import stable_feature_columns
from src.models.mlp import TabularMLP
from src.training.metrics import (
    binary_metrics,
    confusion_by_generator_family,
    per_generator_metrics,
    safe_auc,
    select_global_threshold,
    summarize_per_generator,
)
from src.training.tabular_data import FeaturePreprocessor, TabularDataset


def read_features(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def get_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    try:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    except AttributeError:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def evaluate_model(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
    criterion: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    model.eval()
    ds = TabularDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    logits_all: List[np.ndarray] = []
    losses: List[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            if criterion is not None:
                losses.append(float(criterion(logits, yb).item()))
            logits_all.append(logits.detach().cpu().numpy())
    logits_np = np.concatenate(logits_all) if logits_all else np.zeros((0,), dtype=np.float32)
    probs = np.empty_like(logits_np, dtype=np.float64)
    pos = logits_np >= 0
    probs[pos] = 1.0 / (1.0 + np.exp(-logits_np[pos]))
    exp_x = np.exp(logits_np[~pos])
    probs[~pos] = exp_x / (1.0 + exp_x)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "probs": probs.astype(np.float64),
        "auroc": safe_auc(y, probs),
    }


def plot_curves(history: List[Dict[str, Any]], path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip training curve plot: {exc}")
        return
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [h["val_auroc"] for h in history], label="val AUROC")
    axes[1].plot(epochs, [h["val_balanced_accuracy"] for h in history], label="val balanced acc")
    axes[1].set_title("Validation")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train_one_mlp(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: str,
    feature_columns: List[str],
    device_name: str,
    epochs_override: Optional[int] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    mlp_cfg = config.get("mlp", {})
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    ensure_dir(output_dir)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("features must contain train/val/test splits")

    preprocessor = FeaturePreprocessor.fit(train_df, feature_columns=feature_columns)
    if save_artifacts:
        preprocessor.save(os.path.join(output_dir, "preprocessor.joblib"))

    x_train = preprocessor.transform(train_df)
    x_val = preprocessor.transform(val_df)
    x_test = preprocessor.transform(test_df)
    y_train = train_df["label"].astype(int).values
    y_val = val_df["label"].astype(int).values
    y_test = test_df["label"].astype(int).values

    device = get_device(device_name)
    model = TabularMLP(
        input_dim=x_train.shape[1],
        hidden_dims=mlp_cfg.get("hidden_dims", [256, 128]),
        dropout=float(mlp_cfg.get("dropout", 0.25)),
        batch_norm=bool(mlp_cfg.get("batch_norm", True)),
    ).to(device)

    pos_weight_cfg = mlp_cfg.get("pos_weight", "auto")
    if pos_weight_cfg == "auto":
        pos = max(1, int(np.sum(y_train == 1)))
        neg = max(1, int(np.sum(y_train == 0)))
        pos_weight_value = float(neg / pos)
    elif pos_weight_cfg is None:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(pos_weight_cfg)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = make_optimizer(
        model,
        lr=float(mlp_cfg.get("lr", 1e-3)),
        weight_decay=float(mlp_cfg.get("weight_decay", 1e-4)),
    )

    batch_size = int(mlp_cfg.get("batch_size", 256))
    epochs = int(epochs_override or mlp_cfg.get("epochs", 80))
    patience = int(mlp_cfg.get("patience", 12))
    threshold_metric = str(mlp_cfg.get("threshold_metric", "balanced_accuracy"))
    early_metric = str(mlp_cfg.get("early_stop_metric", "val_auroc"))
    train_ds = TabularDataset(x_train, y_train)
    drop_last = len(train_ds) > batch_size and (len(train_ds) % batch_size == 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    best_state = copy.deepcopy(model.state_dict())
    best_value = -np.inf
    best_epoch = 0
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    for epoch in tqdm(range(1, epochs + 1), desc=f"mlp_train[{Path(output_dir).name}]"):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_eval = evaluate_model(model, x_val, y_val, device, batch_size, criterion)
        val_threshold, val_threshold_metrics = select_global_threshold(y_val, val_eval["probs"], metric=threshold_metric)
        val_bal = float(val_threshold_metrics.get("balanced_accuracy", np.nan))
        val_auroc = float(val_eval["auroc"])
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(val_eval["loss"]),
            "val_auroc": val_auroc,
            "val_balanced_accuracy": val_bal,
            "val_threshold": float(val_threshold),
        }
        history.append(item)
        metric_value = val_auroc if early_metric == "val_auroc" else val_bal
        if np.isnan(metric_value):
            metric_value = -np.inf
        if metric_value > best_value:
            best_value = metric_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    val_eval = evaluate_model(model, x_val, y_val, device, batch_size, criterion)
    test_eval = evaluate_model(model, x_test, y_test, device, batch_size, criterion)
    threshold, threshold_metrics = select_global_threshold(y_val, val_eval["probs"], metric=threshold_metric)
    test_metrics = binary_metrics(y_test, test_eval["probs"], threshold)
    val_metrics = binary_metrics(y_val, val_eval["probs"], threshold)
    per_gen = per_generator_metrics(test_df, test_eval["probs"], threshold, split="test")
    summary = summarize_per_generator(per_gen)
    family_confusion = confusion_by_generator_family(per_gen)

    metrics = {
        "input_dim": int(x_train.shape[1]),
        "feature_columns": feature_columns,
        "best_epoch": int(best_epoch),
        "best_validation_value": float(best_value),
        "pos_weight": float(pos_weight_value),
        "val": val_metrics,
        "test": test_metrics,
        "per_generator_summary": summary,
        "protocol": {
            "preprocessor_fit": "train only",
            "model_fit": "train only",
            "threshold_selection": "pooled val global threshold",
            "test": "frozen preprocessor/model/threshold",
            "metadata_used_as_features": False,
        },
    }

    if save_artifacts:
        torch.save(
            {
                "state_dict": best_state,
                "input_dim": int(x_train.shape[1]),
                "feature_columns": feature_columns,
                "model_config": {
                    "hidden_dims": mlp_cfg.get("hidden_dims", [256, 128]),
                    "dropout": float(mlp_cfg.get("dropout", 0.25)),
                    "batch_norm": bool(mlp_cfg.get("batch_norm", True)),
                },
            },
            os.path.join(output_dir, "best.pt"),
        )
        write_json(history, os.path.join(output_dir, "history.json"))
        write_json(metrics, os.path.join(output_dir, "metrics.json"))
        write_json(
            {
                "threshold": float(threshold),
                "selection_split": "val",
                "selection_metric": threshold_metric,
                "val_metrics_at_threshold": threshold_metrics,
            },
            os.path.join(output_dir, "thresholds.json"),
        )
        per_gen.to_csv(os.path.join(output_dir, "per_generator_metrics.csv"), index=False)
        per_gen.to_json(os.path.join(output_dir, "per_generator_metrics.json"), orient="records", indent=2)
        if not family_confusion.empty:
            family_confusion.to_csv(os.path.join(output_dir, "confusion_by_generator_family.csv"), index=False)
            family_confusion.to_json(os.path.join(output_dir, "confusion_by_generator_family.json"), orient="records", indent=2)
        plot_curves(history, os.path.join(output_dir, "training_curves.png"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--group-ablation", action="store_true")
    parser.add_argument("--no-group-ablation", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    set_seed(int(config.get("project", {}).get("seed", 42)))
    out_root = config["paths"]["output_dir"]
    features_path = args.features or os.path.join(out_root, "features.parquet")
    df = read_features(features_path)
    all_feature_cols = stable_feature_columns(df)
    mlp_out = ensure_dir(os.path.join(out_root, "mlp"))

    metrics = train_one_mlp(
        df=df,
        config=config,
        output_dir=mlp_out,
        feature_columns=all_feature_cols,
        device_name=args.device,
        epochs_override=args.epochs,
        save_artifacts=True,
    )
    print(f"Wrote MLP artifacts: {mlp_out}")

    run_ablation = bool(config.get("mlp", {}).get("group_ablation", True))
    if args.group_ablation:
        run_ablation = True
    if args.no_group_ablation:
        run_ablation = False
    if run_ablation:
        analysis_dir = ensure_dir(os.path.join(out_root, "analysis"))
        rows = []
        for group in ["noise", "srm", "glcm", "frequency", "all"]:
            cols = feature_group_columns(all_feature_cols, group)
            if not cols:
                continue
            group_dir = ensure_dir(os.path.join(mlp_out, "group_ablation", group))
            group_metrics = train_one_mlp(
                df=df,
                config=config,
                output_dir=group_dir,
                feature_columns=cols,
                device_name=args.device,
                epochs_override=args.epochs,
                save_artifacts=False,
            )
            rows.append(
                {
                    "group": group,
                    "n_features": len(cols),
                    "test_accuracy": group_metrics["test"]["accuracy"],
                    "test_balanced_accuracy": group_metrics["test"]["balanced_accuracy"],
                    "test_AUROC": group_metrics["test"]["AUROC"],
                    "test_AP": group_metrics["test"]["AP"],
                    "macro_balanced_accuracy": group_metrics["per_generator_summary"].get("macro_balanced_accuracy"),
                    "macro_AUROC": group_metrics["per_generator_summary"].get("macro_AUROC"),
                    "macro_AP": group_metrics["per_generator_summary"].get("macro_AP"),
                    "worst_generator": group_metrics["per_generator_summary"].get("worst_generator_by_balanced_accuracy"),
                    "worst_balanced_accuracy": group_metrics["per_generator_summary"].get("worst_balanced_accuracy"),
                }
            )
        ablation = pd.DataFrame(rows)
        ablation.to_csv(os.path.join(analysis_dir, "group_ablation_mlp.csv"), index=False)
        ablation.to_json(os.path.join(analysis_dir, "group_ablation_mlp.json"), orient="records", indent=2)


if __name__ == "__main__":
    main()
