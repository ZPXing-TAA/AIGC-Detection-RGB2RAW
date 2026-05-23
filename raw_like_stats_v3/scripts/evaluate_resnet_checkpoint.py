#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aigibench_dataset import AIGIBenchRgbRawDataset
from src.models import build_model
from src.training.metrics import compute_detection_metrics
from src.utils.config import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved ResNet checkpoint without retraining.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument(
        "--model-mode",
        default=None,
        choices=["rgb_only_resnet50", "raw_only_resnet50", "two_stream_resnet50"],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--predictions-name", default=None)
    return parser.parse_args()


def build_dataset(cfg: Dict[str, Any], split: str, model_mode: str, limit: int = None) -> AIGIBenchRgbRawDataset:
    data_cfg = cfg["data"]
    return AIGIBenchRgbRawDataset(
        manifest_path=data_cfg["manifest_path"],
        split=split,
        image_size=int(data_cfg["image_size"]),
        rgb_mean=data_cfg["rgb_mean"],
        rgb_std=data_cfg["rgb_std"],
        raw_mean=data_cfg["raw_mean"],
        raw_std=data_cfg["raw_std"],
        raw_size=int(data_cfg.get("raw_cache_size", data_cfg["image_size"])),
        use_rgb=model_mode in ("rgb_only_resnet50", "two_stream_resnet50"),
        use_raw=model_mode in ("raw_only_resnet50", "two_stream_resnet50"),
        limit=limit,
    )


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, split: str) -> Dict[str, Any]:
    model.eval()
    labels_all: List[int] = []
    scores_all: List[float] = []
    rows: List[Dict[str, Any]] = []
    losses: List[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=split, leave=False):
            batch = move_batch_to_device(batch, device)
            labels = batch["label"]
            logits = model(rgb=batch.get("rgb"), raw_like=batch.get("raw_like"))
            loss = criterion(logits, labels)
            scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(int)
            preds = (scores >= 0.5).astype(int)
            losses.append(float(loss.detach().cpu().item()) * int(labels.size(0)))
            labels_all.extend(labels_np.tolist())
            scores_all.extend(scores.tolist())
            for i in range(len(labels_np)):
                rows.append(
                    {
                        "image_id": batch["image_id"][i],
                        "dataset": batch["dataset"][i],
                        "split": batch["split"][i],
                        "subset": batch["subset"][i],
                        "generator": batch["generator"][i],
                        "generator_family": batch["generator_family"][i],
                        "source": batch["source"][i],
                        "path": batch["path"][i],
                        "raw_like_path": batch["raw_like_path"][i],
                        "label": int(labels_np[i]),
                        "score_fake": float(scores[i]),
                        "pred_0p5": int(preds[i]),
                    }
                )
    metrics = compute_detection_metrics(labels_all, scores_all, generators=[r["generator"] for r in rows])
    metrics["loss"] = float(sum(losses) / max(1, len(labels_all)))
    return {"metrics": metrics, "predictions": pd.DataFrame(rows)}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = int(args.num_workers)
    model_mode = args.model_mode or cfg["model"]["name"]
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg["project"]["output_dir"]) / model_mode
    ensure_dir(out_dir)

    checkpoint = torch.load(str(args.checkpoint), map_location=device)
    checkpoint_mode = checkpoint.get("model_mode")
    if checkpoint_mode and checkpoint_mode != model_mode:
        raise ValueError("Checkpoint model_mode={} does not match requested {}".format(checkpoint_mode, model_mode))

    model = build_model(
        model_mode,
        num_classes=int(cfg["model"].get("num_classes", 2)),
        pretrained=False,
        dropout=float(cfg["model"].get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    ds = build_dataset(cfg, args.split, model_mode, args.limit)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )
    result = evaluate(model, loader, nn.CrossEntropyLoss(), device, args.split)
    pred_name = args.predictions_name or "predictions_{}.csv".format(args.split)
    result["predictions"].to_csv(str(out_dir / pred_name), index=False)
    pd.DataFrame(result["metrics"].get("per_generator_tpr", [])).to_csv(
        str(out_dir / "per_generator_{}.csv".format(args.split)),
        index=False,
    )

    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        with open(str(metrics_path)) as f:
            payload = json.load(f)
    else:
        payload = {"model_mode": model_mode}
    payload[args.split] = result["metrics"]
    payload["eval_checkpoint"] = str(args.checkpoint)
    with open(str(metrics_path), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("Saved {} predictions and metrics under {}".format(args.split, out_dir))


if __name__ == "__main__":
    main()
