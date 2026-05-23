from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_json(payload: Dict[str, Any], path: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_abs(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def get_device(device_arg: str = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg if torch.cuda.is_available() or not device_arg.startswith("cuda") else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_snapshot() -> Dict[str, Any]:
    return {
        "cwd": os.getcwd(),
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }

