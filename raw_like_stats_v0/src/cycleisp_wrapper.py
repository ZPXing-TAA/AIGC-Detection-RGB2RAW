from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch


class CycleISPRgb2Raw:
    def __init__(self, repo_dir: str, checkpoint_path: str, device: str = "cuda:0"):
        self.repo_dir = str(Path(repo_dir))
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        self._install_gaussian_blur_shim()

        from networks.cycleisp import Rgb2Raw

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = Rgb2Raw()
        self._load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _install_gaussian_blur_shim(self) -> None:
        """Load only CycleISP's GaussianBlur helper without importing its legacy utils package."""
        if "utils.GaussianBlur" in sys.modules:
            return
        utils_dir = Path(self.repo_dir) / "utils"
        gaussian_path = utils_dir / "GaussianBlur.py"
        pkg = types.ModuleType("utils")
        pkg.__path__ = [str(utils_dir)]
        sys.modules["utils"] = pkg
        spec = importlib.util.spec_from_file_location("utils.GaussianBlur", str(gaussian_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load CycleISP GaussianBlur from {gaussian_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["utils.GaussianBlur"] = module
        spec.loader.exec_module(module)
        setattr(pkg, "GaussianBlur", module)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]
            cleaned[key] = value
        self.model.load_state_dict(cleaned)

    def __call__(self, batch_chw: Iterable[np.ndarray]) -> List[np.ndarray]:
        arrays = [np.asarray(x, dtype=np.float32) for x in batch_chw]
        if not arrays:
            return []
        tensor = torch.from_numpy(np.stack(arrays, axis=0)).to(self.device)
        with torch.no_grad():
            raw = self.model(tensor)
            raw = torch.clamp(raw, 0.0, 1.0)
        raw_np = raw.detach().cpu().numpy().astype(np.float32)
        return [raw_np[i] for i in range(raw_np.shape[0])]
