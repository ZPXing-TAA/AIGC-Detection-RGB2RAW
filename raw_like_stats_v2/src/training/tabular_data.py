from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.io_utils import stable_feature_columns


@dataclass
class FeaturePreprocessor:
    feature_columns: List[str]
    imputer: SimpleImputer
    scaler: StandardScaler

    @classmethod
    def fit(cls, df_train: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> "FeaturePreprocessor":
        columns = feature_columns or stable_feature_columns(df_train)
        if not columns:
            raise ValueError("no statistical feature columns found")
        x = df_train[columns].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_imp = imputer.fit_transform(x)
        scaler.fit(x_imp)
        return cls(feature_columns=list(columns), imputer=imputer, scaler=scaler)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_columns].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
        x = self.imputer.transform(x)
        x = self.scaler.transform(x)
        return x.astype(np.float32)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "feature_columns": self.feature_columns,
                "imputer": self.imputer,
                "scaler": self.scaler,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "FeaturePreprocessor":
        obj = joblib.load(path)
        return cls(
            feature_columns=list(obj["feature_columns"]),
            imputer=obj["imputer"],
            scaler=obj["scaler"],
        )


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

