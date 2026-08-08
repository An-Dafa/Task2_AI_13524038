"""Reproducible stratified validation utilities for Task #2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

TARGET = "loan_status"
ID_COL = "person_id"


@dataclass(frozen=True)
class HoldoutData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: np.ndarray
    y_valid: np.ndarray
    train_indices: np.ndarray
    valid_indices: np.ndarray


def make_stratified_holdout(
    train: pd.DataFrame,
    *,
    target_column: str = TARGET,
    test_size: float = 0.20,
    random_state: int = 42,
) -> HoldoutData:
    """Create one reproducible stratified holdout split."""
    if target_column not in train.columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size harus berada di antara 0 dan 1.")

    y = train[target_column].to_numpy(dtype=np.int64)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Target harus biner: 0 dan 1.")

    indices = np.arange(len(train))
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    feature_columns = [column for column in train.columns if column != target_column]
    X_train = train.iloc[train_idx][feature_columns].copy()
    X_valid = train.iloc[valid_idx][feature_columns].copy()
    y_train = y[train_idx]
    y_valid = y[valid_idx]

    if ID_COL in X_train.columns:
        overlap = set(X_train[ID_COL]).intersection(set(X_valid[ID_COL]))
        if overlap:
            raise RuntimeError("Ditemukan person_id yang overlap antara train dan validation.")

    return HoldoutData(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
        train_indices=train_idx,
        valid_indices=valid_idx,
    )


def class_distribution(y: np.ndarray) -> dict[str, dict[str, float | int]]:
    values = np.asarray(y, dtype=np.int64).reshape(-1)
    total = values.size
    return {
        str(label): {
            "count": int(np.sum(values == label)),
            "proportion": float(np.mean(values == label)),
        }
        for label in (0, 1)
    } | {"total": {"count": int(total), "proportion": 1.0}}


def holdout_summary(split: HoldoutData) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for split_name, labels in (("train", split.y_train), ("validation", split.y_valid)):
        for class_label in (0, 1):
            rows.append(
                {
                    "split": split_name,
                    "class": class_label,
                    "count": int(np.sum(labels == class_label)),
                    "proportion": float(np.mean(labels == class_label)),
                }
            )
    return pd.DataFrame(rows)


def make_stratified_folds(
    y: np.ndarray | pd.Series,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(fold_number, train_indices, validation_indices)``."""
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    if n_splits < 2:
        raise ValueError("n_splits minimal 2.")
    if min(np.bincount(labels, minlength=2)) < n_splits:
        raise ValueError("Setiap kelas harus memiliki setidaknya n_splits sampel.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    dummy_features = np.zeros((len(labels), 1), dtype=float)
    for fold_number, (train_idx, valid_idx) in enumerate(
        splitter.split(dummy_features, labels), start=1
    ):
        yield fold_number, train_idx, valid_idx


def fold_distribution_summary(
    y: np.ndarray | pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    rows: list[dict[str, float | int | str]] = []
    for fold_number, train_idx, valid_idx in make_stratified_folds(
        labels, n_splits=n_splits, shuffle=True, random_state=random_state
    ):
        for subset_name, indices in (("train", train_idx), ("validation", valid_idx)):
            subset = labels[indices]
            rows.append(
                {
                    "fold": fold_number,
                    "subset": subset_name,
                    "n_samples": int(len(indices)),
                    "class_0_count": int(np.sum(subset == 0)),
                    "class_1_count": int(np.sum(subset == 1)),
                    "class_0_proportion": float(np.mean(subset == 0)),
                    "class_1_proportion": float(np.mean(subset == 1)),
                }
            )
    return pd.DataFrame(rows)
