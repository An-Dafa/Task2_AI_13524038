"""Final manual Decision Tree Learning model used for the rank-1 submission.

The model is an entropy / information-gain binary decision tree implemented with
NumPy.  Categorical features are one-hot encoded, continuous features are split
by numeric thresholds, and the final configuration uses an exact threshold scan
for ``person_id`` while limiting other features to 512 deterministic candidate
midpoints.

Training data:
- 28,800 competition training rows;
- 9,000 allowed labeled source rows unused by competition train/test;
- external features ``person_education`` and ``loan_intent``.

Important: labels corresponding to competition-test rows in the source dataset
are never used by this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptimizedDTLConfig:
    max_depth: int = 12
    min_samples_leaf: int = 10
    min_samples_split: int = 20
    positive_weight_multiplier: float = 0.84
    external_weight: float = 1.10
    max_thresholds_per_feature: int = 512
    exact_feature_names: tuple[str, ...] = ("person_id",)
    random_state: int = 42


@dataclass
class TreeNode:
    depth: int
    n_samples: int
    probability_1: float
    entropy: float
    feature_index: int | None = None
    threshold: float | None = None
    information_gain: float = 0.0
    left: "TreeNode | None" = field(default=None, repr=False)
    right: "TreeNode | None" = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class OneHotFeatureEncoder:
    """Fold-safe full one-hot encoding without scaling.

    Numeric features are kept unchanged. Every observed categorical level gets a
    separate 0/1 column, e.g. ``loan_intent=VENTURE``. Unknown levels transform
    to all-zero columns for that categorical feature.
    """

    def __init__(self, feature_names: Iterable[str]) -> None:
        self.requested_features = list(feature_names)
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.category_levels_: dict[str, list[str]] = {}
        self.feature_names_: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "OneHotFeatureEncoder":
        self.numeric_columns_ = [
            column for column in self.requested_features
            if pd.api.types.is_numeric_dtype(frame[column])
        ]
        self.categorical_columns_ = [
            column for column in self.requested_features
            if column not in self.numeric_columns_
        ]
        self.category_levels_ = {
            column: sorted(frame[column].astype(str).unique().tolist())
            for column in self.categorical_columns_
        }
        one_hot_names = [
            f"{column}={level}"
            for column in self.categorical_columns_
            for level in self.category_levels_[column]
        ]
        self.feature_names_ = self.numeric_columns_ + one_hot_names
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = [
            frame[self.numeric_columns_].astype(float).to_numpy(copy=True)
        ]
        for column in self.categorical_columns_:
            values = frame[column].astype(str).to_numpy()
            levels = self.category_levels_[column]
            blocks.append(
                np.column_stack(
                    [(values == level).astype(float) for level in levels]
                )
            )
        matrix = np.column_stack(blocks).astype(float, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError("Encoded matrix contains NaN/inf.")
        return matrix

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


class OptimizedEntropyTree:
    def __init__(self, config: OptimizedDTLConfig | None = None) -> None:
        self.config = config or OptimizedDTLConfig()
        self.tree_: TreeNode | None = None
        self.feature_names_: list[str] = []
        self.exact_features_: set[int] = set()
        self.n_features_in_: int = 0
        self.n_nodes_: int = 0
        self.n_leaves_: int = 0
        self.max_depth_reached_: int = 0
        self.feature_importances_: np.ndarray | None = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._w: np.ndarray | None = None
        self._raw_importances: np.ndarray | None = None

    @staticmethod
    def _entropy(weight_0, weight_1):
        w0 = np.asarray(weight_0, dtype=float)
        w1 = np.asarray(weight_1, dtype=float)
        total = w0 + w1
        p0 = np.divide(w0, total, out=np.zeros_like(total), where=total > 0)
        p1 = np.divide(w1, total, out=np.zeros_like(total), where=total > 0)
        result = np.zeros_like(total, dtype=float)
        mask0 = p0 > 0
        mask1 = p1 > 0
        result[mask0] -= p0[mask0] * np.log2(p0[mask0])
        result[mask1] -= p1[mask1] * np.log2(p1[mask1])
        return result

    def _node_stats(self, indices: np.ndarray):
        assert self._y is not None and self._w is not None
        labels = self._y[indices]
        weights = self._w[indices]
        weight_0 = float(weights[labels == 0].sum())
        weight_1 = float(weights[labels == 1].sum())
        total = weight_0 + weight_1
        probability_1 = 0.0 if total <= 0 else weight_1 / total
        return weight_0, weight_1, probability_1, float(
            self._entropy(weight_0, weight_1)
        )

    def _candidate_positions(
        self,
        sorted_values: np.ndarray,
        *,
        exact: bool,
    ) -> np.ndarray:
        valid = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
        if valid.size == 0:
            return valid
        left_n = valid + 1
        right_n = sorted_values.size - left_n
        valid = valid[
            (left_n >= self.config.min_samples_leaf)
            & (right_n >= self.config.min_samples_leaf)
        ]
        if exact or valid.size <= self.config.max_thresholds_per_feature:
            return valid
        selected = np.linspace(
            0,
            valid.size - 1,
            num=self.config.max_thresholds_per_feature,
        ).round().astype(int)
        return valid[np.unique(selected)]

    def _best_split_for_feature(
        self,
        indices: np.ndarray,
        feature_index: int,
        parent_entropy: float,
        parent_weight: float,
    ) -> tuple[float | None, float]:
        assert self._X is not None and self._y is not None and self._w is not None
        values = self._X[indices, feature_index]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        positions = self._candidate_positions(
            sorted_values,
            exact=feature_index in self.exact_features_,
        )
        if positions.size == 0:
            return None, 0.0

        sorted_indices = indices[order]
        labels = self._y[sorted_indices]
        weights = self._w[sorted_indices]
        cumulative_weight = np.cumsum(weights)
        cumulative_positive = np.cumsum(weights * labels)
        total_weight = float(cumulative_weight[-1])
        total_positive = float(cumulative_positive[-1])

        left_weight = cumulative_weight[positions]
        left_positive = cumulative_positive[positions]
        left_negative = left_weight - left_positive
        right_weight = total_weight - left_weight
        right_positive = total_positive - left_positive
        right_negative = right_weight - right_positive

        child_entropy = (
            left_weight * self._entropy(left_negative, left_positive)
            + right_weight * self._entropy(right_negative, right_positive)
        ) / parent_weight
        gains = parent_entropy - child_entropy
        best_index = int(np.argmax(gains))
        position = int(positions[best_index])
        threshold = float(
            sorted_values[position]
            + (sorted_values[position + 1] - sorted_values[position]) / 2.0
        )
        return threshold, float(gains[best_index])

    def _best_split(
        self,
        indices: np.ndarray,
        parent_entropy: float,
        parent_weight: float,
    ) -> tuple[int | None, float | None, float]:
        best_feature: int | None = None
        best_threshold: float | None = None
        best_gain = 0.0
        for feature_index in range(self.n_features_in_):
            threshold, gain = self._best_split_for_feature(
                indices,
                feature_index,
                parent_entropy,
                parent_weight,
            )
            if threshold is not None and gain > best_gain:
                best_feature = feature_index
                best_threshold = threshold
                best_gain = gain
        return best_feature, best_threshold, best_gain

    def _grow(self, indices: np.ndarray, depth: int) -> TreeNode:
        assert self._X is not None
        w0, w1, p1, entropy = self._node_stats(indices)
        node = TreeNode(depth, int(indices.size), p1, entropy)
        self.n_nodes_ += 1
        self.max_depth_reached_ = max(self.max_depth_reached_, depth)

        if (
            entropy <= 1e-15
            or depth >= self.config.max_depth
            or indices.size < self.config.min_samples_split
            or indices.size < 2 * self.config.min_samples_leaf
        ):
            self.n_leaves_ += 1
            return node

        feature, threshold, gain = self._best_split(
            indices,
            entropy,
            w0 + w1,
        )
        if feature is None or threshold is None:
            self.n_leaves_ += 1
            return node

        values = self._X[indices, feature]
        left_mask = values <= threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        if (
            left_indices.size < self.config.min_samples_leaf
            or right_indices.size < self.config.min_samples_leaf
        ):
            self.n_leaves_ += 1
            return node

        node.feature_index = feature
        node.threshold = threshold
        node.information_gain = gain
        assert self._raw_importances is not None
        self._raw_importances[feature] += (w0 + w1) * gain
        node.left = self._grow(left_indices, depth + 1)
        node.right = self._grow(right_indices, depth + 1)
        return node

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: list[str],
        source_is_external: np.ndarray | None = None,
    ) -> "OptimizedEntropyTree":
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=int).reshape(-1)
        if features.shape[0] != labels.size:
            raise ValueError("X/y length mismatch.")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Binary labels 0/1 required.")

        self.feature_names_ = list(feature_names)
        self.n_features_in_ = features.shape[1]
        self.exact_features_ = {
            self.feature_names_.index(name)
            for name in self.config.exact_feature_names
            if name in self.feature_names_
        }

        counts = np.bincount(labels, minlength=2)
        positive_weight = (
            float(counts[0] / counts[1])
            * self.config.positive_weight_multiplier
        )
        class_weight = np.where(labels == 1, positive_weight, 1.0)
        if source_is_external is None:
            source_weight = np.ones(labels.size, dtype=float)
        else:
            external = np.asarray(source_is_external, dtype=bool)
            if external.size != labels.size:
                raise ValueError("source_is_external length mismatch.")
            source_weight = np.where(external, self.config.external_weight, 1.0)

        self._X = features
        self._y = labels
        self._w = class_weight * source_weight
        self._raw_importances = np.zeros(self.n_features_in_, dtype=float)
        self.n_nodes_ = 0
        self.n_leaves_ = 0
        self.max_depth_reached_ = 0
        self.tree_ = self._grow(np.arange(labels.size, dtype=int), 0)

        total = float(self._raw_importances.sum())
        self.feature_importances_ = (
            self._raw_importances / total
            if total > 0 else self._raw_importances.copy()
        )
        self._X = self._y = self._w = self._raw_importances = None
        return self

    @staticmethod
    def _leaf(row: np.ndarray, root: TreeNode) -> TreeNode:
        node = root
        while not node.is_leaf:
            assert node.feature_index is not None and node.threshold is not None
            node = node.left if row[node.feature_index] <= node.threshold else node.right
            assert node is not None
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Model belum di-fit.")
        features = np.asarray(X, dtype=float)
        p1 = np.fromiter(
            (self._leaf(row, self.tree_).probability_1 for row in features),
            dtype=float,
            count=features.shape[0],
        )
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int64)

    def split_counts(self) -> dict[str, int]:
        if self.tree_ is None:
            raise RuntimeError("Model belum di-fit.")
        counts: dict[str, int] = {}
        stack = [self.tree_]
        while stack:
            node = stack.pop()
            if node.feature_index is None:
                continue
            name = self.feature_names_[node.feature_index]
            counts[name] = counts.get(name, 0) + 1
            assert node.left is not None and node.right is not None
            stack.extend([node.left, node.right])
        return counts
