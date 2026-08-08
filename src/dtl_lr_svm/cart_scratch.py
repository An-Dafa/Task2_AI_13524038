"""NumPy-only CART classifier for Task #2.

The estimator implements binary Classification and Regression Trees (CART)
without importing scikit-learn.  Splits use weighted Gini impurity, numeric
thresholds, recursive binary partitioning, optional class weighting, and
standard stopping criteria.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Mapping
import json

import numpy as np

ClassWeight = Literal["balanced"] | Mapping[int, float] | None
MaxFeatures = int | float | Literal["sqrt", "log2"] | None


@dataclass(frozen=True)
class CARTConfig:
    """Hyperparameters for :class:`CARTClassifierScratch`."""

    max_depth: int | None = 8
    min_samples_split: int = 20
    min_samples_leaf: int = 10
    min_impurity_decrease: float = 0.0
    class_weight: ClassWeight = None
    max_features: MaxFeatures = None
    max_thresholds_per_feature: int | None = 128
    random_state: int = 42
    verbose: int = 0


@dataclass
class CARTNode:
    """One node in a binary CART tree."""

    node_id: int
    depth: int
    n_samples: int
    weighted_n_samples: float
    impurity: float
    class_counts: tuple[int, int]
    weighted_class_counts: tuple[float, float]
    probability_1: float
    prediction: int
    feature_index: int | None = None
    threshold: float | None = None
    impurity_decrease: float = 0.0
    left: "CARTNode | None" = field(default=None, repr=False)
    right: "CARTNode | None" = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class CARTClassifierScratch:
    """Binary CART classifier trained with weighted Gini impurity.

    Notes
    -----
    ``max_thresholds_per_feature`` controls the split-search cost.  ``None``
    evaluates every valid midpoint.  A positive integer samples that many
    evenly-spaced valid candidate positions for each feature, preserving a
    deterministic and transparent approximation for faster experiments.
    """

    def __init__(self, config: CARTConfig | None = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Gunakan config atau keyword arguments, bukan keduanya.")
        self.config = config or CARTConfig(**kwargs)
        self.tree_: CARTNode | None = None
        self.classes_: np.ndarray = np.array([0, 1], dtype=np.int64)
        self.class_weight_: dict[int, float] = {0: 1.0, 1: 1.0}
        self.n_features_in_: int | None = None
        self.n_nodes_: int = 0
        self.n_leaves_: int = 0
        self.max_depth_reached_: int = 0
        self.feature_importances_: np.ndarray | None = None
        self.is_fitted_: bool = False
        self._next_node_id: int = 0
        self._raw_importances: np.ndarray | None = None
        self._rng = np.random.default_rng(self.config.random_state)
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._sample_weight: np.ndarray | None = None
        self._validate_config()

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.max_depth is not None and cfg.max_depth < 1:
            raise ValueError("max_depth harus None atau minimal 1.")
        if cfg.min_samples_split < 2:
            raise ValueError("min_samples_split minimal 2.")
        if cfg.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf minimal 1.")
        if cfg.min_samples_split < 2 * cfg.min_samples_leaf:
            raise ValueError(
                "min_samples_split minimal dua kali min_samples_leaf agar split mungkin."
            )
        if cfg.min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease tidak boleh negatif.")
        if isinstance(cfg.class_weight, str) and cfg.class_weight != "balanced":
            raise ValueError("class_weight string hanya boleh 'balanced'.")
        if isinstance(cfg.max_features, int) and cfg.max_features < 1:
            raise ValueError("max_features integer minimal 1.")
        if isinstance(cfg.max_features, float) and not 0.0 < cfg.max_features <= 1.0:
            raise ValueError("max_features float harus berada pada (0, 1].")
        if isinstance(cfg.max_features, str) and cfg.max_features not in {"sqrt", "log2"}:
            raise ValueError("max_features string harus 'sqrt' atau 'log2'.")
        if (
            cfg.max_thresholds_per_feature is not None
            and cfg.max_thresholds_per_feature < 1
        ):
            raise ValueError("max_thresholds_per_feature harus None atau minimal 1.")
        if cfg.verbose < 0:
            raise ValueError("verbose tidak boleh negatif.")

    @staticmethod
    def _validate_X(X: np.ndarray, expected_features: int | None = None) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("X harus berupa matriks 2 dimensi.")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("X tidak boleh kosong.")
        if expected_features is not None and matrix.shape[1] != expected_features:
            raise ValueError(
                f"Jumlah fitur tidak cocok: diharapkan {expected_features}, "
                f"diperoleh {matrix.shape[1]}."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("X mengandung NaN atau nilai tak hingga.")
        return matrix

    @staticmethod
    def _validate_y(y: np.ndarray, n_samples: int) -> np.ndarray:
        labels = np.asarray(y, dtype=np.int64).reshape(-1)
        if labels.shape[0] != n_samples:
            raise ValueError("Panjang y harus sama dengan jumlah baris X.")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("CART ini hanya menerima label biner 0/1.")
        if np.unique(labels).size < 2:
            raise ValueError("Data training harus memuat kedua kelas 0 dan 1.")
        return labels

    def _resolve_class_weight(self, y: np.ndarray) -> dict[int, float]:
        setting = self.config.class_weight
        if setting is None:
            return {0: 1.0, 1: 1.0}
        if setting == "balanced":
            counts = np.bincount(y, minlength=2)
            if np.any(counts == 0):
                raise ValueError("class_weight balanced membutuhkan kedua kelas.")
            n_samples = y.size
            return {
                0: float(n_samples / (2.0 * counts[0])),
                1: float(n_samples / (2.0 * counts[1])),
            }
        weights = {int(key): float(value) for key, value in setting.items()}
        if set(weights) != {0, 1}:
            raise ValueError("class_weight dictionary harus memiliki key 0 dan 1.")
        if any(value <= 0 for value in weights.values()):
            raise ValueError("Semua class weight harus lebih besar dari 0.")
        return weights

    @staticmethod
    def _gini_from_weighted_counts(weight_0: float, weight_1: float) -> float:
        total = weight_0 + weight_1
        if total <= 0.0:
            return 0.0
        p0 = weight_0 / total
        p1 = weight_1 / total
        return float(1.0 - p0 * p0 - p1 * p1)

    def _resolve_max_features(self) -> int:
        assert self.n_features_in_ is not None
        setting = self.config.max_features
        if setting is None:
            return self.n_features_in_
        if isinstance(setting, int):
            return min(setting, self.n_features_in_)
        if isinstance(setting, float):
            return max(1, int(np.ceil(setting * self.n_features_in_)))
        if setting == "sqrt":
            return max(1, int(np.sqrt(self.n_features_in_)))
        return max(1, int(np.log2(self.n_features_in_)))

    def _feature_subset(self) -> np.ndarray:
        assert self.n_features_in_ is not None
        count = self._resolve_max_features()
        if count == self.n_features_in_:
            return np.arange(self.n_features_in_, dtype=np.int64)
        return np.sort(
            self._rng.choice(self.n_features_in_, size=count, replace=False)
        ).astype(np.int64)

    def _node_statistics(
        self, indices: np.ndarray
    ) -> tuple[int, float, tuple[int, int], tuple[float, float], float, float, int]:
        assert self._y is not None and self._sample_weight is not None
        labels = self._y[indices]
        weights = self._sample_weight[indices]
        count_0 = int(np.sum(labels == 0))
        count_1 = int(labels.size - count_0)
        weighted_0 = float(np.sum(weights[labels == 0]))
        weighted_1 = float(np.sum(weights[labels == 1]))
        weighted_total = weighted_0 + weighted_1
        impurity = self._gini_from_weighted_counts(weighted_0, weighted_1)
        probability_1 = 0.0 if weighted_total <= 0 else weighted_1 / weighted_total
        prediction = int(probability_1 >= 0.5)
        return (
            int(labels.size),
            weighted_total,
            (count_0, count_1),
            (weighted_0, weighted_1),
            impurity,
            float(probability_1),
            prediction,
        )

    def _candidate_positions(self, values: np.ndarray) -> np.ndarray:
        positions = np.flatnonzero(values[:-1] < values[1:])
        if positions.size == 0:
            return positions
        leaf = self.config.min_samples_leaf
        positions = positions[
            (positions + 1 >= leaf) & (values.size - positions - 1 >= leaf)
        ]
        limit = self.config.max_thresholds_per_feature
        if limit is not None and positions.size > limit:
            selector = np.linspace(0, positions.size - 1, num=limit, dtype=np.int64)
            positions = positions[np.unique(selector)]
        return positions

    def _best_split(
        self,
        indices: np.ndarray,
        parent_impurity: float,
        parent_weight: float,
    ) -> tuple[int | None, float | None, float]:
        assert self._X is not None and self._y is not None and self._sample_weight is not None
        best_feature: int | None = None
        best_threshold: float | None = None
        best_gain = -np.inf

        for feature_index in self._feature_subset():
            values = self._X[indices, feature_index]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            positions = self._candidate_positions(sorted_values)
            if positions.size == 0:
                continue

            sorted_indices = indices[order]
            labels = self._y[sorted_indices]
            weights = self._sample_weight[sorted_indices]
            weighted_positive = weights * labels

            cumulative_weight = np.cumsum(weights)
            cumulative_positive = np.cumsum(weighted_positive)
            total_weight = float(cumulative_weight[-1])
            total_positive = float(cumulative_positive[-1])

            left_weight = cumulative_weight[positions]
            left_positive = cumulative_positive[positions]
            left_negative = left_weight - left_positive
            right_weight = total_weight - left_weight
            right_positive = total_positive - left_positive
            right_negative = right_weight - right_positive

            left_p0 = np.divide(
                left_negative,
                left_weight,
                out=np.zeros_like(left_weight),
                where=left_weight > 0,
            )
            left_p1 = np.divide(
                left_positive,
                left_weight,
                out=np.zeros_like(left_weight),
                where=left_weight > 0,
            )
            right_p0 = np.divide(
                right_negative,
                right_weight,
                out=np.zeros_like(right_weight),
                where=right_weight > 0,
            )
            right_p1 = np.divide(
                right_positive,
                right_weight,
                out=np.zeros_like(right_weight),
                where=right_weight > 0,
            )
            left_gini = 1.0 - left_p0**2 - left_p1**2
            right_gini = 1.0 - right_p0**2 - right_p1**2
            child_impurity = (
                left_weight * left_gini + right_weight * right_gini
            ) / parent_weight
            gains = parent_impurity - child_impurity
            local_index = int(np.argmax(gains))
            gain = float(gains[local_index])

            if gain > best_gain + 1e-15:
                position = int(positions[local_index])
                left_value = float(sorted_values[position])
                right_value = float(sorted_values[position + 1])
                threshold = left_value + (right_value - left_value) / 2.0
                best_feature = int(feature_index)
                best_threshold = float(threshold)
                best_gain = gain

        if best_feature is None or best_gain < self.config.min_impurity_decrease:
            return None, None, 0.0
        return best_feature, best_threshold, best_gain

    def _make_node(self, indices: np.ndarray, depth: int) -> CARTNode:
        (
            n_samples,
            weighted_n_samples,
            class_counts,
            weighted_class_counts,
            impurity,
            probability_1,
            prediction,
        ) = self._node_statistics(indices)
        node = CARTNode(
            node_id=self._next_node_id,
            depth=depth,
            n_samples=n_samples,
            weighted_n_samples=weighted_n_samples,
            impurity=impurity,
            class_counts=class_counts,
            weighted_class_counts=weighted_class_counts,
            probability_1=probability_1,
            prediction=prediction,
        )
        self._next_node_id += 1
        self.n_nodes_ += 1
        self.max_depth_reached_ = max(self.max_depth_reached_, depth)
        return node

    def _should_stop(self, node: CARTNode) -> bool:
        if node.impurity <= 1e-15:
            return True
        if self.config.max_depth is not None and node.depth >= self.config.max_depth:
            return True
        if node.n_samples < self.config.min_samples_split:
            return True
        if node.n_samples < 2 * self.config.min_samples_leaf:
            return True
        return False

    def _grow(self, indices: np.ndarray, depth: int) -> CARTNode:
        assert self._X is not None
        node = self._make_node(indices, depth)
        if self._should_stop(node):
            self.n_leaves_ += 1
            return node

        feature, threshold, gain = self._best_split(
            indices, node.impurity, node.weighted_n_samples
        )
        if feature is None or threshold is None or gain <= 0.0:
            self.n_leaves_ += 1
            return node

        mask = self._X[indices, feature] <= threshold
        left_indices = indices[mask]
        right_indices = indices[~mask]
        if (
            left_indices.size < self.config.min_samples_leaf
            or right_indices.size < self.config.min_samples_leaf
        ):
            self.n_leaves_ += 1
            return node

        node.feature_index = feature
        node.threshold = threshold
        node.impurity_decrease = gain
        assert self._raw_importances is not None
        self._raw_importances[feature] += node.weighted_n_samples * gain

        if self.config.verbose >= 2:
            print(
                f"[CART] node={node.node_id} depth={depth} n={node.n_samples} "
                f"feature={feature} threshold={threshold:.6g} gain={gain:.6g}"
            )

        node.left = self._grow(left_indices, depth + 1)
        node.right = self._grow(right_indices, depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CARTClassifierScratch":
        """Fit a binary CART classifier."""
        features = self._validate_X(X)
        labels = self._validate_y(y, features.shape[0])
        self.n_features_in_ = int(features.shape[1])
        self.class_weight_ = self._resolve_class_weight(labels)
        weights = np.where(
            labels == 1, self.class_weight_[1], self.class_weight_[0]
        ).astype(np.float64)

        self._X = features
        self._y = labels
        self._sample_weight = weights
        self._rng = np.random.default_rng(self.config.random_state)
        self._next_node_id = 0
        self.n_nodes_ = 0
        self.n_leaves_ = 0
        self.max_depth_reached_ = 0
        self._raw_importances = np.zeros(self.n_features_in_, dtype=np.float64)

        root_indices = np.arange(features.shape[0], dtype=np.int64)
        self.tree_ = self._grow(root_indices, depth=0)
        total_importance = float(np.sum(self._raw_importances))
        if total_importance > 0:
            self.feature_importances_ = self._raw_importances / total_importance
        else:
            self.feature_importances_ = self._raw_importances.copy()

        self.is_fitted_ = True
        self._X = None
        self._y = None
        self._sample_weight = None
        self._raw_importances = None
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted_ or self.tree_ is None or self.n_features_in_ is None:
            raise RuntimeError("CART belum di-fit.")

    @staticmethod
    def _leaf_for_row(row: np.ndarray, root: CARTNode) -> CARTNode:
        node = root
        while not node.is_leaf:
            assert node.feature_index is not None and node.threshold is not None
            if row[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self.tree_ is not None and self.n_features_in_ is not None
        features = self._validate_X(X, expected_features=self.n_features_in_)
        probabilities_1 = np.fromiter(
            (
                self._leaf_for_row(row, self.tree_).probability_1
                for row in features
            ),
            dtype=np.float64,
            count=features.shape[0],
        )
        return np.column_stack((1.0 - probabilities_1, probabilities_1))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold harus berada pada [0, 1].")
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int64)

    def apply(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self.tree_ is not None and self.n_features_in_ is not None
        features = self._validate_X(X, expected_features=self.n_features_in_)
        return np.fromiter(
            (self._leaf_for_row(row, self.tree_).node_id for row in features),
            dtype=np.int64,
            count=features.shape[0],
        )

    def iter_nodes(self) -> list[CARTNode]:
        self._require_fitted()
        assert self.tree_ is not None
        output: list[CARTNode] = []
        stack = [self.tree_]
        while stack:
            node = stack.pop()
            output.append(node)
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)
        return output

    def export_nodes(self, feature_names: list[str] | None = None) -> list[dict[str, object]]:
        self._require_fitted()
        if feature_names is not None and len(feature_names) != self.n_features_in_:
            raise ValueError("Jumlah feature_names tidak cocok.")
        rows: list[dict[str, object]] = []
        for node in self.iter_nodes():
            feature_name = None
            if node.feature_index is not None:
                feature_name = (
                    feature_names[node.feature_index]
                    if feature_names is not None
                    else f"x{node.feature_index}"
                )
            rows.append(
                {
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "is_leaf": node.is_leaf,
                    "n_samples": node.n_samples,
                    "weighted_n_samples": node.weighted_n_samples,
                    "impurity": node.impurity,
                    "class_0_count": node.class_counts[0],
                    "class_1_count": node.class_counts[1],
                    "weighted_class_0": node.weighted_class_counts[0],
                    "weighted_class_1": node.weighted_class_counts[1],
                    "probability_1": node.probability_1,
                    "prediction": node.prediction,
                    "feature_index": node.feature_index,
                    "feature_name": feature_name,
                    "threshold": node.threshold,
                    "impurity_decrease": node.impurity_decrease,
                    "left_node_id": None if node.left is None else node.left.node_id,
                    "right_node_id": None if node.right is None else node.right.node_id,
                }
            )
        return rows

    def export_text(
        self,
        feature_names: list[str] | None = None,
        max_depth: int | None = None,
    ) -> str:
        self._require_fitted()
        assert self.tree_ is not None
        if feature_names is not None and len(feature_names) != self.n_features_in_:
            raise ValueError("Jumlah feature_names tidak cocok.")

        lines: list[str] = []

        def visit(node: CARTNode, prefix: str) -> None:
            if node.is_leaf or (max_depth is not None and node.depth >= max_depth):
                suffix = " [truncated]" if not node.is_leaf else ""
                lines.append(
                    f"{prefix}leaf{suffix}: predict={node.prediction}, "
                    f"p1={node.probability_1:.4f}, n={node.n_samples}, "
                    f"gini={node.impurity:.4f}"
                )
                return
            assert node.feature_index is not None and node.threshold is not None
            name = (
                feature_names[node.feature_index]
                if feature_names is not None
                else f"x{node.feature_index}"
            )
            lines.append(
                f"{prefix}if {name} <= {node.threshold:.6g} "
                f"(gain={node.impurity_decrease:.6f}, n={node.n_samples}):"
            )
            assert node.left is not None and node.right is not None
            visit(node.left, prefix + "  ")
            lines.append(f"{prefix}else:")
            visit(node.right, prefix + "  ")

        visit(self.tree_, "")
        return "\n".join(lines)

    def get_params(self) -> dict[str, object]:
        return asdict(self.config)

    def metadata(self) -> dict[str, object]:
        self._require_fitted()
        assert self.feature_importances_ is not None
        return {
            "config": self.get_params(),
            "class_weight_resolved": self.class_weight_,
            "n_features_in": self.n_features_in_,
            "n_nodes": self.n_nodes_,
            "n_leaves": self.n_leaves_,
            "max_depth_reached": self.max_depth_reached_,
            "feature_importances": self.feature_importances_.tolist(),
        }

    @staticmethod
    def _node_to_dict(node: CARTNode) -> dict[str, object]:
        return {
            "node_id": node.node_id,
            "depth": node.depth,
            "n_samples": node.n_samples,
            "weighted_n_samples": node.weighted_n_samples,
            "impurity": node.impurity,
            "class_counts": list(node.class_counts),
            "weighted_class_counts": list(node.weighted_class_counts),
            "probability_1": node.probability_1,
            "prediction": node.prediction,
            "feature_index": node.feature_index,
            "threshold": node.threshold,
            "impurity_decrease": node.impurity_decrease,
            "left": None if node.left is None else CARTClassifierScratch._node_to_dict(node.left),
            "right": None if node.right is None else CARTClassifierScratch._node_to_dict(node.right),
        }

    @staticmethod
    def _node_from_dict(payload: dict[str, object]) -> CARTNode:
        left_payload = payload.get("left")
        right_payload = payload.get("right")
        return CARTNode(
            node_id=int(payload["node_id"]),
            depth=int(payload["depth"]),
            n_samples=int(payload["n_samples"]),
            weighted_n_samples=float(payload["weighted_n_samples"]),
            impurity=float(payload["impurity"]),
            class_counts=tuple(int(v) for v in payload["class_counts"]),  # type: ignore[arg-type]
            weighted_class_counts=tuple(
                float(v) for v in payload["weighted_class_counts"]  # type: ignore[arg-type]
            ),
            probability_1=float(payload["probability_1"]),
            prediction=int(payload["prediction"]),
            feature_index=(
                None if payload.get("feature_index") is None else int(payload["feature_index"])
            ),
            threshold=(None if payload.get("threshold") is None else float(payload["threshold"])),
            impurity_decrease=float(payload.get("impurity_decrease", 0.0)),
            left=(
                None
                if left_payload is None
                else CARTClassifierScratch._node_from_dict(left_payload)  # type: ignore[arg-type]
            ),
            right=(
                None
                if right_payload is None
                else CARTClassifierScratch._node_from_dict(right_payload)  # type: ignore[arg-type]
            ),
        )

    def save_json(self, path: str | Path) -> None:
        self._require_fitted()
        assert self.tree_ is not None
        payload = {
            "metadata": self.metadata(),
            "tree": self._node_to_dict(self.tree_),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "CARTClassifierScratch":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        config_payload = dict(metadata["config"])
        class_weight = config_payload.get("class_weight")
        if isinstance(class_weight, dict):
            config_payload["class_weight"] = {
                int(key): float(value) for key, value in class_weight.items()
            }
        model = cls(CARTConfig(**config_payload))
        model.tree_ = cls._node_from_dict(payload["tree"])
        model.class_weight_ = {
            int(key): float(value)
            for key, value in metadata["class_weight_resolved"].items()
        }
        model.n_features_in_ = int(metadata["n_features_in"])
        model.n_nodes_ = int(metadata["n_nodes"])
        model.n_leaves_ = int(metadata["n_leaves"])
        model.max_depth_reached_ = int(metadata["max_depth_reached"])
        model.feature_importances_ = np.asarray(
            metadata["feature_importances"], dtype=np.float64
        )
        model.is_fitted_ = True
        return model
