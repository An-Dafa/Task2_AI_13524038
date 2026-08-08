"""NumPy-only linear Support Vector Machine for binary classification.

The estimator implements a primal linear SVM trained with (mini-)batch
subgradient descent.  The optimized objective is weighted hinge loss plus L2
regularization.  This module deliberately does not import scikit-learn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping

import numpy as np

ClassWeight = Literal["balanced"] | Mapping[int, float] | None
LearningRateSchedule = Literal["constant", "inverse_time", "sqrt_decay"]
EarlyStoppingMonitor = Literal["train_loss", "val_loss"]
Initialization = Literal["zeros", "normal"]


@dataclass(frozen=True)
class LinearSVMConfig:
    """Hyperparameters for :class:`LinearSVMScratch`."""

    learning_rate: float = 0.03
    max_epochs: int = 1000
    regularization: float = 1e-4
    fit_intercept: bool = True
    class_weight: ClassWeight = None
    batch_size: int | None = None
    shuffle: bool = True
    random_state: int = 42
    tolerance: float = 1e-7
    patience: int | None = 50
    eval_interval: int = 5
    learning_rate_schedule: LearningRateSchedule = "inverse_time"
    decay: float = 0.002
    early_stopping_monitor: EarlyStoppingMonitor = "train_loss"
    initialization: Initialization = "zeros"
    gradient_clip_norm: float | None = 10.0
    verbose: int = 0


class LinearSVMScratch:
    """Binary linear SVM trained with primal subgradient descent.

    Labels ``0`` and ``1`` are internally mapped to ``-1`` and ``+1``.  The
    objective is::

        0.5 * regularization * ||w||^2
        + weighted_mean(max(0, 1 - y * (Xw + b)))

    The intercept is not regularized.  ``class_weight='balanced'`` resolves to
    ``n_samples / (2 * class_count)`` for each class.
    """

    def __init__(self, config: LinearSVMConfig | None = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Gunakan config atau keyword arguments, bukan keduanya.")
        self.config = config or LinearSVMConfig(**kwargs)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.classes_: np.ndarray = np.array([0, 1], dtype=np.int64)
        self.class_weight_: dict[int, float] = {0: 1.0, 1: 1.0}
        self.history_: list[dict[str, float | int]] = []
        self.n_features_in_: int | None = None
        self.n_epochs_: int = 0
        self.n_updates_: int = 0
        self.best_epoch_: int = 0
        self.best_loss_: float = np.inf
        self.stopped_early_: bool = False
        self.stop_reason_: str = "not_fitted"
        self.is_fitted_: bool = False
        self._validate_config()

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.learning_rate <= 0:
            raise ValueError("learning_rate harus lebih besar dari 0.")
        if cfg.max_epochs < 1:
            raise ValueError("max_epochs minimal 1.")
        if cfg.regularization < 0:
            raise ValueError("regularization tidak boleh negatif.")
        if cfg.batch_size is not None and cfg.batch_size < 1:
            raise ValueError("batch_size harus None atau minimal 1.")
        if cfg.tolerance < 0:
            raise ValueError("tolerance tidak boleh negatif.")
        if cfg.patience is not None and cfg.patience < 1:
            raise ValueError("patience harus None atau minimal 1.")
        if cfg.eval_interval < 1:
            raise ValueError("eval_interval minimal 1.")
        if cfg.learning_rate_schedule not in {"constant", "inverse_time", "sqrt_decay"}:
            raise ValueError("learning_rate_schedule tidak dikenal.")
        if cfg.decay < 0:
            raise ValueError("decay tidak boleh negatif.")
        if cfg.early_stopping_monitor not in {"train_loss", "val_loss"}:
            raise ValueError("early_stopping_monitor harus train_loss atau val_loss.")
        if cfg.initialization not in {"zeros", "normal"}:
            raise ValueError("initialization harus zeros atau normal.")
        if cfg.gradient_clip_norm is not None and cfg.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm harus None atau lebih besar dari 0.")
        if isinstance(cfg.class_weight, str) and cfg.class_weight != "balanced":
            raise ValueError("class_weight string hanya boleh 'balanced'.")

    @staticmethod
    def _validate_X(X: np.ndarray, *, expected_features: int | None = None) -> np.ndarray:
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
    def _validate_y(
        y: np.ndarray,
        n_samples: int,
        *,
        require_both_classes: bool = True,
    ) -> np.ndarray:
        labels = np.asarray(y, dtype=np.int64).reshape(-1)
        if labels.shape[0] != n_samples:
            raise ValueError("Panjang y harus sama dengan jumlah baris X.")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Linear SVM ini hanya menerima label biner 0/1.")
        if require_both_classes and np.unique(labels).size < 2:
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

    def _sample_weights(self, labels_01: np.ndarray) -> np.ndarray:
        return np.where(
            labels_01 == 1,
            self.class_weight_[1],
            self.class_weight_[0],
        ).astype(np.float64)

    def _learning_rate_at(self, epoch: int) -> float:
        cfg = self.config
        if cfg.learning_rate_schedule == "constant":
            return float(cfg.learning_rate)
        if cfg.learning_rate_schedule == "inverse_time":
            return float(cfg.learning_rate / (1.0 + cfg.decay * (epoch - 1)))
        return float(cfg.learning_rate / np.sqrt(1.0 + cfg.decay * (epoch - 1)))

    def _loss_and_margin_stats(
        self,
        X: np.ndarray,
        labels_01: np.ndarray,
        coef: np.ndarray,
        intercept: float,
    ) -> tuple[float, float, float, float]:
        y_signed = np.where(labels_01 == 1, 1.0, -1.0)
        scores = X @ coef + intercept
        signed_margins = y_signed * scores
        hinge = np.maximum(0.0, 1.0 - signed_margins)
        weights = self._sample_weights(labels_01)
        weighted_hinge = float(np.sum(weights * hinge) / np.sum(weights))
        regularization = 0.5 * self.config.regularization * float(np.dot(coef, coef))
        support_fraction = float(np.mean(signed_margins <= 1.0))
        misclassified_fraction = float(np.mean(signed_margins <= 0.0))
        mean_signed_margin = float(np.mean(signed_margins))
        return (
            weighted_hinge + regularization,
            support_fraction,
            misclassified_fraction,
            mean_signed_margin,
        )

    def _gradient(
        self,
        X: np.ndarray,
        labels_01: np.ndarray,
        coef: np.ndarray,
        intercept: float,
    ) -> tuple[np.ndarray, float]:
        y_signed = np.where(labels_01 == 1, 1.0, -1.0)
        scores = X @ coef + intercept
        active = y_signed * scores < 1.0
        weights = self._sample_weights(labels_01)
        denominator = float(np.sum(weights))

        grad_coef = self.config.regularization * coef
        grad_intercept = 0.0
        if np.any(active):
            active_factor = weights[active] * y_signed[active]
            grad_coef = grad_coef - (X[active].T @ active_factor) / denominator
            if self.config.fit_intercept:
                grad_intercept = -float(np.sum(active_factor) / denominator)

        clip = self.config.gradient_clip_norm
        if clip is not None:
            norm = float(
                np.sqrt(np.dot(grad_coef, grad_coef) + grad_intercept * grad_intercept)
            )
            if norm > clip:
                scale = clip / norm
                grad_coef = grad_coef * scale
                grad_intercept = grad_intercept * scale
        return grad_coef, grad_intercept

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> "LinearSVMScratch":
        """Fit the SVM and optionally record validation objective values."""
        features = self._validate_X(X)
        labels = self._validate_y(y, features.shape[0], require_both_classes=True)
        self.n_features_in_ = int(features.shape[1])

        valid_features: np.ndarray | None = None
        valid_labels: np.ndarray | None = None
        if (X_valid is None) != (y_valid is None):
            raise ValueError("X_valid dan y_valid harus diberikan bersamaan.")
        if X_valid is not None and y_valid is not None:
            valid_features = self._validate_X(
                X_valid, expected_features=self.n_features_in_
            )
            valid_labels = self._validate_y(
                y_valid,
                valid_features.shape[0],
                require_both_classes=False,
            )
        if self.config.early_stopping_monitor == "val_loss" and valid_features is None:
            raise ValueError("Monitor val_loss membutuhkan X_valid dan y_valid.")

        self.class_weight_ = self._resolve_class_weight(labels)
        rng = np.random.default_rng(self.config.random_state)
        if self.config.initialization == "zeros":
            coef = np.zeros(self.n_features_in_, dtype=np.float64)
        else:
            coef = rng.normal(0.0, 0.01, size=self.n_features_in_).astype(np.float64)
        intercept = 0.0

        n_samples = features.shape[0]
        batch_size = self.config.batch_size or n_samples
        batch_size = min(batch_size, n_samples)
        indices = np.arange(n_samples)

        self.history_ = []
        self.n_updates_ = 0
        self.stopped_early_ = False
        self.stop_reason_ = "max_epochs_reached"
        best_coef = coef.copy()
        best_intercept = float(intercept)
        best_loss = np.inf
        best_epoch = 0
        checks_without_improvement = 0

        for epoch in range(1, self.config.max_epochs + 1):
            learning_rate = self._learning_rate_at(epoch)
            if self.config.shuffle and batch_size < n_samples:
                rng.shuffle(indices)

            for start in range(0, n_samples, batch_size):
                batch_indices = indices[start : start + batch_size]
                grad_coef, grad_intercept = self._gradient(
                    features[batch_indices],
                    labels[batch_indices],
                    coef,
                    intercept,
                )
                coef -= learning_rate * grad_coef
                if self.config.fit_intercept:
                    intercept -= learning_rate * grad_intercept
                self.n_updates_ += 1

            should_evaluate = (
                epoch == 1
                or epoch == self.config.max_epochs
                or epoch % self.config.eval_interval == 0
            )
            if not should_evaluate:
                continue

            (
                train_loss,
                support_fraction,
                misclassified_fraction,
                mean_signed_margin,
            ) = self._loss_and_margin_stats(features, labels, coef, intercept)
            full_grad_coef, full_grad_intercept = self._gradient(
                features, labels, coef, intercept
            )
            gradient_norm = float(
                np.sqrt(
                    np.dot(full_grad_coef, full_grad_coef)
                    + full_grad_intercept * full_grad_intercept
                )
            )
            record: dict[str, float | int] = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "gradient_norm": gradient_norm,
                "support_fraction": support_fraction,
                "misclassified_fraction": misclassified_fraction,
                "mean_signed_margin": mean_signed_margin,
            }

            val_loss = np.nan
            if valid_features is not None and valid_labels is not None:
                val_loss, val_support, val_misclassified, val_mean_margin = (
                    self._loss_and_margin_stats(
                        valid_features, valid_labels, coef, intercept
                    )
                )
                record["val_loss"] = val_loss
                record["val_support_fraction"] = val_support
                record["val_misclassified_fraction"] = val_misclassified
                record["val_mean_signed_margin"] = val_mean_margin
            self.history_.append(record)

            monitored_loss = (
                val_loss
                if self.config.early_stopping_monitor == "val_loss"
                else train_loss
            )
            if monitored_loss < best_loss - self.config.tolerance:
                best_loss = float(monitored_loss)
                best_epoch = epoch
                best_coef = coef.copy()
                best_intercept = float(intercept)
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1

            if self.config.verbose and (
                len(self.history_) == 1
                or len(self.history_) % self.config.verbose == 0
            ):
                message = (
                    f"epoch={epoch} train_loss={train_loss:.8f} "
                    f"gradient_norm={gradient_norm:.6g} "
                    f"support_fraction={support_fraction:.4f}"
                )
                if np.isfinite(val_loss):
                    message += f" val_loss={val_loss:.8f}"
                print(message)

            if (
                self.config.patience is not None
                and checks_without_improvement >= self.config.patience
            ):
                self.stopped_early_ = True
                self.stop_reason_ = (
                    f"no_{self.config.early_stopping_monitor}_improvement_"
                    f"for_{self.config.patience}_checks"
                )
                break

        self.coef_ = best_coef
        self.intercept_ = best_intercept if self.config.fit_intercept else 0.0
        self.n_epochs_ = int(epoch)
        self.best_epoch_ = int(best_epoch)
        self.best_loss_ = float(best_loss)
        self.is_fitted_ = True
        return self

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_ or self.coef_ is None or self.n_features_in_ is None:
            raise RuntimeError("Model harus di-fit sebelum prediksi.")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        assert self.coef_ is not None and self.n_features_in_ is not None
        features = self._validate_X(X, expected_features=self.n_features_in_)
        return features @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        if not np.isfinite(threshold):
            raise ValueError("threshold harus berupa nilai finite.")
        return (self.decision_function(X) >= threshold).astype(np.int64)

    def get_params(self) -> dict[str, object]:
        params = asdict(self.config)
        if isinstance(params.get("class_weight"), Mapping):
            params["class_weight"] = {
                str(key): float(value) for key, value in params["class_weight"].items()
            }
        return params

    def metadata(self) -> dict[str, object]:
        self._check_is_fitted()
        return {
            "config": self.get_params(),
            "n_features_in": self.n_features_in_,
            "class_weight_resolved": {
                str(key): float(value) for key, value in self.class_weight_.items()
            },
            "n_epochs": self.n_epochs_,
            "n_updates": self.n_updates_,
            "best_epoch": self.best_epoch_,
            "best_loss": self.best_loss_,
            "stopped_early": self.stopped_early_,
            "stop_reason": self.stop_reason_,
            "intercept": float(self.intercept_),
        }
