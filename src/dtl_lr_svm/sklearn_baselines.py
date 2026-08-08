"""Scikit-learn baselines for Logistic Regression, CART, and linear SVM.

These models are comparison baselines only.  The competition submission must
later use the from-scratch implementation required by the task specification.

The module deliberately reuses :class:`LoanPreprocessor` so preprocessing is
identical in spirit to the later manual models and is fitted only on each
training split/fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .metrics import binary_classification_metrics, threshold_search
from .preprocessing import make_linear_preprocessor, make_tree_preprocessor
from .validation import make_stratified_folds, make_stratified_holdout


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    family: str
    params: dict[str, Any]


DEFAULT_CONFIGS: tuple[BaselineConfig, ...] = (
    BaselineConfig(
        name="lr_default",
        family="logistic_regression",
        params={"C": 1.0, "class_weight": None, "max_iter": 3000},
    ),
    BaselineConfig(
        name="lr_balanced",
        family="logistic_regression",
        params={"C": 1.0, "class_weight": "balanced", "max_iter": 3000},
    ),
    BaselineConfig(
        name="cart_depth_5",
        family="cart",
        params={
            "criterion": "gini",
            "max_depth": 5,
            "min_samples_leaf": 10,
            "class_weight": None,
        },
    ),
    BaselineConfig(
        name="cart_depth_8_balanced",
        family="cart",
        params={
            "criterion": "gini",
            "max_depth": 8,
            "min_samples_leaf": 10,
            "class_weight": "balanced",
        },
    ),
    BaselineConfig(
        name="linear_svm_default",
        family="linear_svm",
        params={"C": 1.0, "class_weight": None, "max_iter": 10000},
    ),
    BaselineConfig(
        name="linear_svm_balanced",
        family="linear_svm",
        params={"C": 1.0, "class_weight": "balanced", "max_iter": 10000},
    ),
)


def _build_model(config: BaselineConfig, random_state: int):
    if config.family == "logistic_regression":
        return LogisticRegression(
            solver="lbfgs",
            random_state=random_state,
            **config.params,
        )
    if config.family == "cart":
        return DecisionTreeClassifier(
            random_state=random_state,
            **config.params,
        )
    if config.family == "linear_svm":
        return LinearSVC(
            dual="auto",
            random_state=random_state,
            **config.params,
        )
    raise ValueError(f"Keluarga model tidak dikenal: {config.family}")


def _build_preprocessor(config: BaselineConfig):
    if config.family == "cart":
        return make_tree_preprocessor()
    return make_linear_preprocessor()


def _scores_and_default_threshold(model, X: np.ndarray, family: str) -> tuple[np.ndarray, float]:
    if family in {"logistic_regression", "cart"}:
        scores = model.predict_proba(X)[:, 1]
        return np.asarray(scores, dtype=float), 0.5
    if family == "linear_svm":
        scores = model.decision_function(X)
        return np.asarray(scores, dtype=float), 0.0
    raise ValueError(f"Keluarga model tidak dikenal: {family}")


def _threshold_grid(family: str) -> np.ndarray:
    if family in {"logistic_regression", "cart"}:
        return np.linspace(0.05, 0.95, 181)
    return np.linspace(-2.0, 2.0, 321)


def _flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            result[f"{prefix}{key}"] = json.dumps(value)
        else:
            result[f"{prefix}{key}"] = value
    return result


def evaluate_holdout(
    train: pd.DataFrame,
    configs: Iterable[BaselineConfig] = DEFAULT_CONFIGS,
    *,
    random_state: int = 42,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate candidate baselines on one reproducible stratified holdout."""
    split = make_stratified_holdout(
        train,
        test_size=test_size,
        random_state=random_state,
    )
    rows: list[dict[str, Any]] = []
    threshold_rows: list[pd.DataFrame] = []

    for config in configs:
        preprocessor = _build_preprocessor(config)
        X_train = preprocessor.fit_transform(split.X_train)
        X_valid = preprocessor.transform(split.X_valid)
        model = _build_model(config, random_state)

        start = perf_counter()
        model.fit(X_train, split.y_train)
        train_seconds = perf_counter() - start

        start = perf_counter()
        scores, default_threshold = _scores_and_default_threshold(
            model, X_valid, config.family
        )
        inference_seconds = perf_counter() - start

        default_pred = (scores >= default_threshold).astype(np.int64)
        default_metrics = binary_classification_metrics(split.y_valid, default_pred)

        search = threshold_search(
            split.y_valid,
            scores,
            thresholds=_threshold_grid(config.family),
        )
        best = search.iloc[0]
        tuned_pred = (scores >= float(best["threshold"])).astype(np.int64)
        tuned_metrics = binary_classification_metrics(split.y_valid, tuned_pred)

        search.insert(0, "model", config.name)
        search.insert(1, "family", config.family)
        threshold_rows.append(search)

        rows.append(
            {
                "model": config.name,
                "family": config.family,
                "params": json.dumps(config.params, sort_keys=True),
                "n_features": int(X_train.shape[1]),
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
                "default_threshold": default_threshold,
                "best_holdout_threshold": float(best["threshold"]),
                **_flatten_metrics(default_metrics, "default_"),
                **_flatten_metrics(tuned_metrics, "tuned_"),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["tuned_macro_f1", "default_macro_f1"], ascending=False
    )
    thresholds = pd.concat(threshold_rows, ignore_index=True)
    return results.reset_index(drop=True), thresholds


def evaluate_cross_validation(
    train: pd.DataFrame,
    configs: Iterable[BaselineConfig] = DEFAULT_CONFIGS,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run leakage-safe stratified CV and collect out-of-fold predictions.

    Threshold tuning is performed once on all out-of-fold scores.  Default
    threshold metrics remain the primary non-tuned comparison; OOF threshold
    metrics are reported as an exploratory improvement for later experiments.
    """
    y = train["loan_status"].to_numpy(dtype=np.int64)
    X = train.drop(columns=["loan_status"])
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []

    for config in configs:
        oof_scores = np.full(len(train), np.nan, dtype=float)
        fit_seconds_total = 0.0
        inference_seconds_total = 0.0

        for fold, train_idx, valid_idx in make_stratified_folds(
            y,
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        ):
            X_train_raw = X.iloc[train_idx].copy()
            X_valid_raw = X.iloc[valid_idx].copy()
            y_train = y[train_idx]
            y_valid = y[valid_idx]

            preprocessor = _build_preprocessor(config)
            X_train = preprocessor.fit_transform(X_train_raw)
            X_valid = preprocessor.transform(X_valid_raw)
            model = _build_model(config, random_state + fold)

            start = perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = perf_counter() - start
            fit_seconds_total += fit_seconds

            start = perf_counter()
            scores, default_threshold = _scores_and_default_threshold(
                model, X_valid, config.family
            )
            inference_seconds = perf_counter() - start
            inference_seconds_total += inference_seconds
            oof_scores[valid_idx] = scores

            pred = (scores >= default_threshold).astype(np.int64)
            metrics = binary_classification_metrics(y_valid, pred)
            fold_rows.append(
                {
                    "model": config.name,
                    "family": config.family,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_valid": len(valid_idx),
                    "n_features": int(X_train.shape[1]),
                    "train_seconds": fit_seconds,
                    "inference_seconds": inference_seconds,
                    **_flatten_metrics(metrics),
                }
            )

        if not np.isfinite(oof_scores).all():
            raise RuntimeError(f"OOF score tidak lengkap untuk {config.name}.")

        _, default_threshold = _scores_and_default_threshold(
            _build_model(config, random_state), np.zeros((1, 1)), config.family
        ) if False else (None, 0.5 if config.family != "linear_svm" else 0.0)
        default_pred = (oof_scores >= default_threshold).astype(np.int64)
        default_metrics = binary_classification_metrics(y, default_pred)

        search = threshold_search(y, oof_scores, _threshold_grid(config.family))
        best = search.iloc[0]
        best_threshold = float(best["threshold"])
        tuned_pred = (oof_scores >= best_threshold).astype(np.int64)
        tuned_metrics = binary_classification_metrics(y, tuned_pred)

        model_fold_rows = pd.DataFrame(
            [row for row in fold_rows if row["model"] == config.name]
        )
        summary_rows.append(
            {
                "model": config.name,
                "family": config.family,
                "params": json.dumps(config.params, sort_keys=True),
                "n_splits": n_splits,
                "fold_macro_f1_mean": float(model_fold_rows["macro_f1"].mean()),
                "fold_macro_f1_std": float(model_fold_rows["macro_f1"].std(ddof=1)),
                "fold_f1_0_mean": float(model_fold_rows["f1_0"].mean()),
                "fold_f1_1_mean": float(model_fold_rows["f1_1"].mean()),
                "fit_seconds_total": fit_seconds_total,
                "inference_seconds_total": inference_seconds_total,
                "default_threshold": default_threshold,
                "best_oof_threshold": best_threshold,
                **_flatten_metrics(default_metrics, "oof_default_"),
                **_flatten_metrics(tuned_metrics, "oof_tuned_"),
            }
        )

        oof_rows.append(
            pd.DataFrame(
                {
                    "row_index": np.arange(len(train)),
                    "person_id": train["person_id"].to_numpy(),
                    "y_true": y,
                    "score": oof_scores,
                    "pred_default": default_pred,
                    "pred_tuned": tuned_pred,
                    "model": config.name,
                    "family": config.family,
                }
            )
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["oof_tuned_macro_f1", "oof_default_macro_f1"], ascending=False
    )
    folds = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_rows, ignore_index=True)
    return summary.reset_index(drop=True), folds, oof


def save_confusion_matrix_plot(
    matrix: list[list[int]],
    title: str,
    path: Path,
) -> None:
    values = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(values)
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(values[row, col]), ha="center", va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_and_save_baselines(
    train_path: Path,
    output_metrics_dir: Path,
    output_figures_dir: Path,
    *,
    mode: str = "quick",
    random_state: int = 42,
) -> dict[str, Any]:
    """Run quick holdout or full 5-fold baselines and save all artifacts."""
    train = pd.read_csv(train_path)
    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    if mode == "quick":
        results, thresholds = evaluate_holdout(train, random_state=random_state)
        results.to_csv(output_metrics_dir / "sklearn_baseline_holdout.csv", index=False)
        thresholds.to_csv(
            output_metrics_dir / "sklearn_baseline_threshold_search.csv", index=False
        )
        best = results.iloc[0].to_dict()
        matrix = json.loads(best["tuned_confusion_matrix"])
        save_confusion_matrix_plot(
            matrix,
            f"Best holdout baseline: {best['model']}",
            output_figures_dir / "sklearn_baseline_best_confusion_matrix.png",
        )
        summary = {
            "mode": mode,
            "random_state": random_state,
            "best_model": best["model"],
            "best_family": best["family"],
            "best_macro_f1": float(best["tuned_macro_f1"]),
            "best_threshold": float(best["best_holdout_threshold"]),
            "results_file": "sklearn_baseline_holdout.csv",
            "threshold_file": "sklearn_baseline_threshold_search.csv",
        }
    elif mode == "full":
        summary_df, folds, oof = evaluate_cross_validation(
            train, n_splits=5, random_state=random_state
        )
        summary_df.to_csv(output_metrics_dir / "sklearn_baseline_cv_summary.csv", index=False)
        folds.to_csv(output_metrics_dir / "sklearn_baseline_cv_folds.csv", index=False)
        oof.to_csv(output_metrics_dir / "sklearn_baseline_oof_predictions.csv", index=False)
        best = summary_df.iloc[0].to_dict()
        matrix = json.loads(best["oof_tuned_confusion_matrix"])
        save_confusion_matrix_plot(
            matrix,
            f"Best OOF baseline: {best['model']}",
            output_figures_dir / "sklearn_baseline_best_oof_confusion_matrix.png",
        )
        summary = {
            "mode": mode,
            "random_state": random_state,
            "n_splits": 5,
            "best_model": best["model"],
            "best_family": best["family"],
            "best_macro_f1": float(best["oof_tuned_macro_f1"]),
            "best_threshold": float(best["best_oof_threshold"]),
            "results_file": "sklearn_baseline_cv_summary.csv",
            "folds_file": "sklearn_baseline_cv_folds.csv",
            "oof_file": "sklearn_baseline_oof_predictions.csv",
        }
    else:
        raise ValueError("mode harus 'quick' atau 'full'.")

    summary_path = output_metrics_dir / "sklearn_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
