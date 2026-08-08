"""Leakage-safe preprocessing for the loan approval dataset.

The transformer is deliberately implemented with pandas and NumPy so it can be
used by the manual Logistic Regression, CART, and SVM implementations later.
It learns category levels, clipping bounds, means, and standard deviations only
from the data passed to :meth:`fit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TARGET = "loan_status"
ID_COL = "person_id"

BASE_NUMERIC_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

CATEGORICAL_COLUMNS = [
    "person_gender",
    "person_home_ownership",
    "previous_loan_defaults_on_file",
]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for :class:`LoanPreprocessor`."""

    standardize: bool = True
    drop_first: bool = True
    clip_quantiles: tuple[float, float] | None = None
    log1p_columns: tuple[str, ...] = ()
    add_engineered_features: bool = False


class LoanPreprocessor:
    """Fit/transform preprocessing without target or test leakage."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self.category_levels_: dict[str, list[str]] = {}
        self.baseline_categories_: dict[str, str | None] = {}
        self.clip_bounds_: dict[str, tuple[float, float]] = {}
        self.numeric_feature_names_: list[str] = []
        self.encoded_feature_names_: list[str] = []
        self.feature_names_: list[str] = []
        self.means_: pd.Series | None = None
        self.scales_: pd.Series | None = None
        self.is_fitted_: bool = False

    @staticmethod
    def _required_columns() -> list[str]:
        return BASE_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

    def _validate_input(self, frame: pd.DataFrame, *, fitting: bool) -> None:
        missing = [column for column in self._required_columns() if column not in frame.columns]
        if missing:
            raise ValueError(f"Kolom fitur tidak lengkap: {missing}")
        if TARGET in frame.columns:
            raise ValueError(
                f"Kolom target '{TARGET}' tidak boleh diberikan ke preprocessor. "
                "Pisahkan X dan y terlebih dahulu."
            )
        if frame.empty:
            raise ValueError("Data fitur tidak boleh kosong.")
        if fitting and ID_COL in frame.columns and frame[ID_COL].duplicated().any():
            raise ValueError("person_id duplikat ditemukan saat fit preprocessing.")

    @staticmethod
    def _engineered_features(numeric: pd.DataFrame) -> pd.DataFrame:
        engineered = pd.DataFrame(index=numeric.index)
        engineered["income_per_emp_year"] = numeric["person_income"] / (
            numeric["person_emp_exp"] + 1.0
        )
        engineered["loan_to_credit_score"] = numeric["loan_amnt"] / (
            numeric["credit_score"] + 1.0
        )
        engineered["age_minus_credit_history"] = (
            numeric["person_age"] - numeric["cb_person_cred_hist_length"]
        )
        available_work_years = (numeric["person_age"] - 14.0).clip(lower=1.0)
        engineered["employment_age_ratio"] = numeric["person_emp_exp"] / available_work_years
        engineered["interest_loan_interaction"] = (
            numeric["loan_int_rate"] * numeric["loan_amnt"]
        )
        return engineered

    def _build_unscaled_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame[BASE_NUMERIC_COLUMNS].astype(float).copy()

        for column, (lower, upper) in self.clip_bounds_.items():
            numeric[column] = numeric[column].clip(lower=lower, upper=upper)

        for column in self.config.log1p_columns:
            if column not in numeric.columns:
                raise ValueError(f"Kolom log1p tidak dikenal: {column}")
            if (numeric[column] < 0).any():
                raise ValueError(f"Kolom {column} memiliki nilai negatif; log1p tidak aman.")
            numeric[f"log1p_{column}"] = np.log1p(numeric[column])

        if self.config.add_engineered_features:
            numeric = pd.concat([numeric, self._engineered_features(numeric)], axis=1)

        encoded = pd.DataFrame(index=frame.index)
        for column in CATEGORICAL_COLUMNS:
            values = frame[column].astype(str)
            levels = self.category_levels_[column]
            baseline = self.baseline_categories_[column]
            for level in levels:
                if self.config.drop_first and level == baseline:
                    continue
                encoded[f"{column}__{level}"] = (values == level).astype(float)

        return pd.concat([numeric, encoded], axis=1)

    def fit(self, frame: pd.DataFrame) -> "LoanPreprocessor":
        self._validate_input(frame, fitting=True)

        lower_upper = self.config.clip_quantiles
        if lower_upper is not None:
            lower_q, upper_q = lower_upper
            if not (0.0 <= lower_q < upper_q <= 1.0):
                raise ValueError("clip_quantiles harus memenuhi 0 <= lower < upper <= 1.")
            self.clip_bounds_ = {
                column: (
                    float(frame[column].quantile(lower_q)),
                    float(frame[column].quantile(upper_q)),
                )
                for column in BASE_NUMERIC_COLUMNS
            }
        else:
            self.clip_bounds_ = {}

        self.category_levels_ = {
            column: sorted(frame[column].astype(str).unique().tolist())
            for column in CATEGORICAL_COLUMNS
        }
        self.baseline_categories_ = {
            column: (levels[0] if self.config.drop_first else None)
            for column, levels in self.category_levels_.items()
        }

        unscaled = self._build_unscaled_frame(frame)
        self.numeric_feature_names_ = [
            column
            for column in unscaled.columns
            if not any(column.startswith(f"{cat}__") for cat in CATEGORICAL_COLUMNS)
        ]
        self.encoded_feature_names_ = [
            column for column in unscaled.columns if column not in self.numeric_feature_names_
        ]
        self.feature_names_ = list(unscaled.columns)

        if self.config.standardize:
            self.means_ = unscaled[self.numeric_feature_names_].mean()
            scales = unscaled[self.numeric_feature_names_].std(ddof=0)
            self.scales_ = scales.mask(scales == 0.0, 1.0)
        else:
            self.means_ = pd.Series(0.0, index=self.numeric_feature_names_)
            self.scales_ = pd.Series(1.0, index=self.numeric_feature_names_)

        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("LoanPreprocessor harus di-fit sebelum transform.")
        self._validate_input(frame, fitting=False)
        transformed = self._build_unscaled_frame(frame)
        transformed = transformed.reindex(columns=self.feature_names_, fill_value=0.0)

        if self.config.standardize:
            assert self.means_ is not None and self.scales_ is not None
            transformed.loc[:, self.numeric_feature_names_] = (
                transformed[self.numeric_feature_names_] - self.means_
            ) / self.scales_

        matrix = transformed.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(matrix).all():
            raise ValueError("Hasil preprocessing mengandung NaN atau nilai tak hingga.")
        return matrix

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def get_feature_names_out(self) -> list[str]:
        if not self.is_fitted_:
            raise RuntimeError("LoanPreprocessor belum di-fit.")
        return list(self.feature_names_)

    def unknown_category_report(self, frame: pd.DataFrame) -> dict[str, dict[str, int]]:
        if not self.is_fitted_:
            raise RuntimeError("LoanPreprocessor belum di-fit.")
        report: dict[str, dict[str, int]] = {}
        for column in CATEGORICAL_COLUMNS:
            known = set(self.category_levels_[column])
            counts = frame[column].astype(str).value_counts()
            unknown = {str(level): int(count) for level, count in counts.items() if level not in known}
            report[column] = unknown
        return report

    def metadata(self) -> dict[str, object]:
        if not self.is_fitted_:
            raise RuntimeError("LoanPreprocessor belum di-fit.")
        assert self.means_ is not None and self.scales_ is not None
        return {
            "config": {
                "standardize": self.config.standardize,
                "drop_first": self.config.drop_first,
                "clip_quantiles": list(self.config.clip_quantiles)
                if self.config.clip_quantiles is not None
                else None,
                "log1p_columns": list(self.config.log1p_columns),
                "add_engineered_features": self.config.add_engineered_features,
            },
            "category_levels": self.category_levels_,
            "baseline_categories": self.baseline_categories_,
            "clip_bounds": {
                column: [float(lower), float(upper)]
                for column, (lower, upper) in self.clip_bounds_.items()
            },
            "numeric_feature_names": self.numeric_feature_names_,
            "encoded_feature_names": self.encoded_feature_names_,
            "feature_names": self.feature_names_,
            "means": {key: float(value) for key, value in self.means_.items()},
            "scales": {key: float(value) for key, value in self.scales_.items()},
        }


def make_linear_preprocessor(
    *,
    clip_quantiles: tuple[float, float] | None = None,
    log1p_columns: Iterable[str] = (),
    add_engineered_features: bool = False,
) -> LoanPreprocessor:
    """Default preprocessing for Logistic Regression and linear SVM."""
    return LoanPreprocessor(
        PreprocessingConfig(
            standardize=True,
            drop_first=True,
            clip_quantiles=clip_quantiles,
            log1p_columns=tuple(log1p_columns),
            add_engineered_features=add_engineered_features,
        )
    )


def make_tree_preprocessor(
    *,
    clip_quantiles: tuple[float, float] | None = None,
    add_engineered_features: bool = False,
) -> LoanPreprocessor:
    """Default preprocessing for a manual CART classifier."""
    return LoanPreprocessor(
        PreprocessingConfig(
            standardize=False,
            drop_first=False,
            clip_quantiles=clip_quantiles,
            add_engineered_features=add_engineered_features,
        )
    )
