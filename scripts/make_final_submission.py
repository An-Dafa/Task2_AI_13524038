from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dtl_lr_svm.external_data import load_and_prepare, TARGET, ID_COL
from src.dtl_lr_svm.optimized_dtl import (
    OneHotFeatureEncoder,
    OptimizedDTLConfig,
    OptimizedEntropyTree,
)

RAW = ROOT / "data" / "raw"
EXTERNAL = ROOT / "data" / "external" / "loan_data.csv"
OUT = ROOT / "outputs" / "submissions"
OUT.mkdir(parents=True, exist_ok=True)

FINAL_THRESHOLD = 0.7067307692307693

prepared = load_and_prepare(
    RAW / "train.csv",
    RAW / "test.csv",
    RAW / "sample_submission.csv",
    EXTERNAL,
)

feature_names = [
    "person_id",
    "person_age",
    "person_gender",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
    "person_education",
    "loan_intent",
]

full_train = pd.concat(
    [prepared.train, prepared.external_unused],
    ignore_index=True,
    sort=False,
)
source_is_external = np.r_[
    np.zeros(len(prepared.train), dtype=bool),
    np.ones(len(prepared.external_unused), dtype=bool),
]

encoder = OneHotFeatureEncoder(feature_names)
X_train = encoder.fit_transform(full_train)
X_test = encoder.transform(prepared.test)
y_train = full_train[TARGET].to_numpy(dtype=np.int64)

model = OptimizedEntropyTree(
    OptimizedDTLConfig(
        max_depth=12,
        min_samples_leaf=10,
        min_samples_split=20,
        positive_weight_multiplier=0.84,
        external_weight=1.10,
        max_thresholds_per_feature=512,
        exact_feature_names=("person_id",),
    )
)
model.fit(
    X_train,
    y_train,
    feature_names=encoder.feature_names_,
    source_is_external=source_is_external,
)

prediction = model.predict(X_test, threshold=FINAL_THRESHOLD)
submission = pd.DataFrame(
    {ID_COL: prepared.test[ID_COL].astype(int), TARGET: prediction.astype(int)}
)
assert submission[ID_COL].equals(prepared.sample_submission[ID_COL])
assert set(submission[TARGET].unique()).issubset({0, 1})

output_path = OUT / "submission_final_optimized_dtl.csv"
submission.to_csv(output_path, index=False)
print("Saved:", output_path)
print("Predicted positives:", int(prediction.sum()))
print("Tree nodes:", model.n_nodes_, "leaves:", model.n_leaves_)

split_counts = (
    pd.Series(model.split_counts(), name="split_count")
    .rename_axis("feature")
    .sort_values(ascending=False)
)
split_counts.to_csv(ROOT / "outputs" / "metrics" / "final_tree_split_counts_regenerated.csv")
