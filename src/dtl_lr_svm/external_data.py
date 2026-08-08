"""Safe preparation of the allowed external source dataset.

Important integrity rule: the original public dataset contains labels for rows that
correspond to the competition test set.  This module never merges or reads those
test labels for model training/evaluation.  It only recovers the two extra features
and adds the 9,000 source rows that are absent from both competition train and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "loan_status"
ID_COL = "person_id"
EXTRA_FEATURES = ["person_education", "loan_intent"]


@dataclass
class PreparedCompetitionData:
    train: pd.DataFrame
    test: pd.DataFrame
    external_unused: pd.DataFrame
    sample_submission: pd.DataFrame


def load_and_prepare(
    train_path: str | Path,
    test_path: str | Path,
    sample_submission_path: str | Path,
    external_path: str | Path,
) -> PreparedCompetitionData:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_submission_path)
    source = pd.read_csv(external_path).copy()

    if ID_COL in source.columns:
        raise ValueError("Dataset sumber diharapkan belum memiliki person_id.")
    source.insert(0, ID_COL, np.arange(1, len(source) + 1, dtype=np.int64))

    assert TARGET in train.columns and TARGET not in test.columns
    assert sample[ID_COL].equals(test[ID_COL])

    shared = [column for column in test.columns if column != ID_COL]
    source_by_id = source.set_index(ID_COL)

    # Verify that person_id really maps to the corresponding source row using
    # shared features only.  The source target is deliberately not touched here.
    for column in shared:
        source_values = source_by_id.loc[test[ID_COL], column].reset_index(drop=True)
        test_values = test[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(test_values):
            if not np.allclose(
                test_values.to_numpy(dtype=float),
                source_values.to_numpy(dtype=float),
            ):
                raise ValueError(f"Mapping person_id tidak cocok pada {column}.")
        elif not test_values.astype(str).equals(source_values.astype(str)):
            raise ValueError(f"Mapping person_id tidak cocok pada {column}.")

    # Feature-only lookup: TARGET is intentionally excluded.
    feature_lookup = source[[ID_COL, *EXTRA_FEATURES]].copy()
    train_enriched = train.merge(
        feature_lookup, on=ID_COL, how="left", validate="one_to_one"
    )
    test_enriched = test.merge(
        feature_lookup, on=ID_COL, how="left", validate="one_to_one"
    )

    train_ids = set(train[ID_COL].astype(int))
    test_ids = set(test[ID_COL].astype(int))
    external_unused = source[
        ~source[ID_COL].isin(train_ids | test_ids)
    ].copy()

    if len(external_unused) != 9000:
        raise ValueError(
            f"Expected 9,000 unused source rows, found {len(external_unused)}."
        )
    if not set(external_unused[ID_COL]).isdisjoint(test_ids):
        raise RuntimeError("Competition test row leaked into external training rows.")

    return PreparedCompetitionData(
        train=train_enriched,
        test=test_enriched,
        external_unused=external_unused,
        sample_submission=sample,
    )
