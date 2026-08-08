from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXTERNAL = ROOT / "data" / "external" / "loan_data.csv"

required = [RAW / "train.csv", RAW / "test.csv", RAW / "sample_submission.csv"]
missing = [path for path in required if not path.exists()]

print("Repository:", ROOT)
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)

if missing:
    print("Missing competition files:")
    for path in missing:
        print(" -", path.relative_to(ROOT))
else:
    train = pd.read_csv(required[0])
    test = pd.read_csv(required[1])
    sample = pd.read_csv(required[2])
    print("Train:", train.shape, "Test:", test.shape)
    assert "loan_status" in train.columns and "loan_status" not in test.columns
    assert sample["person_id"].equals(test["person_id"])
    print("Competition dataset: OK")

print("External source:", "FOUND" if EXTERNAL.exists() else "NOT FOUND")
print("See data/README.md for placement instructions.")
