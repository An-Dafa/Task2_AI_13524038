from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "data" / "source_archives"
RAW = ROOT / "data" / "raw"
EXTERNAL = ROOT / "data" / "external"
RAW.mkdir(parents=True, exist_ok=True)
EXTERNAL.mkdir(parents=True, exist_ok=True)

competition_zip = ARCHIVES / "ai-lab-recruitment-task-2.zip"
external_zip = ARCHIVES / "archive.zip"

if competition_zip.exists():
    with zipfile.ZipFile(competition_zip) as zf:
        for name in ["train.csv", "test.csv", "sample_submission.csv"]:
            with zf.open(name) as src, (RAW / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    print("Competition CSV extracted to data/raw/.")
else:
    print("Competition archive not found:", competition_zip)

if external_zip.exists():
    with zipfile.ZipFile(external_zip) as zf:
        candidate = next((name for name in zf.namelist() if name.endswith("loan_data.csv")), None)
        if candidate is None:
            raise FileNotFoundError("loan_data.csv not found inside archive.zip")
        with zf.open(candidate) as src, (EXTERNAL / "loan_data.csv").open("wb") as dst:
            shutil.copyfileobj(src, dst)
    print("External loan_data.csv extracted to data/external/.")
else:
    print("External archive not found:", external_zip)
