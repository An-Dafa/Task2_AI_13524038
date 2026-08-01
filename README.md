# Task 2 Seleksi Laboratorium Inteligensi Buatan

Repository untuk Task #2 Seleksi Laboratorium Inteligensi Buatan: Local Search serta implementasi Decision Tree Learning, Logistic Regression, dan Support Vector Machine dari scratch.

> Ganti nama folder/repository `Task2_AI_NIM` menjadi `Task2_AI_<NIM_KAMU>`.

## Struktur

```text
Task2_AI_NIM/
├── data/
│   ├── raw/                 # train.csv, test.csv, sample_submission.csv
│   └── processed/           # data hasil preprocessing (generated)
├── src/
│   ├── local_search/        # PoC Local Search
│   └── dtl_lr_svm/          # implementasi DTL, LR, SVM
├── notebooks/
│   ├── local_search/        # eksperimen Local Search
│   └── dtl_lr_svm/          # EDA dan eksperimen model
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── submissions/
├── docs/                    # PDF gabungan spesifikasi dan write-up
├── scripts/
├── requirements.txt
└── README.md
```

## Setup Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/verify_setup.py
python -m ipykernel install --user --name task2-ai --display-name "Python (Task2 AI)"
jupyter lab
```

Jika PowerShell menolak aktivasi environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Setup Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/verify_setup.py
python -m ipykernel install --user --name task2-ai --display-name "Python (Task2 AI)"
jupyter lab
```

## Aturan penting kompetisi

- Implementasi manual: CART/ID3/C4.5, Logistic Regression, dan SVM.
- NumPy boleh dipakai untuk komputasi matematis.
- scikit-learn hanya untuk baseline/pembanding dan utilitas validasi.
- Submission Kaggle harus berasal dari model manual.
- Tidak menggunakan ensemble atau algoritma di luar DTL, LR, dan SVM.
- Metrik utama: macro F1-score.

## Verifikasi dataset

Jalankan:

```bash
python scripts/verify_setup.py
```

Script memeriksa keberadaan file, bentuk data, kolom, missing values, duplikasi ID, distribusi target, dan kesesuaian sample submission.
