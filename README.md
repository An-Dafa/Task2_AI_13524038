# Task #2 — Seleksi Laboratorium Inteligensi Buatan

Repository final untuk dua bagian Task #2:

1. **Local Search** — proof of concept Diagonal Magic Cube 5×5×5 dengan Hill-Climbing, Simulated Annealing, dan Genetic Algorithm.
2. **DTL, Logistic Regression, dan SVM** — implementasi manual/from scratch untuk Kaggle, pembanding scikit-learn, validasi, eksperimen optimasi, dan model final.

> **Notebook rendering:** mathematical expressions use `$...$` and `$$...$$` delimiters for compatibility with Jupyter, VS Code, and GitHub notebook rendering.

## Hasil utama

| Model / tahap | Local Macro F1 | Public Kaggle |
|---|---:|---:|
| CART scratch baseline terbaik | 0.87198 | 0.85406 |
| Logistic Regression + nonlinear preprocessing | 0.86435 | — |
| SVM + nonlinear preprocessing | 0.86482 | — |
| External-augmented CART | 0.87975 | 0.86230 |
| **Final optimized manual DTL** | **0.91281** (mean 4 CV seeds) | **0.90548** |

Public score `0.90548` mencapai **rank 1** pada leaderboard saat submission final dikumpulkan.

> Catatan integritas: dataset sumber eksternal berisi label untuk baris yang juga muncul sebagai competition test. Label test tersebut **tidak digunakan** untuk training, validation, threshold selection, atau pembuatan submission. External source hanya dipakai untuk 9.000 baris berlabel yang tidak termasuk train/test kompetisi dan dua fitur tambahan (`person_education`, `loan_intent`).

## Struktur repository

```text
Task2_AI_13524038/
├── src/
│   ├── local_search/
│   │   ├── magic_cube.py
│   │   ├── hill_climbing.py
│   │   ├── simulated_annealing.py
│   │   └── genetic_algorithm.py
│   └── dtl_lr_svm/
│       ├── cart_scratch.py
│       ├── logistic_regression_scratch.py
│       ├── svm_scratch.py
│       ├── optimized_dtl.py
│       ├── external_data.py
│       ├── preprocessing.py
│       ├── metrics.py
│       ├── validation.py
│       └── sklearn_baselines.py
├── notebooks/
│   ├── local_search/
│   │   └── 01_diagonal_magic_cube_local_search.ipynb
│   └── dtl_lr_svm/
│       ├── 01_data_validation_and_baselines.ipynb
│       ├── 02_models_from_scratch.ipynb
│       ├── 03_final_optimized_dtl_kaggle.ipynb
│       └── 04_failed_experiments_and_bonus.ipynb
├── data/
│   ├── raw/
│   ├── external/
│   └── source_archives/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── submissions/
├── docs/
│   ├── README.md
│   └── writeup_notes.md
├── scripts/
│   ├── prepare_data.py
│   ├── verify_setup.py
│   └── make_final_submission.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\verify_setup.py
```

Notebook dapat dibuka melalui VS Code/Jupyter dengan kernel `.venv`.

## Menyiapkan data

Lihat `data/README.md`. Cara paling mudah adalah meletakkan dua archive yang diberikan/digunakan selama eksperimen di `data/source_archives/`, lalu menjalankan:

```powershell
python scripts\prepare_data.py
```

File yang diharapkan setelah ekstraksi:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
data/external/loan_data.csv
```

## Local Search

Masalah menggunakan complete-state formulation: sebuah state merupakan permutasi angka `1..125` pada kubus 5×5×5. Magic constant untuk `n=5` adalah `315`. Neighbor dibuat dengan menukar dua posisi. Objective yang diminimalkan:

```text
cost(state) = Σ |sum(line) - 315|
```

untuk 75 axis-aligned lines, 30 diagonal slice, dan 4 space diagonals. Global optimum memiliki cost `0`.

Implementasi mencakup:

- Steepest-Ascent Hill-Climbing;
- Sideways Move Hill-Climbing (bonus);
- Stochastic Hill-Climbing (bonus);
- Random-Restart Hill-Climbing (bonus);
- Simulated Annealing;
- Genetic Algorithm dengan tournament selection, order crossover, swap mutation, dan elitism.

Notebook: `notebooks/local_search/01_diagonal_magic_cube_local_search.ipynb`.

## DTL, LR, dan SVM from scratch

### CART

`cart_scratch.py` membangun binary tree dengan weighted Gini impurity, numeric threshold split, class weighting, leaf probability, dan feature importance. Ini merupakan implementasi DTL wajib yang dibandingkan langsung dengan `sklearn.tree.DecisionTreeClassifier`.

### Logistic Regression

`logistic_regression_scratch.py` mengimplementasikan sigmoid, weighted binary cross-entropy, L2 regularization, gradient descent/mini-batch training, class weighting, early stopping, dan threshold tuning.

### Linear SVM

`svm_scratch.py` mengimplementasikan primal linear SVM dengan weighted hinge loss, L2 regularization, subgradient descent, class weighting, learning-rate schedule, early stopping, dan decision-threshold tuning.

### Final optimized DTL

Model submission terbaik tetap merupakan **Decision Tree Learning manual**. Optimasi final menggunakan entropy/information gain dengan continuous binary threshold, full one-hot categorical encoding, 9.000 external unused rows, `loan_intent`, `person_education`, class/source weighting, dan exact split search pada `person_id`.

Konfigurasi final:

```text
max_depth                    = 12
min_samples_leaf             = 10
positive_weight_multiplier   = 0.84
external_weight              = 1.10
max thresholds (non-ID)      = 512
person_id threshold search   = exact
final threshold              = 0.7067307692307693
```

Robust 5-fold validation menggunakan empat seed menghasilkan macro F1:

```text
seed 42   : 0.911891
seed 314  : 0.913014
seed 2024 : 0.912818
seed 2718 : 0.913518
mean      : 0.912810
```

Untuk menghasilkan submission:

```powershell
python scripts\make_final_submission.py
```

Submission yang benar-benar memperoleh public score `0.90548` juga disimpan sebagai:

```text
outputs/submissions/submission_rank1_public_0.90548.csv
```

Implementasi final di `optimized_dtl.py` menggunakan struktur yang sama dengan submission tersebut (full one-hot + entropy/information gain + exact `person_id` split).

## Scikit-learn

Scikit-learn hanya dipakai sebagai **pembanding** dan utilitas validasi, bukan sebagai model submission. Submission final berasal dari Decision Tree manual.

## Dokumen akhir

`docs/writeup_notes.md` sudah menyiapkan kerangka write-up 5 halaman: TLDR, Problem Overview, DTL, LR, SVM, Validation Strategy, Percobaan Gagal, Pengembangan Lebih Lanjut, dan Referensi. PDF final akan diletakkan di:

```text
docs/Task2_AI_13524038.pdf
```
