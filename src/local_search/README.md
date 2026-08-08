# Local Search PoC — Rotatable Edge-Matching Puzzle

Implementasi lengkap proof of concept untuk **Pencarian Solusi Rotatable Edge-Matching Puzzle dengan Local Search**.

## Formulasi

- Board utama: `5 x 5` = 25 tile.
- State lengkap: permutation 25 tile + rotasi tiap tile (`0°, 90°, 180°, 270°`).
- Neighbor: `Swap Move` dan `Rotate Move`.
- Objective: jumlah pasangan edge horizontal/vertikal yang match.
- Global optimum untuk 5x5: `40/40`.

## Algoritma

Wajib:
- salah satu Hill-Climbing;
- Simulated Annealing;
- Genetic Algorithm.

PoC ini sekaligus mengimplementasikan bonus seluruh Hill-Climbing:
- Steepest-Ascent;
- Sideways Move;
- Stochastic;
- Random Restart.

GA menggunakan tournament selection, Order Crossover yang menjaga uniqueness tile, swap/rotation mutation, dan elitism.

## Quick Start

Dari root repository:

```powershell
python -m src.local_search.cli solve `
  --instance src/local_search/instances/edge_matching_5x5.json `
  --algorithm sa `
  --seed 42 `
  --output outputs/local_search/demo `
  --show
```

Pilihan algorithm: `steepest`, `sideways`, `stochastic`, `restart`, `sa`, `ga`.

Eksperimen seluruh algoritma sebanyak 3 run:

```powershell
python -m src.local_search.cli experiment `
  --instance src/local_search/instances/edge_matching_5x5.json `
  --runs 3
```

Eksperimen parameter GA:

```powershell
python -m src.local_search.cli ga-sweep `
  --instance src/local_search/instances/edge_matching_5x5.json `
  --runs 3
```

Replay bonus:

```powershell
python -m src.local_search.cli replay `
  --instance src/local_search/instances/edge_matching_5x5.json `
  --trace outputs/local_search/demo/sa_trace.json
```

Replay menyediakan play/pause, previous/next, progress slider, dan playback speed.

## File utama

- `model.py` — Tile, complete state, constraints, objective, swap/rotate neighbor.
- `hill_climbing.py` — 4 varian HC.
- `simulated_annealing.py` — SA + temperature/acceptance tracking.
- `genetic_algorithm.py` — population, selection, crossover, mutation, elitism.
- `experiments.py` — 3-run experiment dan GA parameter sweep.
- `visualization.py` — board serta objective/fitness plots.
- `replay.py` — interactive replay player bonus.
- `instance_generator.py` — generator instance solvable.
- `instances/edge_matching_5x5.json` — instance utama 5x5.

Generated experiment outputs ditempatkan di `outputs/local_search/` dan di-ignore oleh Git.
