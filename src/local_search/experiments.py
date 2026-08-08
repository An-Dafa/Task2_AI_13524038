"""Experiment utilities matching the written task specification."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import json

import pandas as pd

from .model import EdgeMatchingProblem
from .hill_climbing import steepest_ascent, sideways_move, stochastic_hill_climbing, random_restart
from .simulated_annealing import simulated_annealing
from .genetic_algorithm import genetic_algorithm
from .results import SearchResult
from .visualization import plot_board, plot_score_history, plot_simulated_annealing, plot_ga_history


Algorithm = Callable[..., SearchResult]


def _summary(result: SearchResult, run: int, seed: int) -> dict[str, object]:
    return {
        "algorithm": result.algorithm,
        "run": int(run),
        "seed": int(seed),
        "initial_score": result.initial_score,
        "best_score": result.best_score,
        "max_score": result.max_score,
        "success": result.success,
        "iterations_or_generations": result.iterations,
        "duration_seconds": result.duration_seconds,
        "stop_reason": result.metadata.get("stop_reason", ""),
    }


def save_experiment_artifacts(
    problem: EdgeMatchingProblem,
    result: SearchResult,
    output_dir: str | Path,
    stem: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save_json(output_dir / f"{stem}_trace.json")
    plot_board(problem, result.initial_state, title=f"Initial — {result.initial_score}/{problem.max_score}", save_path=output_dir / f"{stem}_initial.png")
    plot_board(problem, result.best_state, title=f"Best — {result.best_score}/{problem.max_score}", save_path=output_dir / f"{stem}_best.png")
    if result.algorithm == "simulated_annealing":
        plot_simulated_annealing(result, save_path=output_dir / f"{stem}_history.png")
    elif result.algorithm == "genetic_algorithm":
        plot_ga_history(result, save_path=output_dir / f"{stem}_history.png")
    else:
        plot_score_history(result, save_path=output_dir / f"{stem}_history.png")


def run_standard_experiments(
    problem: EdgeMatchingProblem,
    *,
    runs: int = 3,
    base_seed: int = 2026,
    output_dir: str | Path = "outputs/local_search/standard",
) -> pd.DataFrame:
    """Run all required algorithms plus all Hill-Climbing bonus variants."""
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []

    for run in range(1, runs + 1):
        seed = base_seed + run
        initial = problem.random_state(seed)
        configs: list[tuple[str, Callable[[], SearchResult]]] = [
            ("steepest", lambda: steepest_ascent(problem, initial, random_state=seed)),
            ("sideways", lambda: sideways_move(problem, initial, random_state=seed)),
            ("stochastic", lambda: stochastic_hill_climbing(problem, initial, random_state=seed)),
            ("restart", lambda: random_restart(problem, random_state=seed)),
            ("sa", lambda: simulated_annealing(problem, initial, random_state=seed)),
            ("ga", lambda: genetic_algorithm(problem, random_state=seed)),
        ]
        for short_name, runner in configs:
            result = runner()
            rows.append(_summary(result, run, seed))
            save_experiment_artifacts(problem, result, output_dir, f"run{run:02d}_{short_name}")

    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "summary.csv", index=False)
    return table


def run_ga_parameter_sweep(
    problem: EdgeMatchingProblem,
    *,
    runs: int = 3,
    population_fixed: int = 120,
    generation_values: tuple[int, ...] = (200, 600, 1200),
    generation_fixed: int = 600,
    population_values: tuple[int, ...] = (60, 120, 240),
    base_seed: int = 9000,
    output_dir: str | Path = "outputs/local_search/ga_sweep",
) -> pd.DataFrame:
    """Required GA experiment: vary generations, then vary population size."""
    rows: list[dict[str, object]] = []
    experiment_index = 0

    configs: list[tuple[str, int, int]] = []
    for generations in generation_values:
        configs.append(("vary_generations", population_fixed, int(generations)))
    for population_size in population_values:
        configs.append(("vary_population", int(population_size), generation_fixed))

    for experiment_type, population_size, generations in configs:
        for run in range(1, runs + 1):
            experiment_index += 1
            seed = base_seed + experiment_index
            result = genetic_algorithm(
                problem,
                population_size=population_size,
                generations=generations,
                random_state=seed,
            )
            row = _summary(result, run, seed)
            row.update({
                "experiment_type": experiment_type,
                "population_size": population_size,
                "generations_requested": generations,
            })
            rows.append(row)
            stem = f"{experiment_type}_pop{population_size}_gen{generations}_run{run}"
            save_experiment_artifacts(problem, result, output_dir, stem)

    table = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "ga_parameter_sweep.csv", index=False)
    return table
