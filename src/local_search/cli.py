"""Command-line entry point for the Local Search PoC."""

from __future__ import annotations

from pathlib import Path
import argparse

from .model import EdgeMatchingProblem
from .hill_climbing import steepest_ascent, sideways_move, stochastic_hill_climbing, random_restart
from .simulated_annealing import simulated_annealing
from .genetic_algorithm import genetic_algorithm
from .experiments import run_standard_experiments, run_ga_parameter_sweep, save_experiment_artifacts
from .replay import replay_from_files
from .visualization import plot_board


def _solve(args) -> None:
    problem = EdgeMatchingProblem.from_json(args.instance)
    initial = problem.random_state(args.seed)
    if args.algorithm == "steepest":
        result = steepest_ascent(problem, initial, random_state=args.seed)
    elif args.algorithm == "sideways":
        result = sideways_move(problem, initial, random_state=args.seed)
    elif args.algorithm == "stochastic":
        result = stochastic_hill_climbing(problem, initial, random_state=args.seed)
    elif args.algorithm == "restart":
        result = random_restart(problem, random_state=args.seed)
    elif args.algorithm == "sa":
        result = simulated_annealing(problem, initial, random_state=args.seed)
    elif args.algorithm == "ga":
        result = genetic_algorithm(problem, random_state=args.seed)
    else:
        raise ValueError(args.algorithm)

    print(f"Algorithm       : {result.algorithm}")
    print(f"Initial score   : {result.initial_score}/{problem.max_score}")
    print(f"Best score      : {result.best_score}/{problem.max_score}")
    print(f"Iterations/gen. : {result.iterations}")
    print(f"Duration        : {result.duration_seconds:.4f} s")
    print(f"Success         : {result.success}")

    if args.output:
        save_experiment_artifacts(problem, result, args.output, args.algorithm)
    if args.show:
        plot_board(problem, result.best_state, title=f"{result.algorithm}: {result.best_score}/{problem.max_score}")
        import matplotlib.pyplot as plt
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rotatable Edge-Matching Local Search PoC")
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="Run one algorithm")
    solve.add_argument("--instance", required=True)
    solve.add_argument("--algorithm", choices=["steepest", "sideways", "stochastic", "restart", "sa", "ga"], required=True)
    solve.add_argument("--seed", type=int, default=42)
    solve.add_argument("--output", default=None)
    solve.add_argument("--show", action="store_true")
    solve.set_defaults(func=_solve)

    experiment = sub.add_parser("experiment", help="Run 3x standard experiments")
    experiment.add_argument("--instance", required=True)
    experiment.add_argument("--runs", type=int, default=3)
    experiment.add_argument("--output", default="outputs/local_search/standard")
    experiment.add_argument("--seed", type=int, default=2026)
    experiment.set_defaults(func=lambda a: print(run_standard_experiments(
        EdgeMatchingProblem.from_json(a.instance), runs=a.runs, base_seed=a.seed, output_dir=a.output
    ).to_string(index=False)))

    sweep = sub.add_parser("ga-sweep", help="Run GA population/generation experiment")
    sweep.add_argument("--instance", required=True)
    sweep.add_argument("--runs", type=int, default=3)
    sweep.add_argument("--output", default="outputs/local_search/ga_sweep")
    sweep.set_defaults(func=lambda a: print(run_ga_parameter_sweep(
        EdgeMatchingProblem.from_json(a.instance), runs=a.runs, output_dir=a.output
    ).to_string(index=False)))

    replay = sub.add_parser("replay", help="Open bonus replay player")
    replay.add_argument("--instance", required=True)
    replay.add_argument("--trace", required=True)
    replay.set_defaults(func=lambda a: replay_from_files(a.instance, a.trace))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
