"""Simulated Annealing for Rotatable Edge-Matching Puzzle."""

from __future__ import annotations

from time import perf_counter
import math
import numpy as np

from .model import EdgeMatchingProblem, EdgeMatchingState
from .results import SearchResult


def simulated_annealing(
    problem: EdgeMatchingProblem,
    initial_state: EdgeMatchingState,
    *,
    initial_temperature: float = 8.0,
    cooling_rate: float = 0.997,
    minimum_temperature: float = 1e-3,
    max_iterations: int = 20000,
    swap_probability: float = 0.60,
    random_state: int = 42,
) -> SearchResult:
    """Geometric-cooling SA that maximizes the number of matching edges."""
    rng = np.random.default_rng(random_state)
    current = initial_state
    current_score = problem.score(current)
    best = current
    best_score = current_score
    temperature = float(initial_temperature)

    scores = [current_score]
    states = [current]
    temperatures = [temperature]
    acceptance_probabilities = [1.0]
    accepted_flags = [True]
    accepted_worse_moves = 0
    escape_events = 0
    worse_since_last_best = False
    moves: list[dict[str, object]] = []
    started = perf_counter()
    stop_reason = "max_iterations"

    for _ in range(max_iterations):
        if best_score == problem.max_score:
            stop_reason = "global_optimum"
            break
        if temperature < minimum_temperature:
            stop_reason = "minimum_temperature"
            break

        neighbor = problem.random_neighbor(current, rng, swap_probability=swap_probability)
        candidate_score = problem.score(neighbor.state)
        delta = candidate_score - current_score
        probability = 1.0 if delta >= 0 else math.exp(delta / max(temperature, 1e-12))
        accepted = delta >= 0 or rng.random() < probability

        if accepted:
            if delta < 0:
                accepted_worse_moves += 1
                worse_since_last_best = True
            current = neighbor.state
            current_score = candidate_score
            moves.append({"type": neighbor.move_type, "move": list(neighbor.move), "delta": int(delta)})
            if current_score > best_score:
                if worse_since_last_best:
                    escape_events += 1
                best = current
                best_score = current_score
                worse_since_last_best = False

        scores.append(current_score)
        states.append(current)
        acceptance_probabilities.append(float(probability))
        accepted_flags.append(bool(accepted))
        temperature *= cooling_rate
        temperatures.append(temperature)

    return SearchResult(
        algorithm="simulated_annealing",
        initial_state=initial_state,
        final_state=current,
        best_state=best,
        score_history=scores,
        state_history=states,
        iterations=len(scores) - 1,
        duration_seconds=perf_counter() - started,
        max_score=problem.max_score,
        metadata={
            "stop_reason": stop_reason,
            "initial_temperature": float(initial_temperature),
            "cooling_rate": float(cooling_rate),
            "minimum_temperature": float(minimum_temperature),
            "temperature_history": temperatures,
            "acceptance_probability_history": acceptance_probabilities,
            "accepted_history": accepted_flags,
            "accepted_worse_moves": int(accepted_worse_moves),
            "local_optimum_escape_events": int(escape_events),
            "moves": moves,
        },
    )
