"""Four Hill-Climbing variants for Rotatable Edge-Matching Puzzle."""

from __future__ import annotations

from time import perf_counter
import numpy as np

from .model import EdgeMatchingProblem, EdgeMatchingState, Neighbor
from .results import SearchResult


def _best_neighbors(problem: EdgeMatchingProblem, state: EdgeMatchingState) -> tuple[int, list[Neighbor]]:
    best_score = -1
    best: list[Neighbor] = []
    for neighbor in problem.iter_neighbors(state):
        score = problem.score(neighbor.state)
        if score > best_score:
            best_score = score
            best = [neighbor]
        elif score == best_score:
            best.append(neighbor)
    return best_score, best


def steepest_ascent(
    problem: EdgeMatchingProblem,
    initial_state: EdgeMatchingState,
    *,
    max_iterations: int = 500,
    random_state: int = 42,
) -> SearchResult:
    rng = np.random.default_rng(random_state)
    current = initial_state
    current_score = problem.score(current)
    best_state = current
    history = [current_score]
    states = [current]
    moves: list[dict[str, object]] = []
    started = perf_counter()
    stop_reason = "max_iterations"

    for _ in range(max_iterations):
        if current_score == problem.max_score:
            stop_reason = "global_optimum"
            break
        best_score, candidates = _best_neighbors(problem, current)
        if best_score <= current_score:
            stop_reason = "local_optimum"
            break
        chosen = candidates[int(rng.integers(len(candidates)))]
        current = chosen.state
        current_score = best_score
        best_state = current
        history.append(current_score)
        states.append(current)
        moves.append({"type": chosen.move_type, "move": list(chosen.move)})

    return SearchResult(
        algorithm="steepest_ascent_hill_climbing",
        initial_state=initial_state,
        final_state=current,
        best_state=best_state,
        score_history=history,
        state_history=states,
        iterations=len(history) - 1,
        duration_seconds=perf_counter() - started,
        max_score=problem.max_score,
        metadata={"stop_reason": stop_reason, "moves": moves},
    )


def sideways_move(
    problem: EdgeMatchingProblem,
    initial_state: EdgeMatchingState,
    *,
    max_iterations: int = 1000,
    max_sideways: int = 50,
    random_state: int = 42,
) -> SearchResult:
    rng = np.random.default_rng(random_state)
    current = initial_state
    current_score = problem.score(current)
    best_state = current
    best_score_seen = current_score
    history = [current_score]
    states = [current]
    sideways_total = 0
    sideways_consecutive = 0
    visited = {current.signature()}
    moves: list[dict[str, object]] = []
    started = perf_counter()
    stop_reason = "max_iterations"

    for _ in range(max_iterations):
        if best_score_seen == problem.max_score:
            stop_reason = "global_optimum"
            break
        candidate_score, candidates = _best_neighbors(problem, current)
        candidates = [x for x in candidates if x.state.signature() not in visited] or candidates
        if candidate_score < current_score:
            stop_reason = "local_optimum"
            break
        if candidate_score == current_score:
            if sideways_consecutive >= max_sideways:
                stop_reason = "maximum_sideways"
                break
            sideways_consecutive += 1
            sideways_total += 1
        else:
            sideways_consecutive = 0

        chosen = candidates[int(rng.integers(len(candidates)))]
        current = chosen.state
        current_score = candidate_score
        visited.add(current.signature())
        if current_score > best_score_seen:
            best_state = current
            best_score_seen = current_score
        history.append(current_score)
        states.append(current)
        moves.append({"type": chosen.move_type, "move": list(chosen.move)})

    return SearchResult(
        algorithm="sideways_move_hill_climbing",
        initial_state=initial_state,
        final_state=current,
        best_state=best_state,
        score_history=history,
        state_history=states,
        iterations=len(history) - 1,
        duration_seconds=perf_counter() - started,
        max_score=problem.max_score,
        metadata={
            "stop_reason": stop_reason,
            "maximum_sideways": int(max_sideways),
            "sideways_moves": int(sideways_total),
            "moves": moves,
        },
    )


def stochastic_hill_climbing(
    problem: EdgeMatchingProblem,
    initial_state: EdgeMatchingState,
    *,
    max_iterations: int = 1000,
    random_state: int = 42,
) -> SearchResult:
    rng = np.random.default_rng(random_state)
    current = initial_state
    current_score = problem.score(current)
    best_state = current
    history = [current_score]
    states = [current]
    moves: list[dict[str, object]] = []
    started = perf_counter()
    stop_reason = "max_iterations"

    for _ in range(max_iterations):
        if current_score == problem.max_score:
            stop_reason = "global_optimum"
            break
        improving: list[tuple[Neighbor, int]] = []
        for neighbor in problem.iter_neighbors(current):
            score = problem.score(neighbor.state)
            if score > current_score:
                improving.append((neighbor, score))
        if not improving:
            stop_reason = "local_optimum"
            break

        gains = np.array([score - current_score for _, score in improving], dtype=float)
        probabilities = gains / gains.sum()
        chosen_index = int(rng.choice(len(improving), p=probabilities))
        chosen, current_score = improving[chosen_index]
        current = chosen.state
        best_state = current
        history.append(current_score)
        states.append(current)
        moves.append({"type": chosen.move_type, "move": list(chosen.move)})

    return SearchResult(
        algorithm="stochastic_hill_climbing",
        initial_state=initial_state,
        final_state=current,
        best_state=best_state,
        score_history=history,
        state_history=states,
        iterations=len(history) - 1,
        duration_seconds=perf_counter() - started,
        max_score=problem.max_score,
        metadata={"stop_reason": stop_reason, "moves": moves},
    )


def random_restart(
    problem: EdgeMatchingProblem,
    *,
    max_restarts: int = 10,
    iterations_per_restart: int = 400,
    random_state: int = 42,
) -> SearchResult:
    rng = np.random.default_rng(random_state)
    global_best: SearchResult | None = None
    combined_scores: list[int] = []
    combined_states: list[EdgeMatchingState] = []
    restart_summaries: list[dict[str, object]] = []
    restart_boundaries: list[int] = []
    first_initial: EdgeMatchingState | None = None
    started = perf_counter()

    for restart_index in range(max_restarts + 1):
        initial = problem.random_state(int(rng.integers(0, 2**31 - 1)))
        if first_initial is None:
            first_initial = initial
        result = steepest_ascent(
            problem,
            initial,
            max_iterations=iterations_per_restart,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        restart_boundaries.append(len(combined_scores))
        combined_scores.extend(result.score_history)
        combined_states.extend(result.state_history)
        restart_summaries.append({
            "restart": restart_index,
            "initial_score": result.initial_score,
            "best_score": result.best_score,
            "iterations": result.iterations,
            "stop_reason": result.metadata.get("stop_reason"),
        })
        if global_best is None or result.best_score > global_best.best_score:
            global_best = result
        if global_best.best_score == problem.max_score:
            break

    assert global_best is not None and first_initial is not None
    return SearchResult(
        algorithm="random_restart_hill_climbing",
        initial_state=first_initial,
        final_state=global_best.final_state,
        best_state=global_best.best_state,
        score_history=combined_scores,
        state_history=combined_states,
        iterations=max(0, len(combined_scores) - 1),
        duration_seconds=perf_counter() - started,
        max_score=problem.max_score,
        metadata={
            "restarts_used": len(restart_summaries) - 1,
            "maximum_restart": int(max_restarts),
            "iterations_per_restart": int(iterations_per_restart),
            "restart_boundaries": restart_boundaries,
            "restart_summaries": restart_summaries,
        },
    )
