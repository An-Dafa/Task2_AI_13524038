"""Genetic Algorithm for Rotatable Edge-Matching Puzzle.

Chromosome = ordered list of (tile_id, rotation) genes.
Order Crossover preserves tile uniqueness, while each copied gene keeps a valid
rotation. Mutation can swap positions, rotate a tile, or apply both.
"""

from __future__ import annotations

from time import perf_counter
import numpy as np

from .model import EdgeMatchingProblem, EdgeMatchingState
from .results import SearchResult


def _tournament(
    population: list[EdgeMatchingState],
    scores: np.ndarray,
    rng: np.random.Generator,
    tournament_size: int,
) -> EdgeMatchingState:
    size = min(int(tournament_size), len(population))
    indices = rng.choice(len(population), size=size, replace=False)
    winner = int(indices[np.argmax(scores[indices])])
    return population[winner]


def order_crossover(
    parent_a: EdgeMatchingState,
    parent_b: EdgeMatchingState,
    rng: np.random.Generator,
) -> EdgeMatchingState:
    """OX on tile IDs; rotations follow the parent that contributed each tile gene."""
    n = len(parent_a.tile_ids)
    left, right = sorted(int(x) for x in rng.choice(n, size=2, replace=False))
    right += 1

    child_ids: list[int | None] = [None] * n
    child_rot: list[int | None] = [None] * n
    used: set[int] = set()

    for pos in range(left, right):
        tile_id = int(parent_a.tile_ids[pos])
        child_ids[pos] = tile_id
        child_rot[pos] = int(parent_a.rotations[pos])
        used.add(tile_id)

    b_genes = [
        (int(tile_id), int(rotation))
        for tile_id, rotation in zip(parent_b.tile_ids, parent_b.rotations)
        if int(tile_id) not in used
    ]
    positions = list(range(right, n)) + list(range(0, left))
    for pos, (tile_id, rotation) in zip(positions, b_genes):
        child_ids[pos] = tile_id
        child_rot[pos] = rotation

    if any(x is None for x in child_ids) or any(x is None for x in child_rot):
        raise RuntimeError("Order crossover menghasilkan chromosome tidak lengkap.")

    return EdgeMatchingState(
        tuple(int(x) for x in child_ids),
        tuple(int(x) for x in child_rot),
    )


def mutate(
    state: EdgeMatchingState,
    rng: np.random.Generator,
    *,
    swap_mutation_rate: float = 0.25,
    rotation_mutation_rate: float = 0.25,
) -> EdgeMatchingState:
    ids = list(state.tile_ids)
    rotations = list(state.rotations)
    if rng.random() < swap_mutation_rate:
        i, j = (int(x) for x in rng.choice(len(ids), size=2, replace=False))
        ids[i], ids[j] = ids[j], ids[i]
        rotations[i], rotations[j] = rotations[j], rotations[i]
    if rng.random() < rotation_mutation_rate:
        index = int(rng.integers(len(ids)))
        rotations[index] = (rotations[index] + int(rng.integers(1, 4))) % 4
    return EdgeMatchingState(tuple(ids), tuple(rotations))


def genetic_algorithm(
    problem: EdgeMatchingProblem,
    *,
    population_size: int = 120,
    generations: int = 600,
    tournament_size: int = 4,
    crossover_rate: float = 0.90,
    swap_mutation_rate: float = 0.25,
    rotation_mutation_rate: float = 0.35,
    elitism: int = 2,
    random_state: int = 42,
) -> SearchResult:
    if population_size < 2:
        raise ValueError("population_size minimal 2.")
    if not 0 <= elitism < population_size:
        raise ValueError("elitism harus berada pada rentang [0, population_size).")

    rng = np.random.default_rng(random_state)
    population = [
        problem.random_state(int(rng.integers(0, 2**31 - 1)))
        for _ in range(population_size)
    ]
    initial_scores = np.array([problem.score(x) for x in population], dtype=float)
    initial_best = population[int(np.argmax(initial_scores))]

    best_history: list[int] = []
    mean_history: list[float] = []
    best_states: list[EdgeMatchingState] = []
    started = perf_counter()
    stop_reason = "maximum_generations"

    for _ in range(generations):
        scores = np.array([problem.score(x) for x in population], dtype=float)
        order = np.argsort(-scores)
        generation_best = population[int(order[0])]
        generation_best_score = int(scores[order[0]])
        best_states.append(generation_best)
        best_history.append(generation_best_score)
        mean_history.append(float(scores.mean()))

        if generation_best_score == problem.max_score:
            stop_reason = "global_optimum"
            break

        next_population = [population[int(index)] for index in order[:elitism]]
        while len(next_population) < population_size:
            parent_a = _tournament(population, scores, rng, tournament_size)
            parent_b = _tournament(population, scores, rng, tournament_size)
            child = (
                order_crossover(parent_a, parent_b, rng)
                if rng.random() < crossover_rate
                else parent_a
            )
            child = mutate(
                child,
                rng,
                swap_mutation_rate=swap_mutation_rate,
                rotation_mutation_rate=rotation_mutation_rate,
            )
            problem.validate(child)
            next_population.append(child)
        population = next_population

    final_scores = np.array([problem.score(x) for x in population], dtype=float)
    final_best = population[int(np.argmax(final_scores))]
    final_best_score = int(np.max(final_scores))

    if not best_history or final_best_score > max(best_history):
        best_history.append(final_best_score)
        mean_history.append(float(final_scores.mean()))
        best_states.append(final_best)

    best_index = int(np.argmax(best_history))
    global_best = best_states[best_index]

    return SearchResult(
        algorithm="genetic_algorithm",
        initial_state=initial_best,
        final_state=final_best,
        best_state=global_best,
        score_history=best_history,
        state_history=best_states,
        iterations=len(best_history),
        duration_seconds=perf_counter() - started,
        max_score=problem.max_score,
        metadata={
            "stop_reason": stop_reason,
            "population_size": int(population_size),
            "generations_requested": int(generations),
            "generations_completed": len(best_history),
            "tournament_size": int(tournament_size),
            "crossover": "order_crossover",
            "crossover_rate": float(crossover_rate),
            "swap_mutation_rate": float(swap_mutation_rate),
            "rotation_mutation_rate": float(rotation_mutation_rate),
            "elitism": int(elitism),
            "average_fitness_history": mean_history,
        },
    )
