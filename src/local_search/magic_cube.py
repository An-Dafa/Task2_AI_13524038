"""Complete-state formulation for a 5x5x5 Diagonal Magic Cube.

The state is a permutation of 1..n^3.  A legal local move swaps any two
positions.  The objective is to minimize the total absolute deviation of all
required lines from the magic constant.  The implementation uses the 75
axis-aligned lines, 30 face/slice diagonals, and 4 space diagonals.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MagicCubeState:
    values: np.ndarray

    def as_cube(self, n: int = 5) -> np.ndarray:
        values = np.asarray(self.values, dtype=np.int64).reshape(-1)
        if values.size != n ** 3:
            raise ValueError(f"State harus memiliki {n ** 3} elemen.")
        return values.reshape(n, n, n)


class MagicCubeProblem:
    def __init__(self, n: int = 5, random_state: int = 42) -> None:
        if n < 3:
            raise ValueError("n minimal 3.")
        self.n = int(n)
        self.random_state = int(random_state)
        self.magic_constant = self.n * (self.n ** 3 + 1) // 2
        self.lines = self._line_indices(self.n)

    @staticmethod
    @lru_cache(maxsize=None)
    def _line_indices(n: int) -> tuple[np.ndarray, ...]:
        def flat(x: int, y: int, z: int) -> int:
            return x * n * n + y * n + z

        lines: list[np.ndarray] = []

        # 75 axis-aligned lines.
        for y in range(n):
            for z in range(n):
                lines.append(np.array([flat(x, y, z) for x in range(n)]))
        for x in range(n):
            for z in range(n):
                lines.append(np.array([flat(x, y, z) for y in range(n)]))
        for x in range(n):
            for y in range(n):
                lines.append(np.array([flat(x, y, z) for z in range(n)]))

        # Two diagonals for every slice in the three orientations: 30 lines.
        for z in range(n):
            lines.append(np.array([flat(i, i, z) for i in range(n)]))
            lines.append(np.array([flat(i, n - 1 - i, z) for i in range(n)]))
        for y in range(n):
            lines.append(np.array([flat(i, y, i) for i in range(n)]))
            lines.append(np.array([flat(i, y, n - 1 - i) for i in range(n)]))
        for x in range(n):
            lines.append(np.array([flat(x, i, i) for i in range(n)]))
            lines.append(np.array([flat(x, i, n - 1 - i) for i in range(n)]))

        # Four space diagonals.
        lines.extend(
            [
                np.array([flat(i, i, i) for i in range(n)]),
                np.array([flat(i, i, n - 1 - i) for i in range(n)]),
                np.array([flat(i, n - 1 - i, i) for i in range(n)]),
                np.array([flat(n - 1 - i, i, i) for i in range(n)]),
            ]
        )
        return tuple(lines)

    def random_state(self, seed: int | None = None) -> MagicCubeState:
        rng = np.random.default_rng(self.random_state if seed is None else seed)
        values = rng.permutation(np.arange(1, self.n ** 3 + 1, dtype=np.int64))
        return MagicCubeState(values)

    def validate(self, state: MagicCubeState) -> None:
        values = np.asarray(state.values, dtype=np.int64).reshape(-1)
        expected = np.arange(1, self.n ** 3 + 1, dtype=np.int64)
        if values.size != expected.size or not np.array_equal(np.sort(values), expected):
            raise ValueError("State harus merupakan permutasi lengkap 1..n^3 tanpa duplikasi.")

    def line_sums(self, state: MagicCubeState) -> np.ndarray:
        self.validate(state)
        values = np.asarray(state.values, dtype=np.int64).reshape(-1)
        return np.array([int(values[line].sum()) for line in self.lines], dtype=np.int64)

    def cost(self, state: MagicCubeState) -> int:
        deviations = np.abs(self.line_sums(state) - self.magic_constant)
        return int(deviations.sum())

    def score(self, state: MagicCubeState) -> int:
        # Higher is better; global optimum is 0.
        return -self.cost(state)

    def violations(self, state: MagicCubeState) -> int:
        return int(np.sum(self.line_sums(state) != self.magic_constant))

    def swap(self, state: MagicCubeState, i: int, j: int) -> MagicCubeState:
        if i == j:
            return state
        values = np.asarray(state.values, dtype=np.int64).copy()
        values[i], values[j] = values[j], values[i]
        return MagicCubeState(values)

    def random_neighbor(self, state: MagicCubeState, rng: np.random.Generator) -> MagicCubeState:
        i, j = rng.choice(self.n ** 3, size=2, replace=False)
        return self.swap(state, int(i), int(j))

    def sampled_neighbors(
        self,
        state: MagicCubeState,
        sample_size: int | None,
        rng: np.random.Generator,
    ) -> Iterable[MagicCubeState]:
        total = self.n ** 3
        if sample_size is None:
            for i in range(total - 1):
                for j in range(i + 1, total):
                    yield self.swap(state, i, j)
            return
        seen: set[tuple[int, int]] = set()
        while len(seen) < sample_size:
            i, j = sorted(rng.choice(total, size=2, replace=False).tolist())
            pair = (int(i), int(j))
            if pair not in seen:
                seen.add(pair)
                yield self.swap(state, *pair)
