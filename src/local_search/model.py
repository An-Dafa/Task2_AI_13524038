"""Core model for the Rotatable Edge-Matching Puzzle.

A state is a complete N x N board represented by:
1) a permutation of all tile IDs; and
2) one rotation value in {0,1,2,3} for every board position.

The objective is maximized: one point is awarded for every horizontal or
vertical pair of touching edges that contains the same symbol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal
import json

import numpy as np


EdgeTuple = tuple[str, str, str, str]  # top, right, bottom, left
MoveType = Literal["swap", "rotate"]


@dataclass(frozen=True)
class Tile:
    tile_id: int
    edges: EdgeTuple

    def rotated_edges(self, rotation: int) -> EdgeTuple:
        """Return edges after rotation * 90 degrees clockwise."""
        k = int(rotation) % 4
        if k == 0:
            return self.edges
        return self.edges[-k:] + self.edges[:-k]


@dataclass(frozen=True)
class EdgeMatchingState:
    tile_ids: tuple[int, ...]
    rotations: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.tile_ids) != len(self.rotations):
            raise ValueError("tile_ids dan rotations harus memiliki panjang yang sama.")

    def signature(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.tile_ids, self.rotations))

    def to_pairs(self) -> list[list[int]]:
        return [[int(t), int(r)] for t, r in self.signature()]

    @classmethod
    def from_pairs(cls, pairs: Iterable[Iterable[int]]) -> "EdgeMatchingState":
        parsed = [(int(a), int(b)) for a, b in pairs]
        return cls(
            tile_ids=tuple(a for a, _ in parsed),
            rotations=tuple(b for _, b in parsed),
        )


@dataclass(frozen=True)
class Neighbor:
    state: EdgeMatchingState
    move_type: MoveType
    move: tuple[int, ...]


class EdgeMatchingProblem:
    """Complete-state formulation of an N x N Rotatable Edge-Matching Puzzle."""

    def __init__(self, board_size: int, tiles: Iterable[Tile], name: str = "edge-matching") -> None:
        self.board_size = int(board_size)
        if self.board_size < 2:
            raise ValueError("board_size minimal 2.")

        self.tiles = {int(tile.tile_id): tile for tile in tiles}
        expected = self.board_size ** 2
        if len(self.tiles) != expected:
            raise ValueError(f"Puzzle {self.board_size}x{self.board_size} membutuhkan {expected} tile unik.")
        if len(set(self.tiles)) != expected:
            raise ValueError("Tile ID harus unik.")

        self.name = str(name)
        self.max_score = 2 * self.board_size * (self.board_size - 1)
        self._valid_ids = tuple(sorted(self.tiles))

    @classmethod
    def from_json(cls, path: str | Path) -> "EdgeMatchingProblem":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tiles = [
            Tile(int(item["id"]), tuple(str(x) for x in item["edges"]))
            for item in data["tiles"]
        ]
        return cls(int(data["board_size"]), tiles, data.get("name", Path(path).stem))

    def validate(self, state: EdgeMatchingState) -> None:
        expected = self.board_size ** 2
        if len(state.tile_ids) != expected:
            raise ValueError(f"State harus memiliki {expected} posisi.")
        if tuple(sorted(state.tile_ids)) != self._valid_ids:
            raise ValueError("State harus memakai setiap tile tepat satu kali.")
        if any(int(r) not in (0, 1, 2, 3) for r in state.rotations):
            raise ValueError("Rotasi tile hanya boleh 0, 1, 2, atau 3.")

    def random_state(self, seed: int | None = None) -> EdgeMatchingState:
        rng = np.random.default_rng(seed)
        ids = list(self._valid_ids)
        rng.shuffle(ids)
        rotations = rng.integers(0, 4, size=len(ids)).tolist()
        return EdgeMatchingState(tuple(int(x) for x in ids), tuple(int(x) for x in rotations))

    def tile_edges_at(self, state: EdgeMatchingState, index: int) -> EdgeTuple:
        tile = self.tiles[int(state.tile_ids[index])]
        return tile.rotated_edges(int(state.rotations[index]))

    def score(self, state: EdgeMatchingState) -> int:
        self.validate(state)
        n = self.board_size
        matched = 0
        for row in range(n):
            for col in range(n):
                idx = row * n + col
                top, right, bottom, left = self.tile_edges_at(state, idx)
                if col + 1 < n:
                    right_idx = idx + 1
                    right_neighbor = self.tile_edges_at(state, right_idx)
                    matched += int(right == right_neighbor[3])
                if row + 1 < n:
                    down_idx = idx + n
                    down_neighbor = self.tile_edges_at(state, down_idx)
                    matched += int(bottom == down_neighbor[0])
        return int(matched)

    def mismatch_count(self, state: EdgeMatchingState) -> int:
        return self.max_score - self.score(state)

    def is_goal(self, state: EdgeMatchingState) -> bool:
        return self.score(state) == self.max_score

    def adjacency_details(self, state: EdgeMatchingState) -> list[dict[str, object]]:
        """Describe every horizontal/vertical touching edge for visualization."""
        self.validate(state)
        n = self.board_size
        rows: list[dict[str, object]] = []
        for row in range(n):
            for col in range(n):
                idx = row * n + col
                edges = self.tile_edges_at(state, idx)
                if col + 1 < n:
                    j = idx + 1
                    other = self.tile_edges_at(state, j)
                    rows.append({
                        "a": idx,
                        "b": j,
                        "direction": "horizontal",
                        "symbol_a": edges[1],
                        "symbol_b": other[3],
                        "match": edges[1] == other[3],
                    })
                if row + 1 < n:
                    j = idx + n
                    other = self.tile_edges_at(state, j)
                    rows.append({
                        "a": idx,
                        "b": j,
                        "direction": "vertical",
                        "symbol_a": edges[2],
                        "symbol_b": other[0],
                        "match": edges[2] == other[0],
                    })
        return rows

    def swap(self, state: EdgeMatchingState, i: int, j: int) -> EdgeMatchingState:
        if i == j:
            return state
        ids = list(state.tile_ids)
        rotations = list(state.rotations)
        ids[i], ids[j] = ids[j], ids[i]
        rotations[i], rotations[j] = rotations[j], rotations[i]
        return EdgeMatchingState(tuple(ids), tuple(rotations))

    def rotate(self, state: EdgeMatchingState, index: int, quarter_turns: int = 1) -> EdgeMatchingState:
        rotations = list(state.rotations)
        rotations[index] = (int(rotations[index]) + int(quarter_turns)) % 4
        return EdgeMatchingState(state.tile_ids, tuple(rotations))

    def iter_neighbors(
        self,
        state: EdgeMatchingState,
        *,
        include_swap: bool = True,
        include_rotate: bool = True,
    ) -> Iterator[Neighbor]:
        """Enumerate the full neighborhood (375 neighbors for a 5x5 board)."""
        self.validate(state)
        total = self.board_size ** 2
        if include_swap:
            for i in range(total - 1):
                for j in range(i + 1, total):
                    yield Neighbor(self.swap(state, i, j), "swap", (i, j))
        if include_rotate:
            for i in range(total):
                for quarter_turns in (1, 2, 3):
                    yield Neighbor(self.rotate(state, i, quarter_turns), "rotate", (i, quarter_turns))

    def random_neighbor(
        self,
        state: EdgeMatchingState,
        rng: np.random.Generator,
        *,
        swap_probability: float = 0.60,
    ) -> Neighbor:
        total = self.board_size ** 2
        if rng.random() < swap_probability:
            i, j = rng.choice(total, size=2, replace=False)
            i, j = int(i), int(j)
            return Neighbor(self.swap(state, i, j), "swap", (i, j))
        index = int(rng.integers(total))
        turns = int(rng.integers(1, 4))
        return Neighbor(self.rotate(state, index, turns), "rotate", (index, turns))
