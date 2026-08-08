"""Generate guaranteed-solvable Edge-Matching instances."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .model import Tile


def _rotate_edges(edges: tuple[str, str, str, str], rotation: int) -> tuple[str, str, str, str]:
    k = rotation % 4
    return edges if k == 0 else edges[-k:] + edges[:-k]


def generate_solvable_instance(
    output_path: str | Path,
    *,
    board_size: int = 5,
    symbols: str = "ABCDE",
    seed: int = 13524038,
) -> None:
    """Construct a board with consistent internal edges, then hide its solution.

    The generated tile definitions are shuffled and independently re-oriented,
    so neither tile IDs nor rotation=0 reveal the generating solution.
    """
    rng = np.random.default_rng(seed)
    n = int(board_size)
    symbol_list = list(symbols)
    raw: list[list[str | None]] = [[None, None, None, None] for _ in range(n*n)]

    for row in range(n):
        for col in range(n):
            idx = row*n + col
            if row == 0:
                raw[idx][0] = str(rng.choice(symbol_list))
            if col == 0:
                raw[idx][3] = str(rng.choice(symbol_list))
            if col + 1 < n:
                symbol = str(rng.choice(symbol_list))
                raw[idx][1] = symbol
                raw[idx + 1][3] = symbol
            else:
                raw[idx][1] = str(rng.choice(symbol_list))
            if row + 1 < n:
                symbol = str(rng.choice(symbol_list))
                raw[idx][2] = symbol
                raw[idx + n][0] = symbol
            else:
                raw[idx][2] = str(rng.choice(symbol_list))

    ids = rng.permutation(np.arange(1, n*n + 1))
    tiles = []
    for idx, tile_id in enumerate(ids):
        edges = tuple(str(x) for x in raw[idx])
        hidden_rotation = int(rng.integers(0, 4))
        tiles.append({
            "id": int(tile_id),
            "edges": list(_rotate_edges(edges, hidden_rotation)),
        })
    rng.shuffle(tiles)

    payload = {
        "name": f"rotatable_edge_matching_{n}x{n}",
        "board_size": n,
        "symbols": symbol_list,
        "known_optimum": 2*n*(n-1),
        "tiles": tiles,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
