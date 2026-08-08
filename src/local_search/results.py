"""Shared search-result structure and JSON serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

from .model import EdgeMatchingState


@dataclass
class SearchResult:
    algorithm: str
    initial_state: EdgeMatchingState
    final_state: EdgeMatchingState
    best_state: EdgeMatchingState
    score_history: list[int]
    state_history: list[EdgeMatchingState]
    iterations: int
    duration_seconds: float
    max_score: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def initial_score(self) -> int:
        return int(self.score_history[0])

    @property
    def best_score(self) -> int:
        return int(max(self.score_history))

    @property
    def success(self) -> bool:
        return self.best_score >= self.max_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "initial_state": self.initial_state.to_pairs(),
            "final_state": self.final_state.to_pairs(),
            "best_state": self.best_state.to_pairs(),
            "score_history": [int(x) for x in self.score_history],
            "state_history": [state.to_pairs() for state in self.state_history],
            "iterations": int(self.iterations),
            "duration_seconds": float(self.duration_seconds),
            "max_score": int(self.max_score),
            "success": bool(self.success),
            "metadata": self.metadata,
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "SearchResult":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            algorithm=str(data["algorithm"]),
            initial_state=EdgeMatchingState.from_pairs(data["initial_state"]),
            final_state=EdgeMatchingState.from_pairs(data["final_state"]),
            best_state=EdgeMatchingState.from_pairs(data["best_state"]),
            score_history=[int(x) for x in data["score_history"]],
            state_history=[EdgeMatchingState.from_pairs(x) for x in data["state_history"]],
            iterations=int(data["iterations"]),
            duration_seconds=float(data["duration_seconds"]),
            max_score=int(data["max_score"]),
            metadata=dict(data.get("metadata", {})),
        )
