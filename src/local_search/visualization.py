"""Matplotlib visualizations for states and search histories."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .model import EdgeMatchingProblem, EdgeMatchingState
from .results import SearchResult


def plot_board(
    problem: EdgeMatchingProblem,
    state: EdgeMatchingState,
    *,
    title: str | None = None,
    highlight_matches: bool = True,
    show_tile_id: bool = True,
    save_path: str | Path | None = None,
):
    problem.validate(state)
    n = problem.board_size
    fig, ax = plt.subplots(figsize=(1.7 * n, 1.7 * n))

    for row in range(n):
        for col in range(n):
            idx = row * n + col
            x, y = col, n - 1 - row
            ax.add_patch(Rectangle((x, y), 1, 1, fill=False, linewidth=1.3))
            top, right, bottom, left = problem.tile_edges_at(state, idx)
            if show_tile_id:
                ax.text(x + 0.5, y + 0.5, f"T{state.tile_ids[idx]}", ha="center", va="center", fontsize=9)
                ax.text(x + 0.5, y + 0.33, f"r={90 * state.rotations[idx]}°", ha="center", va="center", fontsize=7)
            ax.text(x + 0.5, y + 0.91, top, ha="center", va="top", fontsize=8)
            ax.text(x + 0.91, y + 0.5, right, ha="right", va="center", fontsize=8)
            ax.text(x + 0.5, y + 0.09, bottom, ha="center", va="bottom", fontsize=8)
            ax.text(x + 0.09, y + 0.5, left, ha="left", va="center", fontsize=8)

    if highlight_matches:
        for item in problem.adjacency_details(state):
            a = int(item["a"])
            b = int(item["b"])
            match = bool(item["match"])
            style = "-" if match else "--"
            width = 3.0 if match else 2.0
            ra, ca = divmod(a, n)
            rb, cb = divmod(b, n)
            if item["direction"] == "horizontal":
                x = max(ca, cb)
                y0 = n - 1 - ra
                ax.plot([x, x], [y0, y0 + 1], linestyle=style, linewidth=width)
            else:
                x0 = ca
                y = n - 1 - max(ra, rb) + 1
                ax.plot([x0, x0 + 1], [y, y], linestyle=style, linewidth=width)

    score = problem.score(state)
    ax.set_title(title or f"Score = {score}/{problem.max_score}")
    ax.set_xlim(-0.05, n + 0.05)
    ax.set_ylim(-0.05, n + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, ax


def plot_score_history(result: SearchResult, *, save_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(result.score_history)), result.score_history)
    ax.axhline(result.max_score, linestyle="--", linewidth=1, label="Global optimum")
    ax.set_xlabel("Iteration / Generation")
    ax.set_ylabel("Objective / Fitness")
    ax.set_title(result.algorithm)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, ax


def plot_simulated_annealing(result: SearchResult, *, save_path: str | Path | None = None):
    temperatures = result.metadata.get("temperature_history", [])
    probabilities = result.metadata.get("acceptance_probability_history", [])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.score_history, label="Objective")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective")
    if temperatures:
        ax2 = ax.twinx()
        ax2.plot(temperatures, alpha=0.55, label="Temperature")
        ax2.set_ylabel("Temperature")
    ax.set_title("Simulated Annealing")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, ax


def plot_ga_history(result: SearchResult, *, save_path: str | Path | None = None):
    mean_history = result.metadata.get("average_fitness_history", [])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.score_history, label="Best fitness")
    if mean_history:
        ax.plot(mean_history, label="Average fitness")
    ax.axhline(result.max_score, linestyle="--", linewidth=1, label="Global optimum")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Genetic Algorithm")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig, ax
