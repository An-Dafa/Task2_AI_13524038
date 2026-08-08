"""Interactive replay player (bonus) using matplotlib widgets."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

from .model import EdgeMatchingProblem
from .results import SearchResult
from .visualization import plot_board


class ReplayPlayer:
    """Play/pause, prev/next, progress slider, and playback speed controls."""

    def __init__(self, problem: EdgeMatchingProblem, result: SearchResult) -> None:
        if not result.state_history:
            raise ValueError("Trace tidak memiliki state_history.")
        self.problem = problem
        self.result = result
        self.index = 0
        self.playing = False
        self.speed = 1.0

        self.fig = plt.figure(figsize=(9, 8))
        self.board_ax = self.fig.add_axes([0.08, 0.22, 0.84, 0.70])
        self.prev_ax = self.fig.add_axes([0.10, 0.08, 0.12, 0.05])
        self.play_ax = self.fig.add_axes([0.25, 0.08, 0.16, 0.05])
        self.next_ax = self.fig.add_axes([0.44, 0.08, 0.12, 0.05])
        self.progress_ax = self.fig.add_axes([0.10, 0.155, 0.70, 0.03])
        self.speed_ax = self.fig.add_axes([0.68, 0.08, 0.22, 0.03])

        self.prev_button = Button(self.prev_ax, "Previous")
        self.play_button = Button(self.play_ax, "Play / Pause")
        self.next_button = Button(self.next_ax, "Next")
        self.progress = Slider(
            self.progress_ax,
            "Step",
            0,
            max(0, len(result.state_history) - 1),
            valinit=0,
            valstep=1,
        )
        self.speed_slider = Slider(self.speed_ax, "Speed", 0.25, 4.0, valinit=1.0)

        self.timer = self.fig.canvas.new_timer(interval=500)
        self.timer.add_callback(self._tick)
        self.prev_button.on_clicked(self._prev)
        self.play_button.on_clicked(self._toggle)
        self.next_button.on_clicked(self._next)
        self.progress.on_changed(self._seek)
        self.speed_slider.on_changed(self._change_speed)
        self._draw()

    def _draw(self) -> None:
        self.board_ax.clear()
        state = self.result.state_history[self.index]
        n = self.problem.board_size
        for row in range(n):
            for col in range(n):
                idx = row * n + col
                x, y = col, n - 1 - row
                self.board_ax.add_patch(Rectangle((x, y), 1, 1, fill=False, linewidth=1.2))
                top, right, bottom, left = self.problem.tile_edges_at(state, idx)
                self.board_ax.text(x + 0.5, y + 0.55, f"T{state.tile_ids[idx]}", ha="center", va="center")
                self.board_ax.text(x + 0.5, y + 0.33, f"{90*state.rotations[idx]}°", ha="center", va="center", fontsize=7)
                self.board_ax.text(x + 0.5, y + 0.91, top, ha="center", va="top", fontsize=8)
                self.board_ax.text(x + 0.91, y + 0.5, right, ha="right", va="center", fontsize=8)
                self.board_ax.text(x + 0.5, y + 0.09, bottom, ha="center", va="bottom", fontsize=8)
                self.board_ax.text(x + 0.09, y + 0.5, left, ha="left", va="center", fontsize=8)
        score = self.problem.score(state)
        self.board_ax.set_title(f"{self.result.algorithm} — step {self.index} — score {score}/{self.problem.max_score}")
        self.board_ax.set_xlim(0, n)
        self.board_ax.set_ylim(0, n)
        self.board_ax.set_aspect("equal")
        self.board_ax.axis("off")
        if int(self.progress.val) != self.index:
            self.progress.set_val(self.index)
        self.fig.canvas.draw_idle()

    def _prev(self, _event) -> None:
        self.index = max(0, self.index - 1)
        self._draw()

    def _next(self, _event) -> None:
        self.index = min(len(self.result.state_history) - 1, self.index + 1)
        self._draw()

    def _toggle(self, _event) -> None:
        self.playing = not self.playing
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _seek(self, value) -> None:
        new_index = int(value)
        if new_index != self.index:
            self.index = new_index
            self._draw()

    def _change_speed(self, value) -> None:
        self.speed = float(value)
        self.timer.interval = max(50, int(500 / self.speed))

    def _tick(self) -> None:
        if not self.playing:
            return
        if self.index >= len(self.result.state_history) - 1:
            self.playing = False
            self.timer.stop()
            return
        self.index += 1
        self._draw()

    def show(self) -> None:
        plt.show()


def replay_from_files(instance_path: str, trace_path: str) -> None:
    problem = EdgeMatchingProblem.from_json(instance_path)
    result = SearchResult.from_json(trace_path)
    ReplayPlayer(problem, result).show()
