"""Small visualization helper for the final manual decision tree."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt

from .optimized_dtl import OptimizedEntropyTree, TreeNode


def plot_top_tree(
    model: OptimizedEntropyTree,
    output_path: str | Path,
    *,
    max_depth: int = 3,
) -> None:
    """Render the top levels of the fitted tree for the bonus figure."""
    if model.tree_ is None:
        raise RuntimeError("Model belum di-fit.")

    nodes: list[tuple[TreeNode, int, float]] = []
    queue = deque([(model.tree_, 0, 0.5)])
    while queue:
        node, depth, x = queue.popleft()
        nodes.append((node, depth, x))
        if depth >= max_depth or node.is_leaf:
            continue
        span = 1.0 / (2 ** (depth + 2))
        if node.left is not None:
            queue.append((node.left, depth + 1, x - span))
        if node.right is not None:
            queue.append((node.right, depth + 1, x + span))

    fig, ax = plt.subplots(figsize=(12, 6))
    by_node = {id(node): (depth, x) for node, depth, x in nodes}
    for node, depth, x in nodes:
        y = -depth
        if not node.is_leaf and node.feature_index is not None:
            feature = model.feature_names_[node.feature_index]
            rule = f"{feature} <= {node.threshold:.4g}"
            text = f"{rule}\nIG={node.information_gain:.4f}\nn={node.n_samples}"
        else:
            text = f"leaf\np1={node.probability_1:.3f}\nn={node.n_samples}"
        ax.text(x, y, text, ha="center", va="center", bbox={"boxstyle": "round", "alpha": 0.15})
        for child in [node.left, node.right]:
            if child is not None and id(child) in by_node:
                child_depth, child_x = by_node[id(child)]
                ax.plot([x, child_x], [y - 0.08, -child_depth + 0.08])
    ax.set_axis_off()
    ax.set_title(f"Final Manual DTL — top {max_depth} levels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
