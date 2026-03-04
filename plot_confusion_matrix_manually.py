#!/usr/bin/env python3
"""Recreate confusion matrices in a clearer, publication-style format.

This script intentionally hardcodes confusion-matrix values so you can reproduce
an existing figure with improved readability (matching the clearer style shown
in your second attachment).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Hardcoded matrices (10x10) reconstructed from the original figure.
# Update these values if you want to exactly match another run.
# -----------------------------------------------------------------------------
BASELINE_CM = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 24, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 23, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 24, 0],
        [0, 21, 0, 0, 0, 0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 19, 0, 5, 0],
        [0, 1, 18, 0, 0, 0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 24, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 23, 0],
    ],
    dtype=int,
)

TRANSFER_CM = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 24, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 24, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 24, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 24, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 24,0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 24, 23, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 24, 24],
        [0, 0, 0, 17, 0, 0, 0, 1, 5, 0],
    ],
    dtype=int,
)


def _plot_confusion(ax: plt.Axes, cm: np.ndarray, title: str) -> None:
    labels = list(range(cm.shape[0]))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_aspect("equal")

    # Keep lines off annotation centers: disable major grid and draw optional
    # boundary lines on half-steps between cells.
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            ax.text(
                j,
                i,
                f"{v}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if v > thresh else "#1f3a73",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)


def create_figure(output_path: Path) -> Path:
    if BASELINE_CM.shape != TRANSFER_CM.shape:
        raise ValueError("BASELINE_CM and TRANSFER_CM must have the same shape.")
    if BASELINE_CM.shape[0] != BASELINE_CM.shape[1]:
        raise ValueError("Confusion matrices must be square.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    _plot_confusion(axes[0], BASELINE_CM, "Baseline 2 | acc=50.00%")
    _plot_confusion(axes[1], TRANSFER_CM, "Transfer 0→2 | acc=87.89%")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate confusion matrices in clearer format.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures") / "recreated_confusion_matrix_0_2.png",
        help="Path to save the recreated figure.",
    )
    args = parser.parse_args()

    out = create_figure(args.out)
    print(f"Saved recreated confusion-matrix figure to: {out}")


if __name__ == "__main__":
    main()
