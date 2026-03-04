"""Generate Figure XII-1: Battery label formation summary.

Pipeline shown:
SOH curve -> EOL cycle -> quantile bin -> class 0-4
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT_PATH = Path("figures/dissertation_plots/figure_xii_1_battery_label_formation_summary.svg")


def add_box(ax, x, y, w, h, text, fc="#f7f9fc", ec="#1f2937"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color="#111827")


def main():
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w, h = 0.2, 0.34
    y = 0.33
    xs = [0.03, 0.28, 0.53, 0.78]
    labels = [
        "SOH curve\n(per-cell degradation)",
        "EOL cycle\n(cycle at threshold)",
        "Quantile bin\n(rank-based partition)",
        "Class 0–4\n(discrete label)",
    ]
    colors = ["#e8f1ff", "#eefbf3", "#fff6e8", "#f5edff"]

    for x, label, color in zip(xs, labels, colors):
        add_box(ax, x, y, w, h, label, fc=color)

    for i in range(len(xs) - 1):
        x0 = xs[i] + w
        x1 = xs[i + 1]
        ax.annotate(
            "",
            xy=(x1 - 0.012, y + h / 2),
            xytext=(x0 + 0.012, y + h / 2),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#374151"),
        )

    ax.text(
        0.5,
        0.9,
        "Figure XII-1. Battery label formation summary",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.14,
        "SOH curve → EOL cycle → quantile binning → class label assignment (0–4)",
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
