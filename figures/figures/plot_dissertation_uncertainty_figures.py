#!/usr/bin/env python3
"""Create dissertation figures for AUPRC and missing uncertainty diagnostics.

The script packages the existing per-transfer results into a tidy table and emits:
1) AUPRC heatmap (model x transfer task)
2) Balanced accuracy vs H-score scatter
3) Regime-wise metric summary (closed/partial/open/universal)
4) Missing-metric coverage matrix (ECE/AUROC/Brier/NLL placeholders)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TASKS = ["0→1", "0→2", "0→3", "1→0", "1→2", "1→3", "2→0", "2→1", "2→3", "3→0", "3→1", "3→2"]

REGIME_BY_TASK = {
    "0→1": "closed-set",
    "0→2": "closed-set",
    "0→3": "partial-set",
    "1→0": "partial-set",
    "1→2": "open-set",
    "1→3": "open-set",
    "2→0": "open-set",
    "2→1": "open-set",
    "2→3": "universal",
    "3→0": "universal",
    "3→1": "universal",
    "3→2": "universal",
}

AUPRC: Dict[str, List[float]] = {
    "CNN": [0.779, 0.815, 0.880, 0.862, 0.667, 0.667, 0.80, 0.50, 0.63, 0.71, 0.67, 0.67],
    "CNN-SA": [0.802, 0.875, 0.873, 0.741, 0.667, 0.667, 0.80, 0.50, 0.72, 0.60, 0.67, 0.67],
    "CNN-OpenMax-SA": [0.869, 0.858, 0.911, 0.750, 0.667, 0.667, 0.78, 0.50, 0.83, 0.74, 0.67, 0.67],
    "WideResNet": [0.873, 0.766, 0.720, 0.750, 0.667, 0.667, 0.65, 0.50, 0.83, 0.70, 0.67, 0.67],
    "WideResNet-SA": [0.875, 0.844, 0.540, 0.889, 0.667, 0.667, 0.80, 0.50, 0.83, 0.83, 0.67, 0.67],
    "WideResNet-OpenMax-SA": [0.875, 0.875, 0.634, 0.750, 0.667, 0.667, 0.72, 0.50, 0.83, 0.83, 0.67, 0.67],
}

BAL_ACC: Dict[str, List[float]] = {
    "CNN": [75.81, 72.58, 68.97, 84.78, 48.39, 18.75, 36.36, 40.00, 37.50, 41.94, 85.71, 85.71],
    "CNN-SA": [83.33, 87.10, 75.86, 92.86, 51.62, 37.50, 36.36, 40.00, 40.63, 41.94, 78.87, 83.80],
    "CNN-OpenMax-SA": [72.58, 79.03, 79.31, 92.82, 51.62, 28.13, 31.82, 40.00, 37.50, 41.94, 83.81, 83.80],
    "WideResNet": [81.58, 79.00, 68.97, 92.84, 51.62, 37.50, 36.36, 40.00, 40.63, 41.94, 83.81, 83.80],
    "WideResNet-SA": [85.48, 80.00, 55.17, 76.05, 51.62, 34.38, 36.36, 40.00, 40.63, 41.94, 66.90, 83.80],
    "WideResNet-OpenMax-SA": [85.48, 83.33, 68.97, 92.85, 50.53, 37.50, 22.73, 40.00, 40.63, 50.53, 85.71, 85.71],
}

H_SCORE: Dict[str, List[float]] = {
    "CNN": [56.31, 55.49, 53.86, 55.91, 54.27, 42.07, 54.15, 53.23, 46.94, 49.72, 54.14, 48.45],
    "CNN-SA": [63.95, 59.16, 52.09, 58.75, 56.84, 59.27, 56.72, 57.56, 59.77, 61.73, 64.03, 77.11],
    "CNN-OpenMax-SA": [57.45, 56.59, 55.69, 61.56, 54.84, 51.02, 58.03, 58.16, 53.32, 59.97, 71.80, 68.81],
    "WideResNet": [59.36, 54.63, 54.34, 54.20, 61.31, 59.27, 54.57, 49.44, 55.63, 55.04, 73.63, 69.13],
    "WideResNet-SA": [60.04, 64.53, 47.17, 51.64, 53.10, 50.57, 52.49, 54.19, 56.99, 61.07, 61.69, 78.45],
    "WideResNet-OpenMax-SA": [64.91, 55.49, 54.34, 60.55, 59.57, 55.96, 41.98, 53.15, 60.84, 59.95, 59.45, 59.07],
}


def build_dataframe() -> pd.DataFrame:
    rows = []
    for model in AUPRC:
        for idx, task in enumerate(TASKS):
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "regime": REGIME_BY_TASK[task],
                    "auprc": AUPRC[model][idx],
                    "balanced_accuracy": BAL_ACC[model][idx],
                    "h_score": H_SCORE[model][idx],
                    "ece": None,
                    "auroc": None,
                    "brier": None,
                    "nll": None,
                }
            )
    return pd.DataFrame(rows)


def plot_all(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(11, 4.8))
    heat = df.pivot(index="model", columns="task", values="auprc")
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="viridis", vmin=0.45, vmax=0.95)
    plt.title("AUPRC across transfer tasks and model variants")
    plt.tight_layout()
    plt.savefig(out_dir / "auprc-heatmap.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.2, 5.2))
    sns.scatterplot(data=df, x="balanced_accuracy", y="h_score", hue="model", style="regime", s=85)
    plt.title("Balanced accuracy vs H-score")
    plt.xlabel("Balanced accuracy (%)")
    plt.ylabel("H-score (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "balanced-accuracy-vs-hscore.png", dpi=300)
    plt.close()

    regime_summary = (
        df.groupby("regime", as_index=False)[["auprc", "balanced_accuracy", "h_score"]].mean()
        .melt(id_vars="regime", var_name="metric", value_name="value")
    )
    plt.figure(figsize=(8.4, 5.2))
    sns.barplot(data=regime_summary, x="regime", y="value", hue="metric")
    plt.title("Regime-wise metric summary")
    plt.xlabel("Domain adaptation regime")
    plt.ylabel("Mean score")
    plt.tight_layout()
    plt.savefig(out_dir / "regime-metric-summary.png", dpi=300)
    plt.close()

    required = ["auprc", "ece", "auroc", "brier", "nll"]
    coverage = pd.DataFrame({"metric": required, "available": [float(df[m].notna().mean()) for m in required]})
    plt.figure(figsize=(6.5, 3.8))
    sns.barplot(data=coverage, x="metric", y="available", color="#4C78A8")
    plt.ylim(0, 1.0)
    plt.ylabel("Data coverage ratio")
    plt.title("Uncertainty diagnostics coverage (missing metrics highlighted)")
    for i, row in coverage.iterrows():
        plt.text(i, row["available"] + 0.03, f"{row['available']:.0%}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "uncertainty-metric-coverage.png", dpi=300)
    plt.close()


def main() -> None:
    out_dir = Path("dissertation_plots")
    df = build_dataframe()
    df.to_csv(out_dir / "uncertainty_metrics_table.csv", index=False)
    plot_all(df, out_dir)
    print(f"Generated figures in {out_dir}")


if __name__ == "__main__":
    main()
