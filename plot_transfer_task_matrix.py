
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create transfer-task matrix heatmaps across CWRU and Battery datasets.

The matrix visualizes *domain gap magnitude* per source→target transfer pair using:

    gap = 1 - transfer_score

By default, the script reads the ablation CSVs already generated in ``figures/`` and
uses the baseline transfer score columns (``transfer_score_base``). You can switch to
improved scores with ``--score_variant improved``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_CWRU_LOADS = {
    0: {"hp": 0, "rpm": 1797},
    1: {"hp": 1, "rpm": 1772},
    2: {"hp": 2, "rpm": 1750},
    3: {"hp": 3, "rpm": 1730},
}


def _format_cwru_domain(value: str | int) -> str:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return str(value)

    meta = _CWRU_LOADS.get(idx, {})
    hp = meta.get("hp")
    rpm = meta.get("rpm")
    details = []
    if hp is not None:
        details.append(f"{hp}HP")
    if rpm is not None:
        details.append(f"{rpm}rpm")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"Load {idx}{suffix}"


def _detect_score_column(df: pd.DataFrame, score_variant: str) -> str:
    preferred = [
        f"transfer_score_{score_variant}",
        f"transfer_common_acc_{score_variant}",
        f"transfer_accuracy_{score_variant}",
        f"transfer_hscore_{score_variant}",
    ]
    for col in preferred:
        if col in df.columns:
            return col

    fallback_prefixes = (
        "transfer_score",
        "transfer_common_acc",
        "transfer_accuracy",
        "transfer_hscore",
    )
    for prefix in fallback_prefixes:
        for col in df.columns:
            if col.startswith(prefix):
                return col

    raise ValueError("Unable to find a transfer score column in the provided CSV.")


def _build_gap_matrix(csv_path: Path, dataset: str, score_variant: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"source", "target"}.issubset(df.columns):
        raise ValueError(f"CSV missing required columns source/target: {csv_path}")

    score_col = _detect_score_column(df, score_variant)
    work = df[["source", "target", score_col]].copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=[score_col])

    work["domain_gap"] = 1.0 - work[score_col]

    if dataset == "cwru":
        work["source"] = work["source"].map(_format_cwru_domain)
        work["target"] = work["target"].map(_format_cwru_domain)

    matrix = work.pivot_table(index="source", columns="target", values="domain_gap", aggfunc="mean")
    return matrix


def _plot_heatmap(ax: plt.Axes, matrix: pd.DataFrame, title: str, vmin: float, vmax: float) -> None:
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="mako",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Target domain")
    ax.set_ylabel("Source domain")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwru_csv", type=Path, default=Path("figures/cwru_transfer_ablation.csv"))
    parser.add_argument("--battery_csv", type=Path, default=Path("figures/battery_transfer_ablation.csv"))
    parser.add_argument(
        "--score_variant",
        choices=["base", "improved"],
        default="base",
        help="Use baseline or improved transfer score columns when available.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("figures"))
    parser.add_argument("--out_name", type=str, default="transfer_task_matrix_across_datasets.png")
    args = parser.parse_args()

    cwru_matrix = _build_gap_matrix(args.cwru_csv, dataset="cwru", score_variant=args.score_variant)
    battery_matrix = _build_gap_matrix(args.battery_csv, dataset="battery", score_variant=args.score_variant)

    all_values = np.concatenate([
        cwru_matrix.to_numpy(dtype=float).ravel(),
        battery_matrix.to_numpy(dtype=float).ravel(),
    ])
    all_values = all_values[~np.isnan(all_values)]
    vmin = float(np.min(all_values)) if all_values.size else 0.0
    vmax = float(np.max(all_values)) if all_values.size else 1.0

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    _plot_heatmap(
        axes[0],
        cwru_matrix,
        title="CWRU Transfer Task Matrix (Domain Gap = 1 - transfer score)",
        vmin=vmin,
        vmax=vmax,
    )
    _plot_heatmap(
        axes[1],
        battery_matrix,
        title="Battery Transfer Task Matrix (Domain Gap = 1 - transfer score)",
        vmin=vmin,
        vmax=vmax,
    )

    out_path = args.out_dir / args.out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    # Also export the matrix values for reporting tables.
    cwru_matrix.to_csv(args.out_dir / f"cwru_domain_gap_matrix_{args.score_variant}.csv")
    battery_matrix.to_csv(args.out_dir / f"battery_domain_gap_matrix_{args.score_variant}.csv")

    print(f"Saved heatmap figure to: {out_path}")


if __name__ == "__main__":
    main()