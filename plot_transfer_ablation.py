#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot transfer-pair ablations for CWRU and Battery datasets.

This script compares a *baseline* strategy against an improved
per-transfer strategy (e.g., LLM-guided or chemistry-aware). It produces:

* A paired bar plot per transfer pair (baseline vs. improved).
* A delta plot that highlights the per-pair gain/loss.
* A CSV summary with the merged scores for reproducible ablation tables.

Example usage:

    python plot_transfer_ablation.py \
        --dataset cwru \
        --baseline checkpoint/.../deterministic_cnn_summary.csv \
        --improved checkpoint/.../llm_pick_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def _detect_improvement_column(df: pd.DataFrame) -> str:
    for candidate in ("improvement", "delta_common", "delta_metric"):
        for col in df.columns:
            if col.startswith(candidate):
                return col
    raise ValueError("No improvement column found (expected improvement/delta_common/delta_metric).")


def _merge_pairs(baseline: pd.DataFrame, improved: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"source", "target"}
    if not required_cols.issubset(baseline.columns) or not required_cols.issubset(improved.columns):
        missing = required_cols - set(baseline.columns) - set(improved.columns)
        raise ValueError(f"Missing columns for merge: {sorted(missing)}")

    merge_keys = ["source", "target"]
    merged = baseline.merge(
        improved,
        on=merge_keys,
        how="inner",
        suffixes=("_base", "_improved"),
    )
    if merged.empty:
        raise ValueError("No overlapping transfer pairs between baseline and improved CSVs.")
    return merged


def _make_pair_label(series: Iterable) -> list[str]:
    return [f"{s}→{t}" for s, t in series]


def plot_transfer_comparison(df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    imp_col_base = _detect_improvement_column(df.filter(regex="_base$", axis=1)).replace("_base", "")
    imp_col_improved = imp_col_base + "_improved"
    imp_col_base_full = imp_col_base + "_base"

    df = df.copy()
    df["pair"] = _make_pair_label(zip(df["source"], df["target"]))
    df["delta"] = df[imp_col_improved] - df[imp_col_base_full]

    # --- paired bar plot ---
    plt.figure(figsize=(10, 6))
    x = range(len(df))
    width = 0.4
    plt.bar([i - width / 2 for i in x], df[imp_col_base_full] * 100, width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], df[imp_col_improved] * 100, width=width, label="Improved")
    plt.xticks(ticks=list(x), labels=df["pair"], rotation=45, ha="right")
    plt.ylabel("Improvement (%)", fontweight="bold")
    plt.title(f"{dataset}: per-transfer improvement comparison", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    paired_path = out_dir / f"{dataset}_transfer_comparison.png"
    plt.savefig(paired_path, dpi=300)
    plt.close()
    print(f"Saved {paired_path}")

    # --- delta plot ---
    ordered = df.sort_values("delta", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(ordered["pair"], ordered["delta"] * 100, color="seagreen")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Δ Improved – Baseline (%)", fontweight="bold")
    plt.title(f"{dataset}: per-transfer gain/loss (ablation)", fontweight="bold")
    for i, v in enumerate(ordered["delta"] * 100):
        plt.text(i, v + (0.4 if v >= 0 else -0.6), f"{v:+.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    delta_path = out_dir / f"{dataset}_transfer_delta.png"
    plt.savefig(delta_path, dpi=300)
    plt.close()
    print(f"Saved {delta_path}")

    # --- CSV summary ---
    summary_path = out_dir / f"{dataset}_transfer_ablation.csv"
    df.to_csv(summary_path, index=False)
    print(f"Saved merged ablation CSV to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cwru", "battery"], help="Dataset name for labelling outputs.")
    parser.add_argument("--baseline", type=Path, required=True, help="CSV with baseline transfer results.")
    parser.add_argument("--improved", type=Path, required=True, help="CSV with improved/LLM transfer results.")
    parser.add_argument("--out_dir", type=Path, default=Path("figures"), help="Directory to save plots and CSV.")
    args = parser.parse_args()

    base_df = pd.read_csv(args.baseline)
    imp_df = pd.read_csv(args.improved)
    merged = _merge_pairs(base_df, imp_df)
    plot_transfer_comparison(merged, args.dataset, args.out_dir)


if __name__ == "__main__":
    main()