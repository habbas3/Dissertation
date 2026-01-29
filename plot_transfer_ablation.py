#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot transfer-pair ablations for CWRU and Battery datasets.

This script compares a *baseline* strategy against an improved
per-transfer strategy (e.g., LLM-guided or chemistry-aware). It produces:

* A paired bar plot per transfer pair (baseline vs. improved).
* A delta plot that highlights the per-pair gain/loss.
* A CSV summary with the merged scores for reproducible ablation tables.

Typical usage with real experiment outputs::

    python plot_transfer_ablation.py \
        --dataset cwru \
        --baseline checkpoint/.../deterministic_cnn_summary.csv \
        --improved checkpoint/.../llm_pick_summary.csv
        
For quick experimentation (e.g., when running this file directly from an IDE),
you can also run a synthetic demo without passing any arguments::

    python plot_transfer_ablation.py --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from utils.plot_utils import find_latest_compare_csv

_CWRU_LOADS = {
    0: {"hp": 0, "rpm": 1797},
    1: {"hp": 1, "rpm": 1772},
    2: {"hp": 2, "rpm": 1750},
    3: {"hp": 3, "rpm": 1730},
}


def _detect_transfer_column(df: pd.DataFrame) -> str:
    for candidate in ("transfer_score", "transfer_common_acc", "transfer_accuracy", "transfer_hscore"):
        for col in df.columns:
            if col.startswith(candidate):
                return col
    raise ValueError(
        "No transfer score column found (expected transfer_score/transfer_common_acc/transfer_accuracy/transfer_hscore)."
    )


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


def _format_cwru_load(idx: int) -> str:
    """Format a CWRU load label, tolerating non-numeric identifiers."""

    try:
        idx_int = int(idx)
    except (TypeError, ValueError):
        # Fall back to the raw identifier (useful for synthetic/demo inputs).
        return str(idx)

    meta = _CWRU_LOADS.get(idx_int, {})
    hp = meta.get("hp")
    rpm = meta.get("rpm")
    parts = [f"Load {idx_int}"]
    if hp is not None:
        parts.append(f"{hp} HP")
    if rpm is not None:
        parts.append(f"{rpm} rpm")
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} ({', '.join(parts[1:])})"


def _make_pair_label(series: Iterable, dataset: str) -> list[str]:
    labels = []
    for s, t in series:
        if dataset == "cwru":
            labels.append(f"{_format_cwru_load(s)}→{_format_cwru_load(t)}")
        else:
            labels.append(f"{s}→{t}")
    return labels

def _demo_transfer_frames(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build small synthetic transfer tables for quick, argument-free demos."""

    pairs = [
        ("A", "B", 0.78, 0.82),
        ("A", "C", 0.74, 0.77),
        ("B", "C", 0.69, 0.75),
        ("C", "D", 0.72, 0.80),
    ]
    baseline = pd.DataFrame(pairs, columns=["source", "target", "transfer_score", "_improved"])[
        ["source", "target", "transfer_score"]
    ]
    improved = pd.DataFrame(pairs, columns=["source", "target", "_base", "transfer_score"])[
        ["source", "target", "transfer_score"]
    ]

    # Slightly tweak the numbers based on dataset for variety.
    if dataset == "battery":
        improved["transfer_score"] += 0.01
        baseline["transfer_score"] -= 0.005

    return baseline, improved

def plot_transfer_comparison(df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    score_col_base = _detect_transfer_column(df.filter(regex="_base$", axis=1)).replace("_base", "")
    score_col_improved = score_col_base + "_improved"
    score_col_base_full = score_col_base + "_base"

    df = df.copy()
    if dataset == "cwru":
        df[score_col_base_full] = pd.to_numeric(df[score_col_base_full], errors="coerce")
        df[score_col_improved] = pd.to_numeric(df[score_col_improved], errors="coerce")
        df = df.dropna(subset=[score_col_base_full, score_col_improved])
        if df.empty:
            raise ValueError("No CWRU transfer results found after filtering empty transfers.")
    df["pair"] = _make_pair_label(zip(df["source"], df["target"]), dataset)
    df["delta"] = df[score_col_improved] - df[score_col_base_full]

    # --- paired bar plot ---
    plt.figure(figsize=(10, 6))
    x = range(len(df))
    width = 0.4
    plt.bar([i - width / 2 for i in x], df[score_col_base_full] * 100, width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], df[score_col_improved] * 100, width=width, label="Improved")
    plt.xticks(ticks=list(x), labels=df["pair"], rotation=45, ha="right")
    plt.ylabel("Accuracy (%)", fontweight="bold")
    plt.title(f"{dataset}: per-transfer performance comparison", fontweight="bold")
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
    plt.ylabel("Δ Improved – Baseline (score %)", fontweight="bold")
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
    default_results_file = Path(
        "/Users/moondiab/Documents/Dissertation/UDTL_Lable_Inconsistent-main/checkpoint/"
        "llm_run_20260126_213942/compare/cycles_5_summary_0128_023915_CWRU_inconsistent"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["cwru", "battery"],
        default="cwru",
        help="Dataset name for labelling outputs.",
    )
    parser.add_argument("--checkpoint_root", type=Path, default=Path("checkpoint"))
    parser.add_argument("--run_dir", type=Path, default=None, help="Optional llm_run_* directory to use.")
    parser.add_argument("--baseline", type=Path, default=None, help="CSV with baseline transfer results.")
    parser.add_argument("--improved", type=Path, default=None, help="CSV with improved/LLM transfer results.")
    parser.add_argument(
        "--results_file",
        type=Path,
        default=default_results_file,
        help=(
            "CSV path for the improved/ablation results. If set, the baseline defaults to a "
            "deterministic_cnn_summary CSV in the same compare directory."
        ),
    )
    parser.add_argument("--demo", action="store_true", help="Run with synthetic data instead of CSV inputs.")
    parser.add_argument("--out_dir", type=Path, default=Path("figures"), help="Directory to save plots and CSV.")
    args = parser.parse_args()

    # Require an explicit choice between real CSVs and the synthetic demo. This prevents
    # accidental demo plots when users run the script without arguments from an IDE.
    
    baseline_path = getattr(args, "baseline", None)
    improved_path = getattr(args, "improved", None)
    dataset_tag = "CWRU_inconsistent" if args.dataset == "cwru" else "Battery_inconsistent"

    try:
        if args.demo:
            base_df, imp_df = _demo_transfer_frames(args.dataset)
        else:
            if args.results_file is not None and improved_path is None:
                improved_path = _resolve_csv_path(args.results_file)
                if not improved_path.exists():
                    print(f"Results file not found: {improved_path}. Exiting without plotting.")
                    return
                compare_dir = improved_path.parent
                if compare_dir.name != "compare" and (compare_dir / "compare").exists():
                    compare_dir = compare_dir / "compare"
                if baseline_path is None:
                    baseline_path = _find_compare_csv_in_dir(
                        compare_dir,
                        "deterministic_cnn_summary",
                        dataset_tag,
                    )
                    
            if baseline_path is None:
                baseline_path = find_latest_compare_csv(
                    args.checkpoint_root,
                    "deterministic_cnn_summary",
                    dataset_tag,
                    run_dir=args.run_dir,
                )
            if improved_path is None:
                improved_path = find_latest_compare_csv(
                    args.checkpoint_root,
                    "llm_pick_summary",
                    dataset_tag,
                    run_dir=args.run_dir,
                )
        base_df = pd.read_csv(_resolve_csv_path(baseline_path))
        imp_df = pd.read_csv(_resolve_csv_path(improved_path))
        merged = _merge_pairs(base_df, imp_df)
        plot_transfer_comparison(merged, args.dataset, args.out_dir)
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
        print(f"Transfer ablation failed safely: {exc}")
        return


if __name__ == "__main__":
    main()