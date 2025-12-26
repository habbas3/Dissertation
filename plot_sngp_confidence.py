#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot SNGP confidence/entropy summaries for EOL classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _load_uncertainty_records(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in root.rglob("sngp_uncertainty_target_val_summary.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        rel = path.parent.relative_to(root)
        records.append(
            {
                "run": str(rel),
                "mean_entropy": payload.get("mean_entropy"),
                "mean_entropy_known": payload.get("mean_entropy_known"),
                "mean_entropy_outlier": payload.get("mean_entropy_outlier"),
                "mean_max_prob_known": payload.get("mean_max_prob_known"),
                "mean_max_prob_outlier": payload.get("mean_max_prob_outlier"),
            }
        )
    return pd.DataFrame.from_records(records)


def _plot_entropy(df: pd.DataFrame, out_path: Path, topn: int) -> None:
    scoped = df.nsmallest(topn, "mean_entropy") if topn > 0 else df
    plt.figure(figsize=(10, 4))
    plt.bar(scoped["run"], scoped["mean_entropy"], color="indigo")
    plt.ylabel("Mean predictive entropy (target_val)")
    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved entropy bar chart to {out_path}")


def _plot_confidence(df: pd.DataFrame, out_path: Path, topn: int) -> None:
    scoped = df.copy()
    if topn > 0:
        scoped = scoped.nsmallest(topn, "mean_entropy")
    melted = scoped.melt(
        id_vars="run",
        value_vars=["mean_max_prob_known", "mean_max_prob_outlier"],
        var_name="split",
        value_name="probability",
    )
    plt.figure(figsize=(10, 5))
    for split, color in [("mean_max_prob_known", "seagreen"), ("mean_max_prob_outlier", "tomato")]:
        chunk = melted[melted["split"] == split]
        plt.plot(chunk["run"], chunk["probability"], marker="o", label=split.replace("mean_max_prob_", ""), color=color)
    plt.ylabel("Mean max-softmax probability")
    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confidence trend plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("checkpoint"),
        help="Checkpoint root to scan for sngp_uncertainty_target_val_summary.json",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=20,
        help="Limit plots to the lowest-entropy runs (use 0 to show all).",
    )
    args = parser.parse_args()

    df = _load_uncertainty_records(args.root)
    if df.empty:
        raise SystemExit(f"No SNGP uncertainty summaries found under {args.root}")

    out_dir = args.root if args.root.is_dir() else args.root.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "sngp_uncertainty_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved combined uncertainty table to {csv_path}")

    _plot_entropy(df, out_dir / "sngp_entropy.png", args.topn)
    _plot_confidence(df, out_dir / "sngp_confidence.png", args.topn)


if __name__ == "__main__":
    main()