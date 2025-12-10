#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:22:38 2025

@author: habbas
"""

#!/usr/bin/env python3
"""Visualise LLM ablation comparisons (history, cycle limits, head toggles)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _find_latest_llm_run(root: Path) -> Path | None:
    runs = sorted(root.glob("llm_run_*/llm_leaderboard.csv"))
    if not runs:
        return None
    return runs[-1].parent


def _load_ablation_json(run_dir: Path) -> list[dict]:
    path = run_dir / "llm_ablation.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def plot_leaderboard(df: pd.DataFrame, out_path: Path) -> None:
    ordered = df.sort_values("avg_improvement", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(ordered["tag"], ordered["avg_improvement"], color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Avg. improvement vs. baseline")
    plt.title("LLM comparison + ablation leaderboard")
    for i, v in enumerate(ordered["avg_improvement"]):
        plt.text(i, v + 0.001, f"{v:+.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cycle_limits(df: pd.DataFrame, out_path: Path) -> None:
    scoped = df.dropna(subset=["cycle_limit"])  # type: ignore[arg-type]
    if scoped.empty:
        return
    plt.figure(figsize=(8, 4))
    scoped = scoped.sort_values("cycle_limit")
    plt.plot(scoped["cycle_limit"], scoped["avg_improvement"], marker="o")
    plt.xlabel("Cycle horizon exposed")
    plt.ylabel("Avg. improvement vs. baseline")
    plt.title("Early-cycle EOL ablation")
    for x, y in zip(scoped["cycle_limit"], scoped["avg_improvement"]):
        plt.text(x, y, f"{y:+.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("checkpoint"),
        help="Path to an llm_run_* folder or the checkpoint root (auto-picks latest run).",
    )
    args = parser.parse_args()

    run_dir: Path
    if args.run_dir.name.startswith("llm_run_") and args.run_dir.is_dir():
        run_dir = args.run_dir
    else:
        latest = _find_latest_llm_run(args.run_dir)
        if latest is None:
            raise SystemExit("No llm_run_* leaderboard found under the provided path.")
        run_dir = latest

    leaderboard_path = run_dir / "llm_leaderboard.csv"
    if not leaderboard_path.exists():
        raise SystemExit(f"Missing leaderboard at {leaderboard_path}")

    df = pd.read_csv(leaderboard_path)
    ablation_records = _load_ablation_json(run_dir)
    if ablation_records:
        print(f"Loaded {len(ablation_records)} ablation prompt variants from {run_dir}")

    lb_plot = run_dir / "ablation_leaderboard.png"
    plot_leaderboard(df, lb_plot)
    print(f"Saved leaderboard plot to {lb_plot}")

    cycle_plot = run_dir / "ablation_cycles.png"
    plot_cycle_limits(df, cycle_plot)
    if cycle_plot.exists():
        print(f"Saved cycle-ablation plot to {cycle_plot}")


if __name__ == "__main__":
    main()