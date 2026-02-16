#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:29:44 2025

@author: habbas
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.plot_utils import find_latest_compare_csv_optional

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot average CWRU improvement by model.")
    parser.add_argument("--checkpoint_root", type=Path, default=Path("checkpoint"))
    parser.add_argument("--run_dir", type=Path, default=None, help="Optional llm_run_* directory to use.")
    parser.add_argument("--dataset_tag", default="CWRU_inconsistent")
    parser.add_argument("--out_fig", default="A3_cwru_improvement.png")
    args = parser.parse_args()

    summary_paths = {
        "Deterministic CNN": find_latest_compare_csv_optional(
        args.checkpoint_root,
        "deterministic_cnn_summary",
            args.dataset_tag,
            run_dir=args.run_dir,
        ),
        "WRN+SA+SNGP": find_latest_compare_csv_optional(
        args.checkpoint_root,
        "sngp_wrn_sa_summary",
            args.dataset_tag,
            run_dir=args.run_dir,
        ),
        "LLM-picked CNN+OpenMax+SNGP": find_latest_compare_csv_optional(
        args.checkpoint_root,
        "llm_pick_summary",
            args.dataset_tag,
            run_dir=args.run_dir,
        ),
    }

    models: list[str] = []
    improvements: list[float] = []
    for model_name, csv_path in summary_paths.items():
        if csv_path is None:
            print(f"Skipping {model_name}: no summary CSV found for {args.dataset_tag}.")
            continue
        summary = pd.read_csv(csv_path)
        imp_col = next((c for c in ["improvement", "delta_common", "delta_metric"] if c in summary.columns), None)
        if imp_col is None:
            print(f"Skipping {model_name}: no improvement column in {csv_path.name}.")
            continue
        models.append(model_name)
        improvements.append(float(pd.to_numeric(summary[imp_col], errors="coerce").dropna().mean() * 100.0))

    if not models:
        raise ValueError(f"No comparable CWRU summaries found for dataset tag {args.dataset_tag}.")

    # === Plot ===
    plt.figure(figsize=(7, 5))
    plt.bar(models, improvements)
    plt.ylabel("Average Improvement (%)", fontweight='bold')
    plt.title(f"{args.dataset_tag}: Average Improvement by Model", fontweight='bold')
    plt.xticks(rotation=15, ha="right")

    for i, v in enumerate(improvements):
        plt.text(i, v + 0.002, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=300)
    plt.close()
    print(f"Saved {args.out_fig}")
    
    
if __name__ == "__main__":
    main()