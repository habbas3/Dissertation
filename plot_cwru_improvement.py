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

from utils.plot_utils import find_latest_compare_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot average CWRU improvement by model.")
    parser.add_argument("--checkpoint_root", type=Path, default=Path("checkpoint"))
    parser.add_argument("--run_dir", type=Path, default=None, help="Optional llm_run_* directory to use.")
    parser.add_argument("--dataset_tag", default="CWRU_inconsistent")
    parser.add_argument("--out_fig", default="A3_cwru_improvement.png")
    args = parser.parse_args()

    det_path = find_latest_compare_csv(
        args.checkpoint_root,
        "deterministic_cnn_summary",
        args.dataset_tag,
        run_dir=args.run_dir,
    )
    sngp_path = find_latest_compare_csv(
        args.checkpoint_root,
        "sngp_wrn_sa_summary",
        args.dataset_tag,
        run_dir=args.run_dir,
    )
    llm_path = find_latest_compare_csv(
        args.checkpoint_root,
        "llm_pick_summary",
        args.dataset_tag,
        run_dir=args.run_dir,
    )

    # === Load ===
    det = pd.read_csv(det_path)
    sngp = pd.read_csv(sngp_path)
    llm = pd.read_csv(llm_path)

    # Infer improvement column name
    imp_col = None
    for c in ["improvement", "delta_common", "delta_metric"]:
        if c in det.columns:
            imp_col = c
            break

    if imp_col is None:
        raise ValueError("Could not find an 'improvement' column. Adjust if needed.")

    models = ["Deterministic CNN", "WRN+SA+SNGP", "LLM-picked CNN+OpenMax+SNGP"]
    improvements = [
        det[imp_col].mean() * 100,
        sngp[imp_col].mean() * 100,
        llm[imp_col].mean() * 100,
    ]

    # === Plot ===
    plt.figure(figsize=(7, 5))
    plt.bar(models, improvements)
    plt.ylabel("Average Improvement (%)", fontweight='bold')
    plt.title("CWRU Inconsistent: Average Improvement by Model", fontweight='bold')
    plt.xticks(rotation=15, ha="right")

    for i, v in enumerate(improvements):
        plt.text(i, v + 0.002, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=300)
    plt.close()
    print(f"Saved {args.out_fig}")
    
    
if __name__ == "__main__":
    main()