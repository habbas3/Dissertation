#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:33:16 2025

@author: habbas
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.plot_utils import find_latest_compare_csv_optional

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LLM improvement per battery transfer pair.")
    parser.add_argument("--checkpoint_root", type=Path, default=Path("checkpoint"))
    parser.add_argument("--run_dir", type=Path, default=None, help="Optional llm_run_* directory to use.")
    parser.add_argument("--dataset_tag", default="Battery_inconsistent")
    parser.add_argument("--out_fig", default="A4_battery_llm_improvement_per_pair.png")
    args = parser.parse_args()

    llm_path = find_latest_compare_csv_optional(
        args.checkpoint_root,
        "llm_pick_summary",
        args.dataset_tag,
        run_dir=args.run_dir,
    )
    if llm_path is None:
        raise ValueError(f"No llm_pick_summary CSV found for dataset {args.dataset_tag}.")

    llm_b = pd.read_csv(llm_path)

    imp_col = next((c for c in ["improvement", "delta_common", "delta_metric"] if c in llm_b.columns), None)


    if imp_col is None:
        raise ValueError("Could not find an improvement column in LLM battery summary.")

    llm_b[imp_col] = pd.to_numeric(llm_b[imp_col], errors="coerce") * 100.0
    if "source" not in llm_b.columns or "target" not in llm_b.columns:
        raise ValueError("Expected 'source' and 'target' columns are missing.")
        
        
    llm_b = llm_b.dropna(subset=[imp_col]).copy()
    llm_b["pair"] = llm_b["source"].astype(str) + "→" + llm_b["target"].astype(str)
    
    
    plt.figure(figsize=(8, 6))
    plt.bar(llm_b["pair"], llm_b[imp_col])
    plt.ylabel("Improvement (%)", fontweight='bold')
    plt.title("Battery Transfers: LLM-picked Improvement", fontweight='bold')
    plt.xticks(rotation=45, ha="right")

    for i, v in enumerate(llm_b[imp_col]):
        plt.text(i, v + 0.002, f"{v:.2f}", ha="center", va="bottom")
        
        
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=300)
    plt.close()
    print(f"Saved {args.out_fig}")


if __name__ == "__main__":
    main()
