#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:41:01 2025

@author: habbas
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.plot_utils import find_latest_compare_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot battery comparison across models and transfer pairs.")
    parser.add_argument("--checkpoint_root", type=Path, default=Path("checkpoint"))
    parser.add_argument("--run_dir", type=Path, default=None, help="Optional llm_run_* directory to use.")
    parser.add_argument("--dataset_tag", default="Battery_inconsistent")
    parser.add_argument("--out_fig", default="A5_battery_combined_comparison.png")
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

    det_b = pd.read_csv(det_path)
    det_b.improvement = det_b.improvement * 100
    sngp_b = pd.read_csv(sngp_path)
    sngp_b.improvement = sngp_b.improvement * 100
    llm_b = pd.read_csv(llm_path)
    llm_b.improvement = llm_b.improvement * 100

    # Infer improvement columns
    imp_det_col = None
    for c in ["improvement", "delta_common", "delta_metric"]:
        if c in det_b.columns:
            imp_det_col = c
            break

    imp_llm_col = None
    for c in ["improvement", "delta_common", "delta_metric"]:
        if c in llm_b.columns:
            imp_llm_col = c
            break

    if imp_det_col is None or imp_llm_col is None:
        raise ValueError("Missing improvement column in one of the battery summary files.")

    # Ensure source & target exist
    for df in (det_b, sngp_b, llm_b):
        if "source" not in df.columns or "target" not in df.columns:
            raise ValueError("Expected 'source' and 'target' columns missing in one of the files.")
        df["pair"] = df["source"].astype(str) + "→" + df["target"].astype(str)

    # Merge per pair
    merged = det_b[["pair", imp_det_col]].rename(columns={imp_det_col: "det_cnn"})
    merged = merged.merge(
        sngp_b[["pair", imp_det_col]].rename(columns={imp_det_col: "wrn_sngp"}),
        on="pair", how="outer"
    ).merge(
        llm_b[["pair", imp_llm_col]].rename(columns={imp_llm_col: "llm_cnn_sa"}),
        on="pair", how="outer"
    )

    x = range(len(merged))
    width = 0.25

    plt.figure(figsize=(10, 7))
    plt.bar([i - width for i in x], merged["det_cnn"], width=width, label="Deterministic CNN")
    plt.bar(x, merged["wrn_sngp"], width=width, label="WRN+SA+SNGP")
    plt.bar([i + width for i in x], merged["llm_cnn_sa"], width=width, label="LLM-picked CNN+SA")
    plt.xticks(list(x), merged["pair"], rotation=45, ha="right")
    plt.ylabel("Improvement (%)", fontweight='bold')
    plt.title("Battery Transfers: Improvement by Model and Transfer Pair", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=300)
    plt.close()
    print(f"Saved {args.out_fig}")



if __name__ == "__main__":
    main()
