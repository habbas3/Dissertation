#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:41:01 2025

@author: habbas
"""

import argparse
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from utils.plot_utils import find_latest_compare_csv_optional


def _load_model_frame(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    imp_col = next((c for c in ["improvement", "delta_common", "delta_metric"] if c in df.columns), None)
    if imp_col is None:
        raise ValueError(f"Missing improvement-like column in {csv_path}.")
    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError(f"Expected 'source' and 'target' columns in {csv_path}.")

    out = df[["source", "target", imp_col]].copy()
    out[imp_col] = pd.to_numeric(out[imp_col], errors="coerce") * 100.0
    out = out.dropna(subset=[imp_col])
    out["pair"] = out["source"].astype(str) + "→" + out["target"].astype(str)
    return out[["pair", imp_col]].rename(columns={imp_col: label})


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot battery comparison across models and transfer pairs.")
    parser.add_argument("--checkpoint_root", type=Path, default=Path("checkpoint"))
    parser.add_argument("--run_dir", type=Path, default=None, help="Optional llm_run_* directory to use.")
    parser.add_argument("--dataset_tag", default="Battery_inconsistent")
    parser.add_argument("--out_fig", default="A5_battery_combined_comparison.png")
    args = parser.parse_args()

    candidates = [
        ("Deterministic CNN", "deterministic_cnn", "deterministic_cnn_summary"),
        ("WRN+SA+SNGP", "wrn_sngp", "sngp_wrn_sa_summary"),
        ("LLM-picked CNN+SA", "llm_cnn_sa", "llm_pick_summary"),
    ]

    merged: pd.DataFrame | None = None
    plotted_models: list[tuple[str, str]] = []
    for display, key, prefix in candidates:
        csv_path = find_latest_compare_csv_optional(
            args.checkpoint_root,
            prefix,
            args.dataset_tag,
            run_dir=args.run_dir,
        )
        if csv_path is None:
            print(f"Skipping {display}: no summary CSV found for {args.dataset_tag}.")
            continue
        frame = _load_model_frame(csv_path, key)
        merged = frame if merged is None else merged.merge(frame, on="pair", how="outer")
        plotted_models.append((display, key))

    if merged is None or merged.empty:
        raise ValueError(f"No battery summaries available for dataset tag {args.dataset_tag}.")

    merged = merged.sort_values("pair").reset_index(drop=True)
    x = np.arange(len(merged))
    width = 0.8 / max(1, len(plotted_models))

    plt.figure(figsize=(10, 7))
    for idx, (display, key) in enumerate(plotted_models):
        offset = (idx - (len(plotted_models) - 1) / 2.0) * width
        plt.bar(x + offset, merged[key], width=width, label=display)
    plt.xticks(list(x), merged["pair"], rotation=45, ha="right")
    plt.ylabel("Improvement (%)", fontweight='bold')
    plt.title(f"{args.dataset_tag}: Improvement by Model and Transfer Pair", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=300)
    plt.close()
    print(f"Saved {args.out_fig}")



if __name__ == "__main__":
    main()
