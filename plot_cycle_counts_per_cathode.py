#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:21:07 2025

@author: habbas
"""

import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
CSV_PATH = "my_datasets/Battery/battery_cycles_labeled.csv"
OUT_FIG = "A1_cycle_counts_per_cathode.png"

# === Load data ===
df = pd.read_csv(CSV_PATH)

# Try to infer cycle column (adjust if yours is named differently)
cycle_col = None
for c in ["cycle_number", "cycle_idx", "cycle"]:
    if c in df.columns:
        cycle_col = c
        break

if cycle_col is None:
    raise ValueError("Could not find a cycle column. Please adjust the column name.")

if "cathode" not in df.columns or "filename" not in df.columns:
    raise ValueError("Expected 'cathode' and 'filename' columns are missing.")

# === Compute per-cell cycle count (max cycle per file) ===
per_cell_cycles = (
    df.groupby(["cathode", "filename"])[cycle_col]
      .max()
      .reset_index()
      .rename(columns={cycle_col: "cycle_count"})
)

# === Aggregate median cycles per cathode ===
agg_cycles = (
    per_cell_cycles.groupby("cathode")["cycle_count"]
                   .median()
                   .sort_values(ascending=False)
)

# === Plot ===
plt.figure(figsize=(8, 4))
agg_cycles.plot(kind="bar")
plt.ylabel("Median Cycles per Cell", fontweight='bold')
plt.xlabel("Cathode", fontweight='bold')
plt.title("Median EOL Cycle Count per Cathode", fontweight='bold')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()
print(f"Saved {OUT_FIG}")
