#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boxplot of per-cell EOL cycle counts grouped by cathode, with color legend.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# === Config ===
CSV_PATH = "my_datasets/Battery/battery_cycles_labeled.csv"
OUT_FIG = "A1_cycle_counts_per_cathode_boxplot.png"

# === Load data ===
df = pd.read_csv(CSV_PATH)

# Identify cycle column
cycle_col = None
for c in ["cycle_number", "cycle_idx", "cycle"]:
    if c in df.columns:
        cycle_col = c
        break
if cycle_col is None:
    raise ValueError("No cycle column found. Update column name.")

# Check required columns
if "cathode" not in df.columns or "filename" not in df.columns:
    raise ValueError("Missing required columns: 'cathode', 'filename'.")

# === Compute per-cell cycle count ===
per_cell_cycles = (
    df.groupby(["cathode", "filename"])[cycle_col]
      .max()
      .reset_index()
      .rename(columns={cycle_col: "cycle_count"})
)

# Order cathodes by median lifetime
cathode_order = (
    per_cell_cycles.groupby("cathode")["cycle_count"]
    .median()
    .sort_values(ascending=False)
    .index.tolist()
)

# === Prepare Colors ===
# Distinct colors per cathode
colors = plt.cm.Set3(range(len(cathode_order)))  # nice pastel palette

# === Plot ===
fig, ax = plt.subplots(figsize=(11, 5))

# Generate boxplots manually per cathode
boxplots = []

for i, cat in enumerate(cathode_order):
    data = per_cell_cycles[per_cell_cycles["cathode"] == cat]["cycle_count"]
    bp = ax.boxplot(
        data,
        positions=[i],
        patch_artist=True,
        widths=0.6
    )
    # Color the box
    for patch in bp['boxes']:
        patch.set_facecolor(colors[i])
        patch.set_edgecolor('black')
    for whisker in bp['whiskers']:
        whisker.set_color('black')
    for cap in bp['caps']:
        cap.set_color('black')
    for median in bp['medians']:
        median.set_color('black')

    boxplots.append(bp)

# Axes labels
ax.set_title("Distribution of EOL Cycle Counts per Cathode", fontweight='bold')
ax.set_ylabel("Cycle Count at EOL", fontweight='bold')
ax.set_xticks(range(len(cathode_order)))
ax.set_xticklabels(cathode_order, rotation=45, ha="right", fontsize=9)

# === Legend ===
legend_handles = [
    mpatches.Patch(facecolor=colors[i], edgecolor='black', label=cathode_order[i])
    for i in range(len(cathode_order))
]

ax.legend(handles=legend_handles, title="Cathode Type", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()

print(f"Saved {OUT_FIG}")
