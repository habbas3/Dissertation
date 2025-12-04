#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:25:06 2025

@author: habbas
"""

import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
CSV_PATH = "my_datasets/Battery/battery_cycles_labeled.csv"
OUT_FIG = "A2_eol_class_distribution.png"

df = pd.read_csv(CSV_PATH)

# Try to infer EOL class column
eol_col = None
for c in ["eol_class_encoded", "eol_class"]:
    if c in df.columns:
        eol_col = c
        break

if eol_col is None:
    raise ValueError("Could not find eol_class column. Adjust name if needed.")

# === Count class occurrences ===
eol_counts = df[eol_col].value_counts().sort_index()

# === Plot ===
plt.figure(figsize=(6, 4))
eol_counts.plot(kind="bar")
plt.xlabel("EOL Class", fontweight='bold')
plt.ylabel("Number of Windows", fontweight='bold')
plt.title("Battery EOL Class Distribution", fontweight='bold')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()
print(f"Saved {OUT_FIG}")
