#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:33:16 2025

@author: habbas
"""

import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
LLM_BATT_PATH = "checkpoint/llm_run_20251123_130533/compare/llm_pick_summary_1123_130747_Battery_inconsistent.csv"
OUT_FIG       = "A4_battery_llm_improvement_per_pair.png"

llm_b = pd.read_csv(LLM_BATT_PATH)
llm_b.improvement = llm_b.improvement*100

# Infer improvement col
imp_col = None
for c in ["improvement", "delta_common", "delta_metric"]:
    if c in llm_b.columns:
        imp_col = c
        break

if imp_col is None:
    raise ValueError("Could not find an improvement column in LLM battery summary.")

# Make "source→target" label
if "source" not in llm_b.columns or "target" not in llm_b.columns:
    raise ValueError("Expected 'source' and 'target' columns are missing.")

llm_b["pair"] = llm_b["source"].astype(str) + "→" + llm_b["target"].astype(str)

# === Plot ===
plt.figure(figsize=(8, 6))
plt.bar(llm_b["pair"], llm_b[imp_col])
plt.ylabel("Improvement (%)", fontweight='bold')
plt.title("Battery Transfers: LLM-picked CNN+SA (SNGP) Improvement", fontweight='bold')
plt.xticks(rotation=45, ha="right")

for i, v in enumerate(llm_b[imp_col]):
    plt.text(i, v + 0.002, f"{v:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()
print(f"Saved {OUT_FIG}")
