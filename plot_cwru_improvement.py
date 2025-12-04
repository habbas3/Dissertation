#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:29:44 2025

@author: habbas
"""

import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
DET_PATH  = "checkpoint/llm_run_20251123_135737/compare/deterministic_cnn_summary_1123_141215_CWRU_inconsistent.csv"
SNGP_PATH = "checkpoint/llm_run_20251123_135737/compare/sngp_wrn_sa_summary_1123_142720_CWRU_inconsistent.csv"
LLM_PATH  = "checkpoint/llm_run_20251123_135737/compare/llm_pick_summary_1123_140754_CWRU_inconsistent.csv"
OUT_FIG   = "A3_cwru_improvement.png"

# === Load ===
det  = pd.read_csv(DET_PATH)
sngp = pd.read_csv(SNGP_PATH)
llm  = pd.read_csv(LLM_PATH)

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
    det[imp_col].mean()*100,
    sngp[imp_col].mean()*100,
    llm[imp_col].mean()*100,
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
plt.savefig(OUT_FIG, dpi=300)
plt.close()
print(f"Saved {OUT_FIG}")
