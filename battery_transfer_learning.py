#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 14:27:23 2025

@author: habbas
"""

"""Simple example comparing baseline vs. transfer learning using the labelled
battery dataset.

Usage::
    python battery_transfer_learning.py --csv my_datasets/Battery/battery_data_labeled.csv \
        --target-cathode NMC532
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


FEATURE_COLS = [
    "cycle_number",
    "energy_charge",
    "capacity_charge",
    "energy_discharge",
    "capacity_discharge",
    "cycle_start",
    "cycle_duration",
]


def run_experiment(csv_path: str, target_cathode: str) -> tuple[float, float]:
    """Return baseline and transfer accuracies for ``target_cathode``."""

    df = pd.read_csv(csv_path)

    source_df = df[df["cathode"].str.strip() != target_cathode]
    target_df = df[df["cathode"].str.strip() == target_cathode]

    X_source = source_df[FEATURE_COLS].values
    y_source = source_df["eol_class_encoded"].values

    X_target = target_df[FEATURE_COLS].values
    y_target = target_df["eol_class_encoded"].values

    X_tgt_train, X_tgt_test, y_tgt_train, y_tgt_test = train_test_split(
        X_target,
        y_target,
        test_size=0.3,
        random_state=42,
        stratify=y_target,
    )

    scaler = StandardScaler().fit(X_source)
    X_source = scaler.transform(X_source)
    X_tgt_train = scaler.transform(X_tgt_train)
    X_tgt_test = scaler.transform(X_tgt_test)

    baseline_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_clf.fit(X_tgt_train, y_tgt_train)
    baseline_pred = baseline_clf.predict(X_tgt_test)
    baseline_acc = accuracy_score(y_tgt_test, baseline_pred)

    transfer_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_combined = pd.concat(
        [pd.DataFrame(X_source, columns=FEATURE_COLS), pd.DataFrame(X_tgt_train, columns=FEATURE_COLS)]
    )
    y_combined = pd.concat([pd.Series(y_source), pd.Series(y_tgt_train)])
    transfer_clf.fit(X_combined, y_combined)
    transfer_pred = transfer_clf.predict(X_tgt_test)
    transfer_acc = accuracy_score(y_tgt_test, transfer_pred)

    return baseline_acc, transfer_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Battery transfer learning demo")
    parser.add_argument(
        "--csv",
        type=str,
        default="my_datasets/Battery/battery_data_labeled.csv",
        help="Path to labelled battery CSV",
    )
    parser.add_argument(
        "--target-cathode",
        type=str,
        default="NMC532",
        help="Cathode type to treat as target",
    )

    args = parser.parse_args()

    baseline_acc, transfer_acc = run_experiment(args.csv, args.target_cathode)

    print(f"Target cathode: {args.target_cathode}")
    print(f"Baseline accuracy (target only): {baseline_acc:.4f}")
    print(f"Transfer learning accuracy: {transfer_acc:.4f}")


if __name__ == "__main__":
    main()