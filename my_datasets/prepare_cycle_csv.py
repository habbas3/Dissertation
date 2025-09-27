#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 2 13:41:23 2025

@author: habbas
"""

#!/usr/bin/env python3
"""Generate per-cell and cycle-level CSVs with new 5-class lifetime labels."""
from pathlib import Path
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LABEL_NAMES = [
    "short life",
    "short-mid life",
    "mid life",
    "mid-long life",
    "long life",
]


def build_cycle_csv(processed_dir: Path, labels_csv: Path, cycles_csv: Path, num_classes: int = 5) -> None:
    """Compute lifetime classes from processed parquet data and merge with cycles.

    Parameters
    ----------
    processed_dir : Path
        Directory containing ``merged_battery_data.parquet``.
    labels_csv : Path
        Destination path for per-cell labels.
    cycles_csv : Path
        Destination path for cycle-level CSV with labels merged.
    num_classes : int, optional
        Number of lifetime classes to create (default: 5).
    """
    processed_dir = Path(processed_dir)
    labels_csv = Path(labels_csv)
    cycles_csv = Path(cycles_csv)

    merged_path = processed_dir / "merged_battery_data.parquet"
    if not merged_path.exists():
        raise FileNotFoundError(f"{merged_path} not found")

    print(f"Loading merged cycles from {merged_path}")
    cycles = pd.read_parquet(merged_path)

    # Total cycles per cell
    counts = cycles.groupby("filename")["cycle_number"].max().rename("eol_cycle")
    print("\U0001F501 Total cycles per cell:\n", counts)

    labels = counts.reset_index()
    labels.sort_values("eol_cycle", inplace=True)

    if num_classes != 5:
        # Automatically generate labels if a different number requested
        qc = pd.qcut(labels["eol_cycle"], q=num_classes, labels=False, duplicates="drop")
        labels["eol_class"] = qc.map(lambda x: f"class_{int(x)}")
    else:
        labels["eol_class"] = pd.qcut(
            labels["eol_cycle"], q=num_classes, labels=LABEL_NAMES, duplicates="drop"
        )

    labels["eol_class_encoded"] = LabelEncoder().fit_transform(labels["eol_class"])

    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_csv, index=False)
    print(f"Wrote per-cell labels to {labels_csv}")

    df = cycles.merge(labels, on="filename", how="left")
    df.sort_values(["filename", "cycle_number"], inplace=True)

    cycles_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cycles_csv, index=False)
    print(f"Wrote cycle-level data with labels to {cycles_csv} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build battery CSVs with lifetime labels")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="Battery/proper_hdf5/processed",
        help="Directory containing processed parquet files",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="Battery/battery_labeled.csv",
        help="Output path for per-cell label CSV",
    )
    parser.add_argument(
        "--cycles_csv",
        type=str,
        default="Battery/battery_cycles_labeled.csv",
        help="Output path for cycle-level CSV",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of lifetime classes to create",
    )
    args = parser.parse_args()
    build_cycle_csv(args.processed_dir, args.labels_csv, args.cycles_csv, num_classes=args.num_classes)