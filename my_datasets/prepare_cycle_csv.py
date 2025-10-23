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

    # Summarise cycle coverage for each cell so downstream analysis can
    # understand how many charge/discharge cycles were recorded before
    # failure.  ``count`` represents the number of rows while ``max`` provides
    # the last cycle index which, for well-formed files, matches the EOL.
    per_cell = cycles.groupby("filename").agg(
        total_cycles=("cycle_number", "count"),
        eol_cycle=("cycle_number", "max"),
    )
    for meta_col in ["cathode", "batch", "cell_id"]:
        if meta_col in cycles.columns:
            per_cell[meta_col] = cycles.groupby("filename")[meta_col].first()

    counts = per_cell["eol_cycle"].rename("eol_cycle")
    print("\U0001F501 Total cycles per cell (max cycle index):\n", counts)

    # Persist a full table so researchers can inspect the cycle coverage for
    # every cell without having to re-run the script.
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    cycle_count_path = labels_csv.parent / "battery_cell_cycle_counts.csv"
    per_cell.reset_index().to_csv(cycle_count_path, index=False)
    print(f"Saved cycle-count summary to {cycle_count_path}")

    labels = per_cell.reset_index()[["filename", "eol_cycle"]]
    labels.sort_values("eol_cycle", inplace=True)
    
    # Normalise string columns up front so downstream lookups (e.g. cathode
    # grouping) do not have to worry about stray whitespace from the original
    # Argonne metadata.
    if "filename" in labels.columns:
        labels["filename"] = labels["filename"].astype(str).str.strip()
    

    if num_classes != 5:
        # Automatically generate labels if a different number requested
        qc = pd.qcut(labels["eol_cycle"], q=num_classes, labels=False, duplicates="drop")
        labels["eol_class"] = qc.map(lambda x: f"class_{int(x)}")
    else:
        labels["eol_class"] = pd.qcut(
            labels["eol_cycle"], q=num_classes, labels=LABEL_NAMES, duplicates="drop"
        )

    labels["eol_class_encoded"] = LabelEncoder().fit_transform(labels["eol_class"])

    labels.to_csv(labels_csv, index=False)
    print(f"Wrote per-cell labels to {labels_csv}")

    df = cycles.merge(labels, on="filename", how="left")
    df.sort_values(["filename", "cycle_number"], inplace=True)
    
    # Ensure all categorical/string columns are trimmed.  The raw parquet
    # files contain leading spaces in several fields which previously caused
    # cathode-family detection to miss many cells (e.g. " NMC532").
    string_cols = [
        "filename",
        "battery_name",
        "batch",
        "cell_id",
        "anode",
        "cathode",
        "electrolyte",
        "dataset_name",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

            

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