#!/usr/bin/env python3
"""
final_argonne_loader.py

Final script to load an Argonne HDF5 file that places 'raw_data/table'
and 'cycle_stats/table' at the top level (rather than inside one prefix).
"""

import json
import h5py
import numpy as np
import pandas as pd
from tables import File

from battdat.data import BatteryDataset, CellDataset
from battdat.schemas.column import RawData, CycleLevelData

###############################################################################
# 1) PATH TO YOUR FILE (adjust as needed)
###############################################################################
file_path = "/Users/moondiab/Documents/Dissertation/UDTL_Lable_Inconsistent-main/datasets/Battery/full_data/refined/batch_B10A_cell_3.h5"

###############################################################################
# 2) EXPECTED COLUMN NAMES (Argonne-like)
###############################################################################
EXPECTED_RAW_DATA = [
    "cycle_number",
    "file_number",
    "test_time",
    "state",
    "current",
    "voltage",
    "step_index",
    "method",
    "substep_index",
    "cycle_capacity",
    "cycle_energy",
]

EXPECTED_CYCLE_STATS = [
    "cycle_number",
    "energy_charge",
    "capacity_charge",
    "energy_discharge",
    "capacity_discharge",
    "cycle_start",
    "cycle_duration",
]

###############################################################################
# 3) FLATTENING: Convert structured arrays with multi-d fields into 1D columns
###############################################################################
def flatten_structured_array(struct_arr):
    """
    Convert a structured array (possibly with multi-dimensional fields)
    into a dict of 1D columns. e.g. "values_block_0" shape (N,5) => 5 new keys.
    """
    out = {}
    for field_name in struct_arr.dtype.names:
        data_block = struct_arr[field_name]
        if data_block.ndim > 1:
            # Flatten each subcolumn
            for i in range(data_block.shape[1]):
                col_key = f"{field_name}_{i}"
                out[col_key] = data_block[:, i]
        else:
            out[field_name] = data_block
    return out

def decode_byte_strings(df):
    """Convert any columns containing byte-strings to normal Python str."""
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    return df

###############################################################################
# 4) LOADING A GROUP’S TABLE MANUALLY
###############################################################################
def load_table_from_group(hdf5_path, group_name):
    """
    Loads `/group_name/table` from the HDF5 file, flattens multi-d columns,
    decodes bytes, and returns a DataFrame. If `index` col is present, drops it.
    """
    table_path = f"/{group_name}/table"

    with h5py.File(hdf5_path, "r") as f:
        if table_path not in f:
            print(f"Warning: {table_path} not found in {hdf5_path}.")
            return None
        struct_arr = f[table_path][:]
    
    # Flatten
    flat_dict = flatten_structured_array(struct_arr)
    df = pd.DataFrame(flat_dict)
    
    # Decode bytes
    df = decode_byte_strings(df)
    
    # Drop "index" column if present
    if "index" in df.columns:
        df.drop(columns="index", inplace=True, errors="ignore")
    
    return df

###############################################################################
# 5) RENAME COLUMNS TO EXPECTED
###############################################################################
def rename_columns_to_expected(df, expected_names):
    """
    If the df has the same number of columns, rename them in order to expected_names.
    Otherwise, print a warning.
    """
    if df is None:
        return None
    if len(df.columns) == len(expected_names):
        df.columns = expected_names
    else:
        print(f"WARNING: Found {len(df.columns)} columns, expected {len(expected_names)}.")
        print("Columns found:", df.columns.tolist())
    return df

###############################################################################
# 6) MAIN LOGIC
###############################################################################
if __name__ == "__main__":
    # Check file-level metadata + prefixes from Argonne code
    with File(file_path, "r") as fp:
        file_metadata, prefixes = BatteryDataset.inspect_hdf(fp)
        print("File-level metadata:", file_metadata)
        print("Available prefixes:", prefixes)
    
    # Load the "raw_data" and "cycle_stats" tables
    raw_data_df = load_table_from_group(file_path, "raw_data")
    cycle_stats_df = load_table_from_group(file_path, "cycle_stats")

    # Rename columns to Argonne's standard
    raw_data_df = rename_columns_to_expected(raw_data_df, EXPECTED_RAW_DATA)
    cycle_stats_df = rename_columns_to_expected(cycle_stats_df, EXPECTED_CYCLE_STATS)

    # Show a sample
    if raw_data_df is not None:
        print("\n**Raw Data (first 5 rows):**")
        print(raw_data_df.head())

    if cycle_stats_df is not None:
        print("\n**Cycle Stats (first 5 rows):**")
        print(cycle_stats_df.head())

    # Retrieve Argonne metadata
    meta_obj = BatteryDataset.get_metadata_from_hdf5(file_path)

    # Create CellDataset with the final data
    dataset = CellDataset(
        metadata=meta_obj,
        raw_data=raw_data_df,
        cycle_stats=cycle_stats_df,
    )

    print("\n**Constructed CellDataset**")
    print("CellDataset metadata:", dataset.metadata)
    if dataset.raw_data is not None:
        print("CellDataset raw_data columns:", list(dataset.raw_data.columns))
    if dataset.cycle_stats is not None:
        print("CellDataset cycle_stats columns:", list(dataset.cycle_stats.columns))
