#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 16:51:16 2025

@author: habbas
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class BatterySequenceDataset(Dataset):
    """Dataset returning contiguous cycle sequences and discrete labels.

    Parameters
    ----------
    df:
        DataFrame containing cycles for a single cathode chemistry.
    sequence_length:
        Number of consecutive cycles to feed into the model.
    num_classes:
        The number of bins to divide the normalised ``cycle_capacity``
        into.  The value at ``sequence_length`` ahead of the current
        window is used as the label.
    """

    def __init__(self, df: pd.DataFrame, sequence_length: int, num_classes: int):
        self.sequence_length = sequence_length
        self.data: list[np.ndarray] = []
        self.labels: list[int] = []
        self.num_classes = num_classes

        grouped = df.groupby("battery_name")
        for _, group in grouped:
            group = group.sort_values("cycle_number")
            features = group[[
                "cycle_number", "energy_charge", "capacity_charge",
                "energy_discharge", "capacity_discharge", "cycle_capacity"
            ]].values

            for i in range(len(features) - sequence_length):
                seq = features[i : i + sequence_length]
                # Predict the capacity after the current window
                future_cap = features[i + sequence_length][5]
                self.data.append(seq)
                # Discretise capacity into classes
                bins = np.linspace(0.0, 1.0, num_classes + 1)
                label = np.digitize(future_cap, bins) - 1
                label = np.clip(label, 0, num_classes - 1)
                self.labels.append(label)

        self.data = torch.tensor(np.asarray(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.asarray(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_battery_transfer_task(
    path_to_merged_data: str,
    source_cathodes: list[str],
    target_cathode: str,
    sequence_length: int = 30,
    batch_size: int = 64,
    num_classes: int = 3,
    target_train_ratio: float = 0.75,
    shuffle: bool = True,
):
    """Prepare loaders for source training and target fine-tuning/eval.

    Parameters
    ----------
    path_to_merged_data:
        Location of a parquet file containing the merged battery data.
    source_cathodes:
        List of cathode chemistries to use for training the baseline
        model.  All remaining data from ``target_cathode`` is held out
        for evaluation and optional fine-tuning.
    target_cathode:
        Cathode chemistry used as the unseen target domain.
    sequence_length, batch_size, num_classes:
        See :class:`BatterySequenceDataset`.
    target_train_ratio:
        Proportion of the target cathode data to use for transfer
        learning.  The remainder is used purely for evaluation.
    shuffle:
        Whether to shuffle the data in the returned loaders.
    """
    df = pd.read_parquet(path_to_merged_data, engine="fastparquet")
    

    # Validate cathode names
    available = set(df["cathode"].unique())
    missing = set(source_cathodes + [target_cathode]) - available
    if missing:
        raise ValueError(f"Unknown cathode types requested: {sorted(missing)}")

    # Filter for source and target sets
    source_df = df[df["cathode"].isin(source_cathodes)].copy()
    target_df = df[df["cathode"] == target_cathode].copy()

    # Drop any rows with missing values
    source_df.dropna(inplace=True)
    target_df.dropna(inplace=True)

    # Normalize features independently for source and target
    features = [
        "cycle_number", "energy_charge", "capacity_charge",
        "energy_discharge", "capacity_discharge", "cycle_capacity"
    ]
    # Fit scaler on source data and apply to both domains
    # ``MinMaxScaler`` can produce values outside the [0, 1] range when
    # transforming data that falls outside the feature bounds it was fit
    # on.  This leads to all target labels collapsing into a single class
    # after discretisation, hurting transfer learning.  By enabling
    # ``clip=True`` we ensure transformed features remain within the
    # expected range so that label bins retain their intended meaning.
    scaler = MinMaxScaler(clip=True).fit(source_df[features])
    source_df[features] = scaler.transform(source_df[features])
    target_df[features] = scaler.transform(target_df[features])

    # Split target into train and test for transfer learning
    target_train_df, target_test_df = train_test_split(
        target_df, test_size=1 - target_train_ratio, shuffle=True, random_state=42
    )

    # Build datasets
    source_dataset = BatterySequenceDataset(source_df, sequence_length, num_classes)
    target_train_dataset = BatterySequenceDataset(
        target_train_df, sequence_length, num_classes
    )
    target_test_dataset = BatterySequenceDataset(
        target_test_df, sequence_length, num_classes
    )

    source_loader = DataLoader(
        source_dataset, batch_size=batch_size, shuffle=shuffle
    )
    target_train_loader = DataLoader(
        target_train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    target_test_loader = DataLoader(
        target_test_dataset, batch_size=batch_size, shuffle=False
    )

    return source_loader, target_train_loader, target_test_loader
