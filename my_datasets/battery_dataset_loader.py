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


class BatterySequenceDataset(Dataset):
    def __init__(self, df, sequence_length):
        self.sequence_length = sequence_length
        self.data = []
        self.labels = []

        grouped = df.groupby("battery_name")
        for _, group in grouped:
            group = group.sort_values("cycle_number")
            features = group[[
                "cycle_number", "energy_charge", "capacity_charge",
                "energy_discharge", "capacity_discharge", "cycle_capacity"
            ]].values

            for i in range(len(features) - sequence_length):
                seq = features[i:i + sequence_length]
                label = features[i + sequence_length][5]  # Predict future cycle_capacity
                self.data.append(seq)
                self.labels.append(label)

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_battery_transfer_task(
    path_to_merged_data,
    source_cathode,
    target_cathode,
    sequence_length=30,
    batch_size=64,
    shuffle=True
):
    df = pd.read_parquet(path_to_merged_data, engine="fastparquet")
    

    # Filter for only source and target
    source_df = df[df["cathode"] == source_cathode].copy()
    target_df = df[df["cathode"] == target_cathode].copy()

    # Drop any rows with missing values
    source_df.dropna(inplace=True)
    target_df.dropna(inplace=True)

    # Normalize features independently for source and target
    scaler = MinMaxScaler()
    features = [
        "cycle_number", "energy_charge", "capacity_charge",
        "energy_discharge", "capacity_discharge", "cycle_capacity"
    ]
    source_df[features] = scaler.fit_transform(source_df[features])
    target_df[features] = scaler.fit_transform(target_df[features])

    # Build datasets
    train_dataset = BatterySequenceDataset(source_df, sequence_length)
    test_dataset = BatterySequenceDataset(target_df, sequence_length)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
