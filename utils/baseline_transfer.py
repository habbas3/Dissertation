#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 23:24:25 2025

@author: habbas
"""

"""Baseline and transfer learning experiments across cathode groups.

This script trains a simple LSTM-based classifier on data from one or
more cathode chemistries and evaluates it on an unseen cathode.  It then
applies a transfer-learning stage where the model is fine-tuned on a
portion of the target cathode data.  The intention is to demonstrate the
benefits of transfer learning for end-of-life (EOL) class prediction.

Example
-------
```bash
python baseline_transfer.py \
    --data ./path/to/merged.parquet \
    --source-cathodes NMC111 NMC532 \
    --target-cathode LFP
```
"""

from __future__ import annotations

import argparse
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim

# ``my_datasets`` has a heavyweight ``__init__`` that pulls in optional
# dependencies.  To keep this script lightweight we load the needed
# module directly from its file path instead of importing the package.
import importlib.util
from pathlib import Path

_loader_path = Path(__file__).parent / "my_datasets" / "battery_dataset_loader.py"
spec = importlib.util.spec_from_file_location("battery_dataset_loader", _loader_path)
_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_module)  # type: ignore[attr-defined]
load_battery_transfer_task = _module.load_battery_transfer_task


class LSTMClassifier(nn.Module):
    """A minimal LSTM classifier for sequence data."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def train_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimiser: optim.Optimizer,
    device: torch.device,
) -> None:
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimiser.step()


def evaluate(model: nn.Module, loader: Iterable, device: torch.device) -> float:
    """Return accuracy of ``model`` over ``loader``."""

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.numel()
    return correct / total if total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline vs transfer learning")
    parser.add_argument("--data", required=True, help="Path to merged parquet data")
    parser.add_argument(
        "--source-cathodes",
        nargs="+",
        required=True,
        help="Cathode groups used for source training",
    )
    parser.add_argument("--target-cathode", required=True, help="Target cathode group")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--source-epochs", type=int, default=5)
    parser.add_argument("--transfer-epochs", type=int, default=3)
    return parser.parse_args()


def main() -> None:  # pragma: no cover - entry point
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = load_battery_transfer_task(
        path_to_merged_data=args.data,
        source_cathodes=args.source_cathodes,
        target_cathode=args.target_cathode,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
    )

    source_loader, target_train_loader, target_test_loader = loaders

    model = LSTMClassifier(
        input_size=6,  # number of features in BatterySequenceDataset
        hidden_size=args.hidden_size,
        num_classes=args.num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters())

    # ---------------------------- Baseline ----------------------------
    for _ in range(args.source_epochs):
        train_epoch(model, source_loader, criterion, optimiser, device)

    baseline_acc = evaluate(model, target_test_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # ------------------------- Transfer stage -------------------------
    for _ in range(args.transfer_epochs):
        train_epoch(model, target_train_loader, criterion, optimiser, device)

    transfer_acc = evaluate(model, target_test_loader, device)
    print(f"Transfer accuracy: {transfer_acc:.4f}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()