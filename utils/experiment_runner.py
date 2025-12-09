#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 15:47:19 2025

@author: habbas
"""

# utils/experiment_runner.py
import os
import logging
from utils.logger import setlogger
from utils.train_utils_open_univ import train_utils_open_univ
from my_datasets.Battery_label_inconsistent import load_battery_dataset
from my_datasets.CWRU_label_inconsistent import CWRU_inconsistent
from torch.utils.data import Dataset, DataLoader
import torch

# experiment_runner.py

import os
import logging
from utils.logger import setlogger
from utils.train_utils_open_univ import train_utils_open_univ
from my_datasets.Battery_label_inconsistent import load_battery_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import json
import copy


def _read_timing_json(run_dir):
    import json as _json, os as _os
    path = _os.path.join(run_dir, "train_timing.json")
    if _os.path.exists(path):
        try:
            with open(path, "r") as f:
                return _json.load(f)
        except Exception:
            return {}
    return {}


def _cm_with_min_labels(y_true, y_pred, min_labels=3):
    """Return confusion matrix and explicit label list with a minimum size."""
    import numpy as np

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    max_label = int(max(np.max(y_true, initial=0), np.max(y_pred, initial=0)))
    labels = list(range(max(min_labels, max_label + 1)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels


def _run_single_cycle_setting(args, save_dir, trial=None):
    args.early_stop_patience = getattr(args, 'early_stop_patience', 5)
    setlogger(os.path.join(save_dir, 'train.log'))
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # ✅ Load dataset using cathode filters (already returns DataLoaders)
    (
        source_train_loader,
        source_val_loader,
        target_train_loader,
        target_val_loader,
        label_names,
        df,
        cycle_stats,
    ) = load_battery_dataset(
        csv_path=args.csv,
        source_cathodes=args.source_cathode,
        target_cathodes=args.target_cathode,
        classification_label=args.classification_label,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        cycles_per_file=getattr(args, 'cycles_per_file', 50),
        source_cycles_per_file=getattr(args, 'source_cycles_per_file', None),
        target_cycles_per_file=getattr(args, 'target_cycles_per_file', None),
        sample_random_state=getattr(args, 'sample_random_state', 42),
    )
    args.num_classes = len(label_names)
    args.dataset_cycle_stats = cycle_stats

   # Determine size of target training set for LR tuning
    tgt_sample_count = 0
    if target_train_loader is not None:
        if hasattr(target_train_loader, 'dataset'):
            tgt_sample_count = len(target_train_loader.dataset)
        else:
            tgt_sample_count = len(target_train_loader)
            
    # If a pretrained model path is provided, we are fine-tuning and can
    # ignore the source loaders to train purely on target data
    if getattr(args, 'pretrained_model_path', None):
        logging.info("Fine-tuning mode: ignoring source loaders and training only on target data")
        source_train_loader = None
        source_val_loader = None

    # ✅ Inject Optuna trial hyperparameters
    if trial is not None:
        lr_low, lr_high = (1e-5, 1e-3) if 0 < tgt_sample_count < 100 else (1e-4, 5e-3)
        args.lr = trial.suggest_float("lr", lr_low, lr_high, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        args.hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 1024])
        args.bottleneck_num = trial.suggest_categorical("bottleneck_num", [128, 256])
        args.droprate = trial.suggest_float("droprate", 0.1, 0.5)
    elif 0 < tgt_sample_count < 100:
        args.lr = min(args.lr, 1e-4)
        logging.info(f"Reduced learning rate to {args.lr} for small target set ({tgt_sample_count} samples)")

    # ✅ Call trainer fully universally now:
    trainer = train_utils_open_univ(
        args, save_dir,
        source_train_loader, source_val_loader,
        target_train_loader, target_val_loader,
        source_train_loader.dataset if hasattr(source_train_loader, 'dataset') else None,
        target_val_loader.dataset if (target_val_loader is not None and hasattr(target_val_loader, 'dataset')) else None,
    )

    trainer.setup()
    return trainer.train()

def _run_cycle_ablation(args, save_dir, trial=None):
    """Iteratively train on increasing early-cycle horizons until accuracy stops improving."""

    start = getattr(args, "cycle_ablation_start", 5)
    step = getattr(args, "cycle_ablation_step", 10)
    max_cycles = getattr(args, "cycle_ablation_max", None)

    best_model = None
    best_acc = float("-inf")
    best_cycle_count = None

    current_cycles = start
    while True:
        scoped_args = copy.deepcopy(args)
        scoped_args.cycles_per_file = current_cycles
        # Mirror the limit onto source/target when callers did not explicitly override them.
        if getattr(scoped_args, "source_cycles_per_file", None) is None:
            scoped_args.source_cycles_per_file = current_cycles
        if getattr(scoped_args, "target_cycles_per_file", None) is None:
            scoped_args.target_cycles_per_file = current_cycles

        scoped_dir = os.path.join(save_dir, f"cycles_{current_cycles}")
        os.makedirs(scoped_dir, exist_ok=True)
        logging.info(
            "🚀 Cycle ablation run with first %s cycles (step=%s, max=%s)",
            current_cycles,
            step,
            max_cycles,
        )
        model, acc = _run_single_cycle_setting(scoped_args, scoped_dir, trial)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_cycle_count = current_cycles
            logging.info(
                "✅ Improvement detected at %s cycles (acc=%.4f). Continuing sweep...",
                current_cycles,
                acc,
            )
        else:
            logging.info(
                "⏹️  Stopping cycle ablation: %s cycles gave acc=%.4f (no improvement over %.4f at %s cycles)",
                current_cycles,
                acc,
                best_acc,
                best_cycle_count,
            )
            break

        if max_cycles is not None and current_cycles >= max_cycles:
            logging.info(
                "📏 Reached configured maximum cycle horizon (%s). Ending ablation.",
                max_cycles,
            )
            break

        current_cycles += step

    logging.info(
        "🏁 Best cycle horizon: %s cycles with acc=%.4f",
        best_cycle_count,
        best_acc,
    )
    return best_model, best_acc


def run_experiment(args, save_dir, trial=None):
    if getattr(args, "cycle_ablation", False):
        return _run_cycle_ablation(args, save_dir, trial)
    return _run_single_cycle_setting(args, save_dir, trial)

def add_timing_to_row(row, base_dir, ft_dir):
    """Merge timing information into a summary row if available."""
    base_timing = _read_timing_json(base_dir)
    ft_timing = _read_timing_json(ft_dir)

    row["baseline_time_sec"] = float(base_timing.get("wall_time_sec", float("nan")))
    row["transfer_time_sec"] = float(ft_timing.get("wall_time_sec", float("nan")))

    if not (row["baseline_time_sec"] != row["baseline_time_sec"]) and \
       not (row["transfer_time_sec"] != row["transfer_time_sec"]) and \
       row["transfer_time_sec"] > 0:
        row["speedup_x"] = row["baseline_time_sec"] / row["transfer_time_sec"]
    else:
        row["speedup_x"] = float("nan")
    return row


