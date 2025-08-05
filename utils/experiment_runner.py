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

def run_experiment(args, save_dir, trial=None):
    setlogger(os.path.join(save_dir, 'train.log'))
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # ✅ Load dataset using cathode filters
    source_train_loader, source_val_loader, target_train_loader, target_val_loader, label_names, df = load_battery_dataset(
        csv_path=args.csv,
        source_cathodes=args.source_cathode,
        target_cathodes=args.target_cathode,
        classification_label=args.classification_label,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    args.num_classes = len(label_names)

   # Determine size of target training set for LR tuning
    tgt_sample_count = 0
    if target_train_loader is not None:
        if hasattr(target_train_loader, 'dataset'):
            tgt_sample_count = len(target_train_loader.dataset)
        else:
            tgt_sample_count = len(target_train_loader)

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


