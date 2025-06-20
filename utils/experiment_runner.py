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
    source_train_dataset, source_val_dataset, target_train_dataset, target_val_dataset, label_names, df = load_battery_dataset(
        csv_path=args.csv,
        source_cathodes=args.source_cathode,
        target_cathodes=args.target_cathode,
        classification_label=args.classification_label,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    args.num_classes = len(label_names)

    # ✅ Build dataloaders
    source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    source_val_loader = DataLoader(source_val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ✅ Build target loaders *only if available*
    if target_train_dataset is not None:
        target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        target_train_loader = None

    if target_val_dataset is not None:
        target_val_loader = DataLoader(target_val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    else:
        target_val_loader = None

    # ✅ Inject Optuna trial hyperparameters
    if trial is not None:
        args.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        args.hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 1024])
        args.bottleneck_num = trial.suggest_categorical("bottleneck_num", [128, 256])
        args.droprate = trial.suggest_float("droprate", 0.1, 0.5)

    # ✅ Call trainer fully universally now:
    trainer = train_utils_open_univ(
        args, save_dir,
        source_train_loader, source_val_loader,
        target_train_loader, target_val_loader,
        source_train_dataset, target_val_dataset
    )

    trainer.setup()
    return trainer.train()


