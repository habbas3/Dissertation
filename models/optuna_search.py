#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 19:19:26 2025

@author: habbas
"""

import optuna
import os
from datetime import datetime
from utils.experiment_runner import run_experiment
import json


def objective(trial, args, base_dir, model_name):
    # Inject model name so downstream code doesn't break
    args.model_name = model_name

    # Inject hyperparameters
    args.lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    args.hidden_size = trial.suggest_categorical('hidden_size', [256, 512, 1024])
    args.bottleneck_num = trial.suggest_categorical('bottleneck_num', [256])
    args.droprate = trial.suggest_uniform('droprate', 0.1, 0.5)

    trial_id = trial.number
    save_dir = os.path.join(base_dir, f"trial_{trial_id}")
    os.makedirs(save_dir, exist_ok=True)

    # Run training with these hyperparameters
    args.model_name = model_name
    _, target_acc = run_experiment(args, save_dir)
    return float(target_acc)
    # h_score = run_experiment(args, save_dir, trial)
    # return h_score

def run_optuna_search(args, model_name, n_trials=10):
    import optuna
    base_dir = os.path.join(args.checkpoint_dir, f"optuna_{model_name}")
    os.makedirs(base_dir, exist_ok=True)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args, base_dir, model_name), n_trials=n_trials)
    
    best_params = study.best_trial.params
    with open(os.path.join(base_dir, f"{model_name}_best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=2)

    print(f"✅ Optuna Best Trial for {model_name}:")
    print(study.best_trial)
    return study.best_value


