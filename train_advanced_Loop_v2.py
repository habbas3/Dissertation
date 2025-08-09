#!/usr/bin/python
# -*- coding:utf-8 -*-

#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
from utils.logger import setlogger
import logging
import global_habbas3
from utils.train_utils_combines import train_utils
from utils.train_utils_open_univ import train_utils_open_univ
import torch
import numpy as np
import sklearn
import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, ConfusionMatrixDisplay, confusion_matrix
import faulthandler; faulthandler.enable()
import random
import optuna
import json
from my_datasets.CWRU_label_inconsistent import CWRU_inconsistent
from my_datasets.Battery_label_inconsistent import load_battery_dataset
from collections import Counter
from models.optuna_search import run_optuna_search
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
from itertools import combinations


import sys
sys.path.append(os.path.dirname(__file__))

print(torch.__version__)
warnings.filterwarnings('ignore')

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def reset_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

#Reverted back to functioning ver

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data_name', type=str, default='Battery_inconsistent')
    parser.add_argument('--data_dir', type=str, default='./my_datasets/Battery')
    parser.add_argument('--csv', type=str, default='./my_datasets/Battery/battery_data_labeled.csv')
    parser.add_argument('--normlizetype', type=str, default='mean-std')
    parser.add_argument('--method', type=str, default='sngp', choices=['deterministic', 'sngp'])
    parser.add_argument('--gp_hidden_dim', type=int, default=2048)
    parser.add_argument('--spectral_norm_bound', type=float, default=0.95)
    parser.add_argument('--n_power_iterations', type=int, default=1)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--print-freq', '-p', default=10, type=int)
    parser.add_argument('--layers', default=16, type=int)
    parser.add_argument('--widen-factor', default=1, type=int)
    parser.add_argument('--droprate', default=0.3, type=float)
    parser.add_argument('--cuda_device', type=str, default='0')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--bottleneck', type=bool, default=True)
    parser.add_argument('--bottleneck_num', type=int, default=256)
    parser.add_argument('--last_batch', type=bool, default=False)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--trade_off_adversarial', type=str, default='Step')
    parser.add_argument('--lam_adversarial', type=float, default=1)
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--lr_scheduler', type=str, default='step')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--steps', type=str, default='150, 250')
    parser.add_argument('--middle_epoch', type=int, default=30) #30
    parser.add_argument('--max_epoch', type=int, default=100) #100
    parser.add_argument('--print_step', type=int, default=50) #50
    parser.add_argument('--inconsistent', type=str, default='UAN')
    parser.add_argument('--model_name', type=str, default='cnn_features_1d')
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument('--input_channels', type=int, default=7)
    parser.add_argument('--classification_label', type=str, default='eol_class')
    parser.add_argument('--sequence_length', type=int, default=32)
    # parser.add_argument('--source_cathode', nargs='+', default=["NMC532", "NMC811", "HE5050", "NMC111"])
    # parser.add_argument('--target_cathode', nargs='+', default=["NMC622", "5Vspinel"])
    parser.add_argument('--source_cathode', nargs='+', default=[])
    parser.add_argument('--target_cathode', nargs='+', default=[])
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Override the number of label classes if the raw label is numeric')
    parser.add_argument('--domain_temperature', type=float, default=1.0,
                            help='Temperature scaling for domain predictions')
    parser.add_argument('--class_temperature', type=float, default=10.0,
                            help='Temperature scaling for class predictions')
    return parser.parse_args()

def build_cathode_groups(csv_path):
    """Dynamically group cathodes into NMC-based and high-voltage pools.

    This allows experiments to automatically span more combinations without
    manually enumerating every cathode name.  Additional grouping logic can be
    inserted here as new cathode families are added to the dataset.
    """
    df = pd.read_csv(csv_path)
    cathodes = df["cathode"].astype(str).str.strip()
    groups = {
        "nmc_pool": sorted(cathodes[cathodes.str.contains("NMC", case=False)].unique().tolist()),
        "hv_pool": sorted(cathodes[~cathodes.str.contains("NMC", case=False)].unique().tolist()),
    }
    return groups

def run_experiment(args, save_dir, trial=None):
    reset_seed()
    setlogger(os.path.join(save_dir, 'train.log'))
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")
        
    start_time = time.time()

    # ✅ Load dataset using cathode filters
    source_train_dataset, source_val_dataset, target_train_dataset, target_val_dataset, label_names, df = load_battery_dataset(
        csv_path=args.csv,
        source_cathodes=args.source_cathode,
        target_cathodes=args.target_cathode,
        classification_label=args.classification_label,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_classes=args.num_classes,
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
    model, acc = trainer.train()
    elapsed = time.time() - start_time
    return model, acc, elapsed


def evaluate_model(model, args, baseline=False):
    """Run inference on the validation set and return labels and predictions.

    When ``baseline`` is True, the evaluation is performed on the validation
    split of the target cathode data without any transfer learning.  Otherwise
    the function evaluates the fine-tuned model on the held-out target split.
    """
    if baseline:
        # Load only the target cathode data (treated as source within the
        # loader) so that we can evaluate a model trained from scratch.
        _, val_loader, _, _, _, _ = load_battery_dataset(
            csv_path=args.csv,
            source_cathodes=args.source_cathode,
            target_cathodes=[],
            classification_label=args.classification_label,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_classes=args.num_classes,
        )
    else:
        _, _, _, val_loader, _, _ = load_battery_dataset(
            csv_path=args.csv,
            source_cathodes=args.source_cathode,
            target_cathodes=args.target_cathode,
            classification_label=args.classification_label,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_classes=args.num_classes,
        )

    if val_loader is None:
        return np.array([]), np.array([])

    device = next(model.parameters()).device
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device.strip()

    # Dynamically build cathode groups from the CSV so new cathode types are
    # picked up automatically without modifying this script.
    cathode_groups = build_cathode_groups(args.csv)

    # Baseline architecture defaults to CNN-1D but other models can be
    # explored by overriding --model_name.
    

    model_architectures = [
        "cnn_features_1d",
        "cnn_features_1d_sa",
        "cnn_openmax",
        "WideResNet",
        "WideResNet_sa",
        "WideResNet_edited",
    ]

    
    # Generate experiments by pairing every non-empty subset of a source group
    # with every non-empty subset of a different target group.  This yields a
    # rich set of baseline vs. transfer comparisons across cathode families.
    experiment_configs = []
    for src_group, src_cathodes in cathode_groups.items():
        for tgt_group, tgt_cathodes in cathode_groups.items():
            if src_group == tgt_group:
                continue
            for r_s in range(1, len(src_cathodes) + 1):
                for src_subset in combinations(src_cathodes, r_s):
                    for r_t in range(1, len(tgt_cathodes) + 1):
                        for tgt_subset in combinations(tgt_cathodes, r_t):
                            experiment_configs.append((list(src_subset), list(tgt_subset)))

    # Allow command-line overrides for a single experiment or model.
    if args.source_cathode and args.target_cathode:
        experiment_configs = [(args.source_cathode, args.target_cathode)]
    if args.model_name:
        model_architectures = [args.model_name]

    results = []
    for model_name in model_architectures:
        for source_cathodes, target_cathodes in experiment_configs:
            global_habbas3.init()
            args.model_name = model_name
            args.source_cathode = source_cathodes
            # ---------------- Pretraining on source cathodes ----------------
            pre_args = argparse.Namespace(**vars(args))
            pre_args.target_cathode = []
            pre_args.pretrained = False
            pre_args.pretrained_model_path = None
            pre_dir = os.path.join(
                args.checkpoint_dir,
                f"pretrain_{model_name}_{'-'.join(source_cathodes)}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(pre_dir, exist_ok=True)
            run_experiment(pre_args, pre_dir)

            # ---------------- Baseline: train target from scratch ----------------
            baseline_args = argparse.Namespace(**vars(args))
            baseline_args.source_cathode = target_cathodes
            baseline_args.target_cathode = []
            baseline_args.pretrained = False
            baseline_dir = os.path.join(
                args.checkpoint_dir,
                f"baseline_{model_name}_{'-'.join(target_cathodes)}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(baseline_dir, exist_ok=True)
            model_bl, baseline_acc, _ = run_experiment(baseline_args, baseline_dir)
            bl_labels, bl_preds = evaluate_model(model_bl, baseline_args, baseline=True)
            if len(bl_labels):
                baseline_acc = accuracy_score(bl_labels, bl_preds)
            print(
                f"✅ Baseline {target_cathodes}: {baseline_acc:.4f} ({len(bl_labels)} samples)"
            )

            # ---------------- Transfer learning ----------------
            transfer_args = argparse.Namespace(**vars(args))
            transfer_args.pretrained = True
            transfer_args.pretrained_model_path = os.path.join(pre_dir, "best_model.pth")
            transfer_args.source_cathode = target_cathodes
            transfer_args.target_cathode = []
            ft_dir = os.path.join(
                args.checkpoint_dir,
                f"transfer_{model_name}_{'-'.join(source_cathodes)}_to_{'-'.join(target_cathodes)}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(ft_dir, exist_ok=True)
            model_ft, transfer_acc, _ = run_experiment(transfer_args, ft_dir)
            tr_labels, tr_preds = evaluate_model(model_ft, transfer_args, baseline=True)
            if len(tr_labels):
                transfer_acc = accuracy_score(tr_labels, tr_preds)
            print(
                f"✅ Transfer {source_cathodes} → {target_cathodes}: {transfer_acc:.4f} ({len(tr_labels)} samples)"
            )
            
            improvement = transfer_acc - baseline_acc
            print(
                f"📊 {source_cathodes} → {target_cathodes}: baseline={baseline_acc:.4f}, "
                f"transfer={transfer_acc:.4f}, improvement={improvement:+.4f}"
            )
            if transfer_acc < baseline_acc:
                print(
                    f"⚠️ Transfer did not improve over baseline for {source_cathodes} → {target_cathodes}"
                )

            results.append(
                {
                    "model": model_name,
                    "source": "-".join(source_cathodes),
                    "target": "-".join(target_cathodes),
                    "baseline_acc": baseline_acc,
                    "transfer_acc": transfer_acc,
                    "improvement": improvement,
                }
            )

    if results:
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(
            args.checkpoint_dir,
            f"summary_{datetime.now().strftime('%m%d')}.csv",
        )
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")
        print(summary_df[["source", "target", "baseline_acc", "transfer_acc", "improvement"]])
            


if __name__ == '__main__':
    main()
