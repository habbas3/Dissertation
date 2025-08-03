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
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument('--input_channels', type=int, default=7)
    parser.add_argument('--classification_label', type=str, default='eol_class')
    parser.add_argument('--sequence_length', type=int, default=32)
    # parser.add_argument('--source_cathode', nargs='+', default=["NMC532", "NMC811", "HE5050", "NMC111"])
    # parser.add_argument('--target_cathode', nargs='+', default=["NMC622", "5Vspinel"])
    parser.add_argument('--source_cathode', nargs='+', default=[])
    parser.add_argument('--target_cathode', nargs='+', default=[])
    parser.add_argument('--domain_temperature', type=float, default=1.0,
                            help='Temperature scaling for domain predictions')
    parser.add_argument('--class_temperature', type=float, default=10.0,
                            help='Temperature scaling for class predictions')
    return parser.parse_args()

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


def evaluate_model(model, args):
    """Run inference on the target validation set and return labels and predictions."""
    _, _, _, target_val_loader, _, _ = load_battery_dataset(
        csv_path=args.csv,
        source_cathodes=args.source_cathode,
        target_cathodes=args.target_cathode,
        classification_label=args.classification_label,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )

    if target_val_loader is None:
        return np.array([]), np.array([])

    device = next(model.parameters()).device
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in target_val_loader:
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
    global_habbas3.init()
    df_all = pd.read_csv(args.csv)
    
    df_all["cathode"] = df_all["cathode"].astype(str).str.strip()
    all_cathodes = sorted(df_all["cathode"].unique().tolist())
    # Define cathode groups
    cathode_groups = {
        "NMC_Layered_Oxides": ["NMC532", "NMC622", "NMC111", "NMC811"],
        "Li_rich_Layered_Oxides": ["Li1.2Ni0.3Mn0.6O2", "Li1.35Ni0.33Mn0.67O2.35"],
    }

    # Define cathodes
    # pretrain_cathodes = ["HE5050", "NMC111", "NMC532", "FCG", "NMC811"]
    # transfer_cathodes = ["NMC622", "Li1.2Ni0.3Mn0.6O2", "Li1.35Ni0.33Mn0.67O2.35"]
    grouped = [c for group in cathode_groups.values() for c in group]
    for cat in all_cathodes:
        if cat not in grouped:
            cathode_groups[cat] = [cat]

    model_architectures = [
        "cnn_features_1d",
        "cnn_features_1d_sa",
        "cnn_openmax",
        "WideResNet",
        "WideResNet_sa",
        "WideResNet_edited",
    ]

    
    results = []

   
    print("\n📊 Baseline training (cnn_features_1d)")
    baseline_results = {}
    baseline_conf_matrices = {}
    baseline_eval_counts = {}
    transfer_conf_matrices = {}
    transfer_eval_counts = {}
    args.model_name = "cnn_features_1d"
    args.pretrained = False
    args.pretrained_model_path = None
   
            
    for group_name, cathodes in cathode_groups.items():
        global_habbas3.init()
        args.source_cathode = cathodes
        args.target_cathode = []
        
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device.strip()
        base_dir = os.path.join(
            args.checkpoint_dir,
            f"baseline_cnn1d_{group_name}_{datetime.now().strftime('%m%d')}"
        )
        os.makedirs(base_dir, exist_ok=True)
        model_bl, base_acc, base_time = run_experiment(args, base_dir)
        baseline_results[group_name] = (base_acc, base_time)

        # Evaluate baseline on target group
        args.target_cathode = cathodes
        bl_labels, bl_preds = evaluate_model(model_bl, args)
        baseline_eval_counts[group_name] = len(bl_labels)
        if len(bl_labels) > 0:
            baseline_conf_matrices[group_name] = confusion_matrix(bl_labels, bl_preds)
        else:
            baseline_conf_matrices[group_name] = None

        print(f"✅ Baseline cnn_features_1d -> {group_name}: {base_acc:.4f} ({base_time:.1f}s)")

    # --------------- Transfer Learning -----------------
    print("\n🔧 Transfer learning across groups")
    for target_name, target_cathodes in cathode_groups.items():
        for source_name, source_cathodes in cathode_groups.items():
            if source_name == target_name:
                continue
            for model_name in model_architectures:
                global_habbas3.init()
                args.model_name = model_name
                os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device.strip()

                # Pretrain on source group
                args.pretrained = False
                args.source_cathode = source_cathodes
                args.target_cathode = []
                pre_dir = os.path.join(
                    args.checkpoint_dir,
                    f"pretrain_{model_name}_{source_name}_{datetime.now().strftime('%m%d')}",
                )
                os.makedirs(pre_dir, exist_ok=True)
                run_experiment(args, pre_dir)

                # Fine-tune on target group
                args.pretrained = True
                args.pretrained_model_path = os.path.join(pre_dir, "best_model.pth")
                args.source_cathode = target_cathodes
                args.target_cathode = []
                ft_dir = os.path.join(
                    args.checkpoint_dir,
                    f"transfer_{model_name}_{source_name}_to_{target_name}_{datetime.now().strftime('%m%d')}",
                )
                os.makedirs(ft_dir, exist_ok=True)
                model_ft, transfer_acc, transfer_time = run_experiment(args, ft_dir)

                # Evaluate fine-tuned model on target group
                args.target_cathode = target_cathodes
                tr_labels, tr_preds = evaluate_model(model_ft, args)
                transfer_key = (target_name, source_name, model_name)
                transfer_eval_counts[transfer_key] = len(tr_labels)
                if len(tr_labels) > 0:
                    transfer_conf_matrices[transfer_key] = confusion_matrix(tr_labels, tr_preds)
                else:
                    transfer_conf_matrices[transfer_key] = None

                base_acc, base_time = baseline_results[target_name]
                results.append({
                    "source": source_name,
                    "target": target_name,
                    "model": model_name,
                    "baseline": base_acc,
                    "baseline_time": base_time,
                    "transfer": transfer_acc,
                    "transfer_time": transfer_time,
                })
                print(
                    f"✅ {model_name} {source_name} → {target_name}: baseline {base_acc:.4f} ({base_time:.1f}s) | transfer {transfer_acc:.4f} ({transfer_time:.1f}s)"
                )

    # Print final summary
    print("\n===== Summary =====")
    for r in results:
        target = r['target']
        source = r['source']
        model_name = r['model']
        base_count = baseline_eval_counts.get(target, 0)
        transfer_key = (target, source, model_name)
        transfer_count = transfer_eval_counts.get(transfer_key, 0)
        print(
            f"{model_name} {source}→{target}: baseline {r['baseline']:.4f} ({base_count} samples) → transfer {r['transfer']:.4f} ({transfer_count} samples)"
        )
        base_cm = baseline_conf_matrices.get(target)
        transfer_cm = transfer_conf_matrices.get(transfer_key)
        if (
            base_cm is not None
            and transfer_cm is not None
            and base_cm.size > 0
            and transfer_cm.size > 0
        ):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ConfusionMatrixDisplay(base_cm).plot(ax=axes[0])
            axes[0].set_title(f"Baseline {target}")
            ConfusionMatrixDisplay(transfer_cm).plot(ax=axes[1])
            axes[1].set_title(f"Transfer {source}→{target}")
            plt.show()


if __name__ == '__main__':
    main()
