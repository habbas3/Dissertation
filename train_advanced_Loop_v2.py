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
    parser.add_argument('--data_name', type=str, default='Battery_inconsistent',
                        choices=['Battery_inconsistent', 'CWRU_inconsistent'])
    parser.add_argument('--data_dir', type=str, default='./my_datasets/Battery',
                        help='Root directory for datasets')
    parser.add_argument('--csv', type=str, default='./my_datasets/Battery/battery_data_labeled.csv',
                        help='CSV file for Battery dataset')
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
    parser.add_argument('--middle_epoch', type=int, default=15) #30
    parser.add_argument('--max_epoch', type=int, default=50) #100
    parser.add_argument('--print_step', type=int, default=25) #50
    parser.add_argument('--inconsistent', type=str, default='UAN')
    parser.add_argument('--model_name', type=str, default='cnn_features_1d')
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument('--input_channels', type=int, default=7)
    parser.add_argument('--classification_label', type=str, default='eol_class')
    parser.add_argument('--sequence_length', type=int, default=32)
    parser.add_argument('--transfer_task', type=str, default='[[0],[1]]',
                        help='CWRU transfer task as [[source],[target]]')
    parser.add_argument('--source_cathode', nargs='+', default=[])
    parser.add_argument('--target_cathode', nargs='+', default=[])
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Override the number of label classes if the raw label is numeric')
    parser.add_argument('--domain_temperature', type=float, default=1.0,
                            help='Temperature scaling for domain predictions')
    parser.add_argument('--class_temperature', type=float, default=10.0,
                            help='Temperature scaling for class predictions')
    parser.add_argument('--lambda_src', type=float, default=1.0,
                    help='Weight for supervised source loss mixed into target steps during transfer')
    parser.add_argument('--improvement_metric', choices=['common', 'hscore', 'overall'], default='common',
                        help='Metric used to compare transfer vs baseline on target_val')
    args = parser.parse_args()
    if args.data_name == 'CWRU_inconsistent' and args.data_dir == './my_datasets/Battery':
        args.data_dir = './my_datasets/CWRU_dataset'
    if isinstance(args.transfer_task, str):
        try:
            args.transfer_task = eval(args.transfer_task)
        except Exception:
            pass
    return args

def build_cathode_groups(csv_path):
    """
    Group cathodes by chemistry similarity.

    Returns keys compatible with existing code:
      - 'nmc_pool':   conventional & derivative NMC layered oxides (incl. HE5050, FCG if present)
      - 'hv_pool':    all non-NMC cathodes (for backward compatibility)
      - 'li_rich_pool': Li-rich layered (oxygen-redox) NMC variants
      - 'spinel_pool': high-voltage spinel (e.g., 5Vspinel/LNMO)
      - 'other_pool': any remaining non-NMC types not caught above

    Upstream that expects only {'nmc_pool','hv_pool'} still works, while you can
    optionally use the finer pools for better transfer task definitions.
    """
    import pandas as pd
    import re

    df = pd.read_csv(csv_path)
    cathodes = df["cathode"].astype(str).str.strip()

    # Unique label universe (whitespace-cleaned)
    uniq = sorted(cathodes.unique().tolist())

    # --- Detect families by chemistry ---
    # 1) Conventional / derivative NMC (layered oxides): NMC***, HE5050, FCG (Full Concentration Gradient NMC)
    nmc_like_mask = (
        cathodes.str.match(r'^(?i:nmc)') |
        cathodes.str.fullmatch(r'(?i:HE5050)') |
        cathodes.str.fullmatch(r'(?i:FCG)')
    )
    nmc_pool = sorted(cathodes[nmc_like_mask].unique().tolist())

    # 2) Li-rich layered oxides (oxygen-redox), e.g., Li1.2Ni0.3Mn0.6O2, Li1.35Ni0.33Mn0.67O2.35
    #    Heuristic: starts with Li1.xNi...O2*
    li_rich_mask = cathodes.str.match(r'(?i)^Li1\.\d+Ni', na=False)
    li_rich_pool = sorted(cathodes[li_rich_mask].unique().tolist())

    # 3) High-voltage spinel (LNMO), e.g., "5Vspinel"
    spinel_mask = cathodes.str.contains(r'(?i)spinel', na=False)
    spinel_pool = sorted(cathodes[spinel_mask].unique().tolist())

    # Backward-compatible 'hv_pool' = non-NMC (union of Li-rich, Spinel, and any other non-NMC)
    non_nmc = [x for x in uniq if x not in set(nmc_pool)]
    hv_pool = sorted(non_nmc)

    # 'other_pool' = non-NMC that are neither Li-rich nor Spinel (kept separate if significantly different)
    li_rich_set = set(li_rich_pool)
    spinel_set = set(spinel_pool)
    other_pool = sorted([x for x in non_nmc if x not in li_rich_set | spinel_set])

    groups = {
        "nmc_pool": nmc_pool,
        "hv_pool": hv_pool,                 # keeps old callers working (all non-NMC)
        "li_rich_pool": li_rich_pool,       # finer grouping you can use now
        "spinel_pool": spinel_pool,         # finer grouping you can use now
        "other_pool": other_pool,           # anything non-NMC not in the two above
    }
    return groups


def run_experiment(args, save_dir, trial=None, baseline=False):
    reset_seed()
    setlogger(os.path.join(save_dir, 'train.log'))
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")
        
    start_time = time.time()

    if args.data_name == 'Battery_inconsistent':
        source_train_loader, source_val_loader, target_train_loader, target_val_loader, label_names, df = load_battery_dataset(
            csv_path=args.csv,
            source_cathodes=args.source_cathode,
            target_cathodes=args.target_cathode,
            classification_label=args.classification_label,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_classes=args.num_classes,
        )
        # keep dataset references for later use
        source_train_dataset = source_train_loader.dataset
        source_val_dataset = source_val_loader.dataset
        target_train_dataset = target_train_loader.dataset if target_train_loader is not None else None
        target_val_dataset = target_val_loader.dataset if target_val_loader is not None else None
        args.num_classes = len(label_names)
    else:
        cwru_dataset = CWRU_inconsistent(args.data_dir, args.transfer_task, args.inconsistent, args.normlizetype)
        if baseline:
            source_train_dataset, source_val_dataset, target_val_dataset = cwru_dataset.data_split(transfer_learning=False)
            target_train_dataset = None
            args.num_classes = len(np.unique(source_train_dataset.labels))
        else:
            source_train_dataset, source_val_dataset, target_train_dataset, target_val_dataset, num_classes = cwru_dataset.data_split(transfer_learning=True)
            args.num_classes = num_classes

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
    return model, acc, elapsed, source_val_loader, target_val_loader


def evaluate_model(model, val_loader):
    """Run inference on the provided validation loader and return predictions."""

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

def compute_common_outlier_metrics(labels, preds, num_known):
    import numpy as np
    labels = np.asarray(labels)
    preds  = np.asarray(preds)
    if labels.size == 0:
        return 0.0, 0.0, 0.0

    common_mask  = labels < num_known
    outlier_mask = labels >= num_known

    if common_mask.any():
        common_acc = float((preds[common_mask] == labels[common_mask]).mean())
    else:
        common_acc = 0.0

    # "Correct" for outliers means predicting any index >= num_known (unknown bucket)
    if outlier_mask.any():
        outlier_acc = float((preds[outlier_mask] >= num_known).mean())
    else:
        outlier_acc = 0.0

    h = (2 * common_acc * outlier_acc) / (common_acc + outlier_acc) if (common_acc + outlier_acc) > 0 else 0.0
    return common_acc, outlier_acc, h


def compute_open_set_metrics(labels, preds, num_known):
    """Compute common-class accuracy, outlier accuracy and the harmonic score.

    ``num_known`` specifies how many class indices are considered *known*.
    Any label greater than or equal to ``num_known`` is treated as an outlier.
    The function gracefully handles cases where one of the groups is empty,
    which is useful for datasets like the Battery dataset that do not contain
    explicit outlier classes.
    """

    labels_np = np.array(labels)
    preds_np = np.array(preds)
    known_mask = labels_np < num_known
    out_mask = labels_np >= num_known

    common_acc = accuracy_score(labels_np[known_mask], preds_np[known_mask]) if known_mask.any() else 0.0
    outlier_acc = accuracy_score(labels_np[out_mask], preds_np[out_mask]) if out_mask.any() else 0.0

    if not out_mask.any():
        hscore = common_acc
    elif not known_mask.any():
        hscore = outlier_acc
    else:
        denom = common_acc + outlier_acc
        hscore = 2 * common_acc * outlier_acc / denom if denom > 0 else 0.0

    return common_acc, outlier_acc, hscore

def run_battery_experiments(args):
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
            # Return order is: model, acc, elapsed, source_val_loader, target_val_loader
            model_bl, baseline_acc, _, _, bl_loader = run_experiment(baseline_args, baseline_dir, baseline=True)
            bl_labels, bl_preds = evaluate_model(model_bl, bl_loader)
            # Keep raw arrays for later "common vs outlier" scoring
            baseline_labels_np = np.array(bl_labels)
            baseline_preds_np = np.array(bl_preds)
            if baseline_labels_np.size:
                baseline_acc = accuracy_score(baseline_labels_np, baseline_preds_np)
            print(f"✅ Baseline {target_cathodes}: {baseline_acc:.4f} ({len(bl_labels)} samples)")

            # ---------------- Transfer learning ----------------
            transfer_args = argparse.Namespace(**vars(args))
            transfer_args.pretrained = True
            transfer_args.pretrained_model_path = os.path.join(pre_dir, "best_model.pth")
            # During fine-tuning we keep the original source cathodes and
            # provide the new target cathodes so that `load_battery_dataset`
            # returns target loaders and enables transfer_mode.
            transfer_args.source_cathode = source_cathodes
            transfer_args.target_cathode = target_cathodes
            ft_dir = os.path.join(
                args.checkpoint_dir,
                f"transfer_{model_name}_{'-'.join(source_cathodes)}_to_{'-'.join(target_cathodes)}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(ft_dir, exist_ok=True)
            model_ft, transfer_acc, _, _, tr_loader = run_experiment(transfer_args, ft_dir)
            tr_labels, tr_preds = evaluate_model(model_ft, tr_loader)
            
            # Use the same num_known (transfer_args.num_classes) to score both runs fairly.
            num_known = transfer_args.num_classes
            
            t_common, t_out, t_h = compute_common_outlier_metrics(tr_labels, tr_preds, num_known)
            b_common, b_out, b_h = compute_common_outlier_metrics(baseline_labels_np, baseline_preds_np, num_known)
            
            # Compare *common-class* accuracy (apples-to-apples). Switch to hscore if you prefer.
            baseline_acc = b_common
            transfer_acc = t_common
            
            print(f"✅ Transfer {source_cathodes} → {target_cathodes}: {transfer_acc:.4f} ({len(tr_labels)} samples)")
            print(f"   ↳ common_acc={t_common:.4f}, outlier_acc={t_out:.4f}, hscore={t_h:.4f}")
            print(f"🧪 Baseline scored on same split: common_acc={b_common:.4f}, outlier_acc={b_out:.4f}, hscore={b_h:.4f}")
            
            improvement = transfer_acc - baseline_acc
            print(f"📊 {source_cathodes} → {target_cathodes}: baseline(common)={baseline_acc:.4f}, transfer(common)={transfer_acc:.4f}, improvement={improvement:+.4f}")
            if transfer_acc < baseline_acc:
                print(f"⚠️ Transfer did not improve over baseline for {source_cathodes} → {target_cathodes}")


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
        
        
def run_cwru_experiments(args):
    model_architectures = [
        "cnn_features_1d",
        "cnn_features_1d_sa",
        "cnn_openmax",
        "WideResNet",
        "WideResNet_sa",
        "WideResNet_edited",
    ]
    transfer_tasks = [args.transfer_task]
    results = []
    for model_name in model_architectures:
        for transfer_task in transfer_tasks:
            global_habbas3.init()
            args.model_name = model_name
            args.transfer_task = transfer_task
            src_str = '-'.join(map(str, transfer_task[0]))
            tgt_str = '-'.join(map(str, transfer_task[1]))

            pre_args = argparse.Namespace(**vars(args))
            pre_args.transfer_task = [transfer_task[0], transfer_task[0]]
            pre_args.pretrained = False
            pre_dir = os.path.join(
                args.checkpoint_dir,
                f"pretrain_{model_name}_{src_str}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(pre_dir, exist_ok=True)
            run_experiment(pre_args, pre_dir, baseline=True)

            baseline_args = argparse.Namespace(**vars(args))
            baseline_args.transfer_task = [transfer_task[1], transfer_task[1]]
            baseline_args.pretrained = False
            baseline_dir = os.path.join(
                args.checkpoint_dir,
                f"baseline_{model_name}_{tgt_str}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(baseline_dir, exist_ok=True)
            model_bl, baseline_acc, _, _, bl_loader = run_experiment(baseline_args, baseline_dir, baseline=True)
            bl_labels, bl_preds = evaluate_model(model_bl, bl_loader)
            
            # Stash raw arrays; we’ll score later using the transfer-defined num_known
            baseline_labels_np = np.array(bl_labels)
            baseline_preds_np = np.array(bl_preds)
            
            if baseline_labels_np.size:
                baseline_acc = accuracy_score(baseline_labels_np, baseline_preds_np)
            print(f"✅ Baseline {tgt_str}: {baseline_acc:.4f} ({len(bl_labels)} samples)")
                
            # Use the same "known" count that the *transfer* task will use
            transfer_args = argparse.Namespace(**vars(args))
            transfer_args.pretrained = True
            transfer_args.transfer_task = transfer_task
            transfer_args.pretrained_model_path = os.path.join(pre_dir, "best_model.pth")

            ft_dir = os.path.join(
                args.checkpoint_dir,
                f"transfer_{model_name}_{src_str}_to_{tgt_str}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(ft_dir, exist_ok=True)
            model_ft, transfer_acc, _, _, tr_loader = run_experiment(transfer_args, ft_dir)
            tr_labels, tr_preds = evaluate_model(model_ft, tr_loader)
            
            # Score both baseline and transfer on the SAME split:
            # num_known comes from the transfer configuration (CWRU has no cathodes; this is just the known-class count)
            num_known = transfer_args.num_classes
            
            t_common, t_out, t_h = compute_common_outlier_metrics(tr_labels, tr_preds, num_known)
            b_common, b_out, b_h = compute_common_outlier_metrics(baseline_labels_np, baseline_preds_np, num_known)
            
            # Compare apples-to-apples on common-class accuracy (or switch to hscore if you prefer)
            baseline_acc = b_common
            transfer_acc = t_common
            
            print(
                f"✅ Transfer {src_str} → {tgt_str}: {transfer_acc:.4f} ({len(tr_labels)} samples)\n"
                f"   ↳ common_acc={t_common:.4f}, outlier_acc={t_out:.4f}, hscore={t_h:.4f}"
            )
            print(
                f"🧪 Baseline scored on same split: common_acc={b_common:.4f}, outlier_acc={b_out:.4f}, hscore={b_h:.4f}"
            )
            
            improvement = transfer_acc - baseline_acc
            print(
                f"📊 {src_str} → {tgt_str}: baseline(common)={baseline_acc:.4f}, "
                f"transfer(common)={transfer_acc:.4f}, improvement={improvement:+.4f}"
            )
            if transfer_acc < baseline_acc:
                print(f"⚠️ Transfer did not improve over baseline for {src_str} → {tgt_str}")


            results.append(
                {
                    "model": model_name,
                    "source": src_str,
                    "target": tgt_str,
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


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device.strip()

    if args.data_name == 'Battery_inconsistent':
        run_battery_experiments(args)
    else:
        run_cwru_experiments(args)
            


if __name__ == '__main__':
    main()
