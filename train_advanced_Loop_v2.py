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
from llm_selector import select_config
from utils.experiment_runner import _cm_with_min_labels
import os as _os

_os.environ["OPENAI_API_KEY"] = "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"


import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "my_datasets"))

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
    # --- LLM meta-selection flags (enabled by default; add --no-* to disable) ---
    parser.add_argument('--auto_select', dest='auto_select', action='store_true', default=True,
                        help='Use an LLM to choose model/config from data context (default: on)')
    parser.add_argument('--no-auto_select', dest='auto_select', action='store_false',
                        help='Disable LLM auto-selection')
    
    parser.add_argument('--llm_compare', dest='llm_compare', action='store_true', default=True,
                        help='Run a small comparison set to verify the LLM pick (default: on)')
    parser.add_argument('--no-llm_compare', dest='llm_compare', action='store_false',
                        help='Disable multi-candidate comparison proof')
    
    parser.add_argument('--llm_backend', choices=['auto','openai','ollama'], default='openai',
                        help='Which LLM provider to use')
    parser.add_argument('--llm_model', type=str, default=None,
                        help='Provider model id (e.g., gpt-4.1-mini or llama3.1)')
    parser.add_argument('--llm_context', type=str, default='',
                        help='Short text describing dataset (e.g., Argonne cycles; channels=7; seq_len=256; label-inconsistent)')
    
        
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


def _build_numeric_summary(dataloaders, args):
    import numpy as np
    
    for key in ['target_train','source_train','target_val','source_val']:
        dl = dataloaders.get(key)
        if dl is None:
            continue
        try:
           x, y = next(iter(dl))
        except Exception:
            continue
        C, L, B = int(x.shape[1]), int(x.shape[-1]), int(x.shape[0])
        x_np = x.detach().cpu().numpy()
        ch_mean = x_np.mean(axis=(0,2)).tolist()[: min(C, 8)]
        ch_std  = x_np.std(axis=(0,2)).tolist()[: min(C, 8)]
        return {
            "dataset": args.data_name,
            "split_used": key,
            "batch_size_seen": B,
            "channels": C,
            "seq_len": L,
            "num_classes_hint": getattr(args, "num_classes", None) or getattr(args, "n_class", None),
            "lr_hint": args.lr,
            "dropout_hint": getattr(args, "droprate", 0.3),
            "notes": "label_inconsistent" if getattr(args, "inconsistent", False) else "closed_set",
            "ch_mean_head": ch_mean,
            "ch_std_head": ch_std,
        }
    return {"dataset": args.data_name, "notes": "no_batch_available"}



def run_experiment(args, save_dir, trial=None, baseline=False):
    reset_seed()
    setlogger(os.path.join(save_dir, 'train.log'))
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")
        
    start_time = time.time()
    
    # Persist LLM config into this run's save_dir (next to best_model.pth)
    try:
        import os as _os, json as _json
        if save_dir:
            _os.makedirs(save_dir, exist_ok=True)
            if getattr(args, "llm_cfg", None):
                _per_run_path = _os.path.join(
                    save_dir, f"llm_selected_config_{getattr(args,'llm_cfg_stamp','')}.json"
                )
                with open(_per_run_path, "w") as f:
                    _json.dump({
                        "llm_choice": args.llm_cfg,
                        "inputs": getattr(args, "llm_cfg_inputs", {})
                    }, f, indent=2)
                print(f"📝 LLM config copied into run folder: {_per_run_path}")
    except Exception as _e:
        print(f"⚠️ Could not write LLM config into run folder: {_e}")


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

    
    # Define cathode groups as coarse families
    nmc_group = sorted([c for c in cathode_groups.get("nmc_pool", []) if c.upper() != "FCG"])
    fcg_group = [c for c in cathode_groups.get("nmc_pool", []) if c.upper() == "FCG"]
    spinel_group = cathode_groups.get("spinel_pool", [])
    li_group = cathode_groups.get("li_rich_pool", [])
    fcg_li_group = sorted(set(fcg_group + li_group))

    group_defs = {
        "NMC": nmc_group,
        "FCG": fcg_group,
        "5Vspinel": spinel_group,
        "FCG+Li": fcg_li_group,
    }
    group_defs = {k: v for k, v in group_defs.items() if v}

    # Build experiment list using group combinations
    experiment_configs = []
    for src_name, src_cathodes in group_defs.items():
        for tgt_name, tgt_cathodes in group_defs.items():
            if src_name == tgt_name:
                continue
            experiment_configs.append((src_name, tgt_name, src_cathodes, tgt_cathodes))

    # Allow command-line overrides for a single experiment or model.
    if args.source_cathode and args.target_cathode:
        experiment_configs = [("custom", "custom", args.source_cathode, args.target_cathode)]
    if args.model_name:
        model_architectures = [args.model_name]

    results = []
    for model_name in model_architectures:
        for src_name, tgt_name, source_cathodes, target_cathodes in experiment_configs:
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
                f"pretrain_{model_name}_{src_name}_{datetime.now().strftime('%m%d')}",
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
                f"baseline_{model_name}_{tgt_name}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(baseline_dir, exist_ok=True)
            # Return order is: model, acc, elapsed, source_val_loader, target_val_loader
            model_bl, baseline_acc, _, _, _ = run_experiment(
                baseline_args, baseline_dir, baseline=True
            )

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
                f"transfer_{model_name}_{src_name}_to_{tgt_name}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(ft_dir, exist_ok=True)
            model_ft, transfer_acc, _, _, tr_loader = run_experiment(transfer_args, ft_dir)
            tr_labels, tr_preds = evaluate_model(model_ft, tr_loader)
            
            # Evaluate baseline on the SAME target validation loader
            bl_labels, bl_preds = evaluate_model(model_bl, tr_loader)
            baseline_labels_np = np.array(bl_labels)
            baseline_preds_np = np.array(bl_preds)
            
            # Use the same num_known (transfer_args.num_classes) to score both runs fairly.
            num_known = transfer_args.num_classes
            
            t_common, t_out, t_h = compute_common_outlier_metrics(tr_labels, tr_preds, num_known)
            b_common, b_out, b_h = compute_common_outlier_metrics(
                baseline_labels_np, baseline_preds_np, num_known
            )
            
            # Compare *common-class* accuracy (apples-to-apples). Switch to hscore if you prefer.
            baseline_acc = b_common
            transfer_acc = t_common
            
            print(
                f"✅ Baseline {tgt_name}: {baseline_acc:.4f} ({len(bl_labels)} samples)"
            )
            print(
                f"✅ Transfer {src_name} → {tgt_name}: {transfer_acc:.4f} ({len(tr_labels)} samples)"
            )
            print(
                f"   ↳ common_acc={t_common:.4f}, outlier_acc={t_out:.4f}, hscore={t_h:.4f}"
            )
            print(
                f"🧪 Baseline scored on same split: common_acc={b_common:.4f}, outlier_acc={b_out:.4f}, hscore={b_h:.4f}"
            )
            
            improvement = transfer_acc - baseline_acc
            print(f"📊 {src_name} → {tgt_name}: baseline(common)={baseline_acc:.4f}, transfer(common)={transfer_acc:.4f}, improvement={improvement:+.4f}")
            if transfer_acc < baseline_acc:
                print(f"⚠️ Transfer did not improve over baseline for {src_name} → {tgt_name}")

            # Confusion matrices with consistent axes and minimum label coverage
            cm_transfer, labels_tr = _cm_with_min_labels(tr_labels, tr_preds, min_labels=3)
            cm_baseline, labels_bl = _cm_with_min_labels(baseline_labels_np, baseline_preds_np, min_labels=3)
            labels = sorted(set(labels_tr) | set(labels_bl))
            cm_transfer, labels = _cm_with_min_labels(tr_labels, tr_preds, min_labels=len(labels))
            cm_baseline, _ = _cm_with_min_labels(baseline_labels_np, baseline_preds_np, min_labels=len(labels))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_transfer, display_labels=labels)
            disp.plot(cmap='Blues')
            plt.savefig(os.path.join(ft_dir, f"cm_transfer_{src_name}_to_{tgt_name}.png"))
            plt.close()

            disp = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=labels)
            disp.plot(cmap='Blues')
            plt.savefig(os.path.join(ft_dir, f"cm_baseline_{src_name}_to_{tgt_name}.png"))
            plt.close()


            results.append(
                {
                    "model": model_name,
                    "source": src_name,
                    "target": tgt_name,
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
        mean_impr = summary_df["improvement"].mean()
        overall = summary_df["transfer_acc"].mean() - summary_df["baseline_acc"].mean()
        print(f"Average improvement across experiments: {mean_impr:+.4f}")
        print(f"Overall transfer vs baseline: {overall:+.4f}")
        
        
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device).strip()

    
    if args.auto_select:
        # Build tiny loaders once to assemble summary (reuse your existing dataset creation if needed)
        # We use the source loaders because they always exist in pretraining, else target.
        try:
            # Reuse your pretraining loaders quickly (small impact)
            if args.data_name == 'Battery_inconsistent':
                from battery_dataset_loader import load_battery_dataset
                src_tr, src_val, tgt_tr, tgt_val, label_names, _df = load_battery_dataset(
                    csv_path=args.csv,
                    source_cathodes=args.source_cathode,
                    target_cathodes=args.target_cathode,
                    classification_label=args.classification_label,
                    batch_size=min(args.batch_size, 32),
                    sequence_length=args.sequence_length,
                )
                dls_for_peek = {'source_train': src_tr, 'source_val': src_val, 'target_train': tgt_tr, 'target_val': tgt_val}
            else:
                from SequenceDatasets import Dataset
                import torch
                if isinstance(args.transfer_task[0], str):
                    args.transfer_task = eval("".join(args.transfer_task))
                _src_tr, _src_val, _tgt_tr, _tgt_val, _ = Dataset(
                    args.data_dir, args.transfer_task, args.inconsistent, args.normlizetype
                ).data_split(transfer_learning=True)
                g = torch.Generator()
                dls_for_peek = {
                    'source_train': torch.utils.data.DataLoader(_src_tr, batch_size=min(args.batch_size, 64), shuffle=True, generator=g),
                    'source_val': torch.utils.data.DataLoader(_src_val, batch_size=min(args.batch_size, 64), shuffle=False),
                    'target_train': torch.utils.data.DataLoader(_tgt_tr, batch_size=min(args.batch_size, 64), shuffle=True, generator=g),
                    'target_val': torch.utils.data.DataLoader(_tgt_val, batch_size=min(args.batch_size, 64), shuffle=False),
                }
        except Exception:
            dls_for_peek = {'source_train': None, 'source_val': None, 'target_train': None, 'target_val': None}

        num_summary = _build_numeric_summary(dls_for_peek, args)
        text_ctx = (args.llm_context or f"{args.data_name}; transfer={bool(getattr(args,'pretrained',False))}; label_inconsistent={getattr(args,'inconsistent',False)}.").strip()
        llm_cfg = select_config(text_context=text_ctx,
                                num_summary=num_summary,
                                backend=args.llm_backend,
                                model=args.llm_model)
        
        

        from datetime import datetime as _dt
        import os as _os, json as _json
        _llm_stamp = _dt.now().strftime("%Y%m%d_%H%M%S")
        _llm_root = _os.path.join("checkpoint", f"llm_run_{_llm_stamp}")
        _os.makedirs(_llm_root, exist_ok=True)
        _llm_cfg_global = _os.path.join(_llm_root, "llm_selected_config.json")
        with open(_llm_cfg_global, "w") as _f:
            _json.dump({
                "llm_choice": llm_cfg,
                "inputs": {"text_context": text_ctx, "numeric_summary": num_summary},
                "timestamp": _llm_stamp
            }, _f, indent=2)
        print("🤖 LLM-selected configuration:")
        print(_json.dumps(llm_cfg, indent=2))
        print(f"📝 Rationale: {llm_cfg.get('rationale','(none)')}")
        print(f"📁 Saved LLM config to: {_llm_cfg_global}")

        args.model_name = llm_cfg["model_name"]
        args.method = 'sngp' if llm_cfg.get("sngp", False) else 'deterministic'
        if llm_cfg.get("openmax", False):
            args.model_name = "cnn_openmax"
        args.droprate = llm_cfg.get("dropout", getattr(args, "droprate", 0.3))
        args.lr = llm_cfg.get("learning_rate", args.lr)
        args.batch_size = int(llm_cfg.get("batch_size", args.batch_size))
        args.lambda_src = float(llm_cfg.get("lambda_src", getattr(args, "lambda_src", 1.0)))
        if hasattr(args, "bottleneck_num"):
            args.bottleneck_num = int(llm_cfg.get("bottleneck", getattr(args, "bottleneck_num", 256)))
            

        args.llm_cfg_inputs = {"text_context": text_ctx, "numeric_summary": num_summary}
        args.llm_cfg = llm_cfg
        args.llm_cfg_stamp = _llm_stamp

        

    if args.auto_select and args.llm_compare:
        import copy, glob, time, shutil, pandas as _pd, json as _json, os as _os

        base_args = copy.deepcopy(args)
        candidates = []

        # 1) The LLM pick (already applied to args)
        candidates.append(("llm_pick", copy.deepcopy(args)))

        # 2) Deterministic CNN baseline (no SA/OpenMax/SNGP)
        det = copy.deepcopy(base_args)
        det.model_name = "cnn_features_1d"
        det.method = "deterministic"
        det.droprate = min(getattr(base_args, "droprate", 0.3), 0.3)
        det.tag = (getattr(det, "tag", "") + "_detcnn_" + args.llm_cfg_stamp).strip("_")
        candidates.append(("deterministic_cnn", det))

        # 3) SNGP WideResNet + SA (strong calibrated model)
        sngp = copy.deepcopy(base_args)
        sngp.model_name = "WideResNet_sa"
        sngp.method = "sngp"
        sngp.droprate = 0.3
        sngp.tag = (getattr(sngp, "tag", "") + "_sngp_wrn_sa_" + args.llm_cfg_stamp).strip("_")
        candidates.append(("sngp_wrn_sa", sngp))

        _cmp_dir = _os.path.join("checkpoint", f"llm_run_{args.llm_cfg_stamp}", "compare")
        _os.makedirs(_cmp_dir, exist_ok=True)

        leaderboard_rows = []

        def _collect_latest_summary(copy_prefix: str) -> tuple[str, float]:
            summaries = sorted(glob.glob(_os.path.join("checkpoint", "summary_*.csv")), key=os.path.getmtime)
            if not summaries:
                return ("", float("nan"))
            latest = summaries[-1]
            dst = os.path.join(_cmp_dir, f"{copy_prefix}_{os.path.basename(latest)}")
            try:
                shutil.copy2(latest, dst)
            except Exception:
                dst = latest
            try:
                df = _pd.read_csv(latest)
                if "improvement" in df.columns:
                    avg_imp = float(_pd.to_numeric(df["improvement"], errors="coerce").mean())
                elif {"transfer_acc","baseline_acc"}.issubset(df.columns):
                    avg_imp = float((_pd.to_numeric(df["transfer_acc"], errors="coerce") - _pd.to_numeric(df["baseline_acc"], errors="coerce")).mean())
                else:
                    avg_imp = float("nan")
            except Exception:
                avg_imp = float("nan")
            return (dst, avg_imp)
        for tag, cfg in candidates:
            print(f"\n===== LLM comparison run: {tag} =====")
            if cfg.data_name == 'Battery_inconsistent':
                run_battery_experiments(cfg)
            else:
                run_cwru_experiments(cfg)

            time.sleep(0.5)

            copied_path, avg_imp = _collect_latest_summary(copy_prefix=tag)
            leaderboard_rows.append({
                "tag": tag,
                "model_name": cfg.model_name,
                "method": getattr(cfg, "method", "deterministic"),
                "droprate": getattr(cfg, "droprate", None),
                "lr": getattr(cfg, "lr", None),
                "batch_size": getattr(cfg, "batch_size", None),
                "lambda_src": getattr(cfg, "lambda_src", None),
                "summary_csv": copied_path,
                "avg_improvement": avg_imp,
            })

        _llm_root = _os.path.join("checkpoint", f"llm_run_{args.llm_cfg_stamp}")
        _leader_csv = _os.path.join(_llm_root, "llm_leaderboard.csv")
        _leader_json = _os.path.join(_llm_root, "llm_leaderboard.json")
        _pd.DataFrame(leaderboard_rows).to_csv(_leader_csv, index=False)
        with open(_leader_json, "w") as _f:
            _json.dump(leaderboard_rows, _f, indent=2)

        _valid = [r for r in leaderboard_rows if not (r["avg_improvement"] != r["avg_improvement"])]
        if _valid:
            best = max(_valid, key=lambda r: r["avg_improvement"])
            print("\n🏆 Leaderboard (avg improvement over baseline):")
            for r in sorted(_valid, key=lambda x: x["avg_improvement"], reverse=True):
                print(f" - {r['tag']:>18s}: {r['avg_improvement']:+.4f}  ({r['summary_csv']})")
            with open(_os.path.join(_llm_root, "winner.json"), "w") as _f:
                _json.dump(best, _f, indent=2)
            print(f"✅ Best configuration: {best['tag']} ({best['model_name']}, method={best['method']})")
            print(f"🧾 Proof files:\n  - {_leader_csv}\n  - {_leader_json}\n  - {_os.path.join(_llm_root, 'winner.json')}")
        else:
            print("\n⚠️ Could not compute a valid leaderboard (no summaries found).")

        import sys as _sys
        _sys.exit(0)

    if args.data_name == 'Battery_inconsistent':
        run_battery_experiments(args)
    else:
        run_cwru_experiments(args)
            


if __name__ == '__main__':
    main()
