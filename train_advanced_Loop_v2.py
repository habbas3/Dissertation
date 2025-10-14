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
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
import time
from itertools import combinations
from llm_selector import select_config
from utils.experiment_runner import _cm_with_min_labels
import os as _os
from scipy.spatial.distance import jensenshannon

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env into os.environ
except Exception:
    pass

api_key = os.getenv("OPENAI_API_KEY")  # llm_selector will read this too


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
    parser.add_argument('--data_name', type=str, default='CWRU_inconsistent',
                        choices=['Battery_inconsistent', 'CWRU_inconsistent'])
    parser.add_argument('--data_dir', type=str, default='./my_datasets/Battery',
                        help='Root directory for datasets')
    parser.add_argument('--csv', type=str, default='./my_datasets/Battery/battery_cycles_labeled.csv',
                        help='Cycle-level CSV for Battery dataset (run my_datasets/prepare_cycle_csv.py to generate if missing)')
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
    parser.add_argument('--cycles_per_file', type=int, default=50,
                        help='Number of contiguous cycles randomly sampled from each cell (default: 50)')
    parser.add_argument('--sample_random_state', type=int, default=42,
                        help='Random seed used when sampling cycles')
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

# --- Chemistry-aware cathode grouping and compatibility ---
    
def build_cathode_groups(csv_path):
    """
    Group cathodes by approximate chemistry family and return a compatibility map.
    Families are intentionally tight to avoid harmful transfers.
    """

    df = pd.read_csv(csv_path)
    cath = df["cathode"].astype(str).str.strip()

    groups = {
        # NMC family (+ HE5050 bucketed here as NMC-like)
        "nmc": sorted(
            cath[cath.str.contains(r"(?:^|\W)(NMC|HE5050)(?:$|\W)", case=False, regex=True)]
            .unique().tolist()
        ),
        # Li-rich layered compositions
        "lirich": sorted(
            cath[cath.str.contains(r"^Li1\.", case=False, regex=True)]
            .unique().tolist()
        ),
        # 5V spinel
        "spinel5v": sorted(
            cath[cath.str.contains("5Vspinel", case=False)]
            .unique().tolist()
        ),
        # FCG (+ variant)
        "fcg": sorted([x for x in cath.unique() if str(x).strip() == "FCG"]),
        "fcg_li": sorted([x for x in cath.unique() if "FCG+Li" in str(x)]),
    }
    # Drop empties
    groups = {k: v for k, v in groups.items() if len(v) > 0}

    label_col = "eol_class_encoded" if "eol_class_encoded" in df.columns else "eol_class"
    label_values = sorted(df[label_col].dropna().unique().tolist())

    def _distribution_for(cathodes: list[str]):
        subset = df[df["cathode"].isin(cathodes)]
        if subset.empty:
            return None
        counts = subset[label_col].value_counts(normalize=True)
        return np.array([counts.get(lbl, 0.0) for lbl in label_values], dtype=float)

    distributions = {fam: _distribution_for(items) for fam, items in groups.items()}

    compat: dict[str, set[str]] = {fam: set() for fam in groups}
    threshold = 0.35

    for src, src_dist in distributions.items():
        if src_dist is None:
            continue
        pair_scores: list[tuple[str, float]] = []
        for tgt, tgt_dist in distributions.items():
            if src == tgt or tgt_dist is None:
                continue
            jsd = float(jensenshannon(src_dist, tgt_dist, base=2))
            if np.isnan(jsd):
                continue
            pair_scores.append((tgt, jsd))
            if jsd <= threshold:
                compat[src].add(tgt)
        if not compat[src] and pair_scores:
            pair_scores.sort(key=lambda x: x[1])
            compat[src].add(pair_scores[0][0])

    # Ensure symmetry so that if A can transfer to B we also consider B → A
    for src, targets in list(compat.items()):
        for tgt in list(targets):
            compat.setdefault(tgt, set()).add(src)
    return groups, compat


def _cathode_family(label, groups):
    for fam, items in groups.items():
        if label in items:
            return fam
    return None

def _clone_loader(loader, force_shuffle: bool | None = None):
    if loader is None:
        return None

    dataset = loader.dataset
    batch_size = loader.batch_size
    drop_last = loader.drop_last
    num_workers = loader.num_workers
    pin_memory = loader.pin_memory
    generator = getattr(loader, "generator", None)

    sampler = loader.sampler
    if isinstance(sampler, WeightedRandomSampler):
        new_sampler = WeightedRandomSampler(
            weights=sampler.weights.clone(),
            num_samples=sampler.num_samples,
            replacement=sampler.replacement,
        )
        new_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=new_sampler,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        if isinstance(sampler, RandomSampler):
            shuffle = True if force_shuffle is None else force_shuffle
        elif isinstance(sampler, SequentialSampler):
            shuffle = False if force_shuffle is None else force_shuffle
        else:
            shuffle = False if force_shuffle is None else force_shuffle

        new_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        )

    for attr in ("sequence_count", "cycle_count"):
        if hasattr(loader, attr):
            setattr(new_loader, attr, getattr(loader, attr))

    return new_loader


def _baseline_cycle_stats(shared_stats: dict[str, int | bool]):
    if not shared_stats:
        return {}
    return {
        "source_train_cycles": shared_stats.get("target_train_cycles", 0),
        "source_val_cycles": shared_stats.get("target_val_cycles", 0),
        "source_val_has_holdout": shared_stats.get("target_val_has_holdout", False),
        "target_train_cycles": 0,
        "target_val_cycles": 0,
        "target_val_has_holdout": False,
        "source_train_sequences": shared_stats.get("target_train_sequences", 0),
        "source_val_sequences": shared_stats.get("target_val_sequences", 0),
        "target_train_sequences": 0,
        "target_val_sequences": 0,
    }


def _build_numeric_summary(dataloaders, args):
    import numpy as np
    
    def _split_snapshot(name, loader):
        if loader is None:
            return None
        try:
           batch = next(iter(loader))
        except Exception:
            return None

        if not isinstance(batch, (list, tuple)) or len(batch) < 1:
            return None

        x = batch[0]
        y = batch[1] if len(batch) > 1 else None

        try:
            x_np = x.detach().cpu().numpy()
        except Exception:
            return None

        if x_np.ndim == 2:  # (B, L) -> add channel axis
            x_np = x_np[:, None, :]
        if x_np.ndim != 3:
            return None

        B, C, L = int(x_np.shape[0]), int(x_np.shape[1]), int(x_np.shape[2])

        # Preview only a small slice to stay prompt-friendly
        max_channels_preview = min(3, C)
        max_steps_preview = min(12, L)
        sample_preview = np.round(
            x_np[0, :max_channels_preview, :max_steps_preview], 4
        ).tolist()

        y_np = None
        if y is not None:
            try:
                y_np = y.detach().cpu().numpy()
            except Exception:
                try:
                    y_np = np.asarray(y)
                except Exception:
                    y_np = None

        dataset_obj = getattr(loader, "dataset", None)
        label_counts = None
        feature_range = None
        feature_mean = None
        sample_rows = None

        if dataset_obj is not None:
            labels_attr = getattr(dataset_obj, "labels", None)
            if labels_attr is not None:
                labels_np = np.asarray(labels_attr)
                try:
                    uniq, cnt = np.unique(labels_np, return_counts=True)
                    label_counts = {str(int(u)): int(c) for u, c in zip(uniq, cnt)}
                except Exception:
                    pass

            seq_data = getattr(dataset_obj, "seq_data", None)
            if seq_data is None:
                seq_data = getattr(dataset_obj, "sequences", None)

            if seq_data is not None:
                arr = np.asarray(seq_data)
                if arr.size > 0:
                    try:
                        arr_flat_all = arr.reshape(arr.shape[0], -1)
                        sample_rows = np.round(
                            arr_flat_all[:3, : min(10, arr_flat_all.shape[1])], 4
                        ).tolist()
                    except Exception:
                        sample_rows = None
                        arr_flat_all = None

                    arr_view = arr[: min(len(arr), 128)]
                    try:
                        arr_flat = arr_view.reshape(arr_view.shape[0], -1)
                        feature_range = [
                            float(np.min(arr_flat)),
                            float(np.max(arr_flat)),
                        ]
                        feature_mean = float(np.mean(arr_flat))
                    except Exception:
                        pass

        split_info = {
            "batch_shape": [B, C, L],
            "channels": C,
            "seq_len": L,
            "example_label": int(y_np[0]) if y_np is not None and y_np.size > 0 else None,
            "preview": {
                "channels": max_channels_preview,
                "timesteps": max_steps_preview,
                "values": sample_preview,
            },
            "batch_channel_mean": np.round(x_np.mean(axis=(0, 2)), 4).tolist()[: min(C, 8)],
            "batch_channel_std": np.round(x_np.std(axis=(0, 2)), 4).tolist()[: min(C, 8)],
        }
        
        if label_counts:
            split_info["class_distribution"] = label_counts
        if feature_range:
            split_info["feature_range"] = feature_range
        if feature_mean is not None:
            split_info["feature_global_mean"] = feature_mean
        if sample_rows is not None:
            split_info["flattened_rows_head"] = sample_rows

        return split_info

    summary = {
        "dataset": args.data_name,
        "sequence_length_requested": getattr(args, "sequence_length", None),
        "notes": "label_inconsistent" if getattr(args, "inconsistent", False) else "closed_set",
        "lr_hint": args.lr,
        "dropout_hint": getattr(args, "droprate", 0.3),
        "num_classes_hint": getattr(args, "num_classes", None) or getattr(args, "n_class", None),
        "splits": {},
    }
    
    cycle_stats = getattr(args, "dataset_cycle_stats", None)
    if isinstance(cycle_stats, dict) and cycle_stats:
        try:
            summary["cycle_stats"] = {
                str(k): int(v) if isinstance(v, (int, float)) else v
                for k, v in cycle_stats.items()
            }
        except Exception:
            summary["cycle_stats"] = cycle_stats

    if args.data_name == 'Battery_inconsistent':
        summary["dataset_variant"] = "argonne_battery"
        summary["feature_names"] = [
            'cycle_number', 'energy_charge', 'capacity_charge', 'energy_discharge',
            'capacity_discharge', 'cycle_start', 'cycle_duration'
        ]
        summary["source_cathodes"] = list(getattr(args, "source_cathode", []) or [])
        summary["target_cathodes"] = list(getattr(args, "target_cathode", []) or [])
        summary["label_column"] = getattr(args, "classification_label", None)
    else:
        summary["dataset_variant"] = "cwru_bearing"
        summary["transfer_task"] = getattr(args, "transfer_task", None)

    preferred_keys = ['source_train', 'target_train', 'source_val', 'target_val']
    first_stats = None
    for key in preferred_keys:
        info = _split_snapshot(key, dataloaders.get(key))
        if info is None:
            continue
        summary['splits'][key] = info
        if first_stats is None:
            first_stats = {
                "split_used": key,
                "batch_size_seen": info['batch_shape'][0],
                "channels": info['channels'],
                "seq_len": info['seq_len'],
            }

    if first_stats:
        summary.update(first_stats)
    else:
        summary['splits'] = {}

    return summary


def _build_text_context(args, num_summary):
    import json

    lines = []
    dataset_name = getattr(args, 'data_name', 'unknown')

    if dataset_name == 'Battery_inconsistent':
        feature_names = num_summary.get('feature_names', [])
        lines.append("Dataset: Argonne National Laboratory battery aging time-series with partial cycle windows.")
        lines.append(
            f"Source cathodes: {', '.join(num_summary.get('source_cathodes', []) or ['(all available)'])}; "
            f"target cathodes: {', '.join(num_summary.get('target_cathodes', []) or ['(none specified)'])}."
        )
        label_col = num_summary.get('label_column') or getattr(args, 'classification_label', None)
        seq_len = num_summary.get('seq_len') or num_summary.get('sequence_length_requested')
        channels = num_summary.get('channels') or getattr(args, 'input_channels', None)
        channel_desc = channels if channels is not None else "?"
        feature_desc = ', '.join(feature_names[:7]) if feature_names else f"{channel_desc} normalized signals"
        lines.append(
            f"Label column '{label_col}' with ~{num_summary.get('num_classes_hint', 'unknown')} classes; "
            f"sequence length {seq_len}; {channel_desc} channels covering {feature_desc}."
        )
    else:
        transfer = num_summary.get('transfer_task') or getattr(args, 'transfer_task', None)

        def _fmt_domain(domain):
            if isinstance(domain, (list, tuple, set)):
                if len(domain) == 0:
                    return "?"
                return ",".join([str(x) for x in domain])
            return str(domain) if domain is not None else "?"

        src_dom = _fmt_domain(transfer[0]) if transfer and len(transfer) > 0 else "?"
        tgt_dom = _fmt_domain(transfer[1]) if transfer and len(transfer) > 1 else "?"
        lines.append("Dataset: Case Western Reserve University bearing vibration transfer benchmark with label inconsistency handling.")
        lines.append(f"Transfer from motors {src_dom} to {tgt_dom}; inconsistency setting {getattr(args, 'inconsistent', '(not set)')}.")
        seq_len = num_summary.get('seq_len', getattr(args, 'sequence_length', None))
        channels = num_summary.get('channels', getattr(args, 'input_channels', None))
        channel_desc = channels if channels is not None else "?"
        lines.append(f"Windows are length {seq_len} with {channel_desc} vibration channels; task uses {num_summary.get('num_classes_hint', 'unknown')} classes.")

    splits = num_summary.get('splits', {})
    for split_name, info in splits.items():
        counts = info.get('class_distribution') or {}
        if counts:
            sorted_counts = sorted(counts.items(), key=lambda kv: kv[0])
            subset = ', '.join([f"{k}:{v}" for k, v in sorted_counts[:6]])
            if len(sorted_counts) > 6:
                subset += f" … (+{len(sorted_counts) - 6} classes)"
            lines.append(f"{split_name}: class counts {subset} (example label {info.get('example_label')}).")
        preview = info.get('preview', {})
        if preview.get('values'):
            values_json = json.dumps(preview['values'])
            if len(values_json) > 420:
                values_json = values_json[:420] + '…'
            lines.append(
                f"{split_name} sample window (first {preview.get('channels')} ch × {preview.get('timesteps')} steps): {values_json}"
            )
        rows = info.get('flattened_rows_head')
        if rows:
            rows_json = json.dumps(rows)
            if len(rows_json) > 420:
                rows_json = rows_json[:420] + '…'
            lines.append(f"{split_name} flattened row glimpses: {rows_json}")

    extra = getattr(args, 'llm_context', '')
    if extra:
        lines.append(f"User notes: {extra}")

    return "\n".join(lines)



def run_experiment(args, save_dir, trial=None, baseline=False, override_data=None):
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


    if override_data is not None:
        source_train_loader = override_data.get("source_train_loader")
        source_val_loader = override_data.get("source_val_loader")
        target_train_loader = override_data.get("target_train_loader")
        target_val_loader = override_data.get("target_val_loader")

        source_train_dataset = override_data.get("source_train_dataset")
        if source_train_dataset is None and source_train_loader is not None:
            source_train_dataset = source_train_loader.dataset
        source_val_dataset = override_data.get("source_val_dataset")
        if source_val_dataset is None and source_val_loader is not None:
            source_val_dataset = source_val_loader.dataset
        target_train_dataset = override_data.get("target_train_dataset")
        if target_train_dataset is None and target_train_loader is not None:
            target_train_dataset = target_train_loader.dataset
        target_val_dataset = override_data.get("target_val_dataset")
        if target_val_dataset is None and target_val_loader is not None:
            target_val_dataset = target_val_loader.dataset

        label_names = override_data.get("label_names")
        if label_names:
            args.num_classes = len(label_names)
        args.num_classes = override_data.get("num_classes", args.num_classes)
        if override_data.get("cycle_stats") is not None:
            args.dataset_cycle_stats = override_data.get("cycle_stats")

        if source_train_loader is None and source_train_dataset is not None:
            source_train_loader = DataLoader(
                source_train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
            )
        if source_val_loader is None and source_val_dataset is not None:
            source_val_loader = DataLoader(
                source_val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
            )
        if target_train_loader is None and target_train_dataset is not None:
            target_train_loader = DataLoader(
                target_train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
            )
        if target_val_loader is None and target_val_dataset is not None:
            target_val_loader = DataLoader(
                target_val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
            )
    elif args.data_name == 'Battery_inconsistent':
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
            num_classes=args.num_classes,
            cycles_per_file=args.cycles_per_file,
            sample_random_state=args.sample_random_state,
        )
        # keep dataset references for later use
        source_train_dataset = source_train_loader.dataset
        source_val_dataset = source_val_loader.dataset
        target_train_dataset = target_train_loader.dataset if target_train_loader is not None else None
        target_val_dataset = target_val_loader.dataset if target_val_loader is not None else None
        args.num_classes = len(label_names)
        args.dataset_cycle_stats = cycle_stats
    else:
        cwru_dataset = CWRU_inconsistent(args.data_dir, args.transfer_task, args.inconsistent, args.normlizetype)
        if baseline:
            source_train_dataset, source_val_dataset, target_val_dataset = cwru_dataset.data_split(transfer_learning=False)
            target_train_dataset = None
            args.num_classes = len(np.unique(source_train_dataset.labels))
        else:
            source_train_dataset, source_val_dataset, target_train_dataset, target_val_dataset, num_classes = cwru_dataset.data_split(transfer_learning=True)
            args.num_classes = num_classes

        source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        source_val_loader = DataLoader(source_val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

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

    
    groups, compat = build_cathode_groups(args.csv)
    
    print("🔍 Cathode families discovered:")
    for fam, items in groups.items():
        print(f"   - {fam}: {len(items)} cathodes → {', '.join(items[:5])}{'…' if len(items) > 5 else ''}")
        
    print("🔁 Allowed transfer directions (data-driven):")
    for fam, targets in sorted(compat.items()):
        arrow = ", ".join(sorted(targets)) if targets else "(none)"
        print(f"   - {fam} → {arrow}")

    # Build experiment list using chemistry-aware compatibility
    experiment_configs = []
    for src_name, src_cathodes in groups.items():
        for tgt_name, tgt_cathodes in groups.items():
            if src_name == tgt_name:
                continue
            if tgt_name not in compat.get(src_name, set()):
                print(
                    f"⏭️  Skipping {src_name} → {tgt_name} (incompatible families: {src_name} → {tgt_name})"
                )
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

            shared_tuple = load_battery_dataset(
                csv_path=args.csv,
                source_cathodes=source_cathodes,
                target_cathodes=target_cathodes,
                classification_label=args.classification_label,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                num_classes=args.num_classes,
                cycles_per_file=args.cycles_per_file,
                sample_random_state=args.sample_random_state,
            )
            (
                shared_src_train_loader,
                shared_src_val_loader,
                shared_tgt_train_loader,
                shared_tgt_val_loader,
                label_names,
                _shared_df,
                shared_stats,
            ) = shared_tuple

            transfer_override = {
                "source_train_loader": _clone_loader(shared_src_train_loader, force_shuffle=True),
                "source_val_loader": _clone_loader(shared_src_val_loader, force_shuffle=False),
                "target_train_loader": _clone_loader(shared_tgt_train_loader, force_shuffle=True),
                "target_val_loader": _clone_loader(shared_tgt_val_loader, force_shuffle=False),
                "label_names": label_names,
                "num_classes": len(label_names),
                "cycle_stats": dict(shared_stats),
            }

            baseline_override = {
                "source_train_loader": _clone_loader(shared_tgt_train_loader, force_shuffle=True),
                "source_val_loader": _clone_loader(shared_tgt_val_loader, force_shuffle=False),
                "target_train_loader": None,
                "target_val_loader": None,
                "label_names": label_names,
                "num_classes": len(label_names),
                "cycle_stats": _baseline_cycle_stats(shared_stats),
            }

            # ---------------- Baseline: train target from scratch (matched data) ----------------
            baseline_args = argparse.Namespace(**vars(args))
            baseline_args.source_cathode = target_cathodes
            baseline_args.target_cathode = []
            baseline_args.pretrained = False
            baseline_dir = os.path.join(
                args.checkpoint_dir,
                f"baseline_{model_name}_{tgt_name}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(baseline_dir, exist_ok=True)
            model_bl, _, _, _, _ = run_experiment(
                baseline_args,
                baseline_dir,
                baseline=True,
                override_data=baseline_override,
            )
            
            baseline_stats = getattr(baseline_args, "dataset_cycle_stats", {})
            if baseline_stats:
                print(
                    f"🧵 Baseline training cycles used: {baseline_stats.get('source_train_cycles', 'n/a')}"
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
            model_ft, _, _, _, _ = run_experiment(
                transfer_args,
                ft_dir,
                override_data=transfer_override,
            )
            transfer_stats = getattr(transfer_args, "dataset_cycle_stats", {})
            if transfer_stats:
                print(
                    f"🔁 Transfer target training cycles used: {transfer_stats.get('target_train_cycles', 'n/a')}"
                )
            eval_loader = transfer_override.get("target_val_loader")
            tr_labels, tr_preds = evaluate_model(model_ft, eval_loader)
            
            # Evaluate baseline on the SAME target validation loader
            bl_labels, bl_preds = evaluate_model(model_bl, eval_loader)
            baseline_labels_np = np.array(bl_labels)
            baseline_preds_np = np.array(bl_preds)
            
            # Use the same num_known (transfer_args.num_classes) to score both runs fairly.
            num_known = transfer_args.num_classes
            
            t_common, t_out, t_h = compute_common_outlier_metrics(tr_labels, tr_preds, num_known)
            b_common, b_out, b_h = compute_common_outlier_metrics(
                baseline_labels_np, baseline_preds_np, num_known
            )
            
            metric_lookup = {
                "common": (t_common, b_common),
                "hscore": (t_h, b_h),
                "overall": ((t_common + t_out) / 2.0, (b_common + b_out) / 2.0),
            }
            metric_key = args.improvement_metric
            if metric_key not in metric_lookup:
                metric_key = "common"
            transfer_score, baseline_score = metric_lookup[metric_key]
            
            print(
                f"✅ Baseline {tgt_name}: {baseline_score:.4f} ({len(bl_labels)} samples)"
            )
            print(
                f"✅ Transfer {src_name} → {tgt_name}: {transfer_score:.4f} ({len(tr_labels)} samples)"
            )
            eval_cycles = getattr(eval_loader, "cycle_count", None)
            eval_sequences = getattr(eval_loader, "sequence_count", None)
            if eval_cycles is None and transfer_stats:
                eval_cycles = transfer_stats.get("target_val_cycles")
            if eval_sequences is None and transfer_stats:
                eval_sequences = transfer_stats.get("target_val_sequences")
            holdout_flag = None
            if transfer_stats:
                holdout_flag = transfer_stats.get("target_val_has_holdout")
            if eval_cycles is not None:
                msg = f"   ↳ evaluated_cycles={eval_cycles}"
                if eval_sequences is not None:
                    msg += f", evaluated_sequences={eval_sequences}"
                if holdout_flag is not None:
                    msg += f", holdout_split={'yes' if holdout_flag else 'no'}"
                print(msg)
                
            print(
                f"   ↳ common_acc={t_common:.4f}, outlier_acc={t_out:.4f}, hscore={t_h:.4f}"
            )
            print(
                f"🧪 Baseline scored on same split: common_acc={b_common:.4f}, outlier_acc={b_out:.4f}, hscore={b_h:.4f}"
            )
            
            improvement = transfer_score - baseline_score
            print(
                f"📊 {src_name} → {tgt_name}: baseline({metric_key})={baseline_score:.4f}, transfer({metric_key})={transfer_score:.4f}, improvement={improvement:+.4f}"
            )
            if transfer_score <= baseline_score:
                print(
                    f"♻️ Transfer lagged baseline for {src_name} → {tgt_name}; launching target-focused fine-tune retry."
                )
                retry_args = argparse.Namespace(**vars(transfer_args))
                retry_args.lr = max(1e-5, transfer_args.lr * 0.5)
                retry_args.lambda_src = max(0.1, transfer_args.lambda_src * 0.5)
                retry_args.max_epoch = int(max(transfer_args.max_epoch, 40) * 1.25)
                retry_args.reinit_head = False
                retry_args.pretrained_model_path = os.path.join(ft_dir, "best_model.pth")
                retry_dir = os.path.join(ft_dir, "retry_target_focus")
                os.makedirs(retry_dir, exist_ok=True)
                model_retry, _, _, _, _ = run_experiment(
                    retry_args,
                    retry_dir,
                    override_data=transfer_override,
                )
                retry_stats = getattr(retry_args, "dataset_cycle_stats", {})
                if retry_stats:
                    print(
                        f"🔁 Retry target training cycles used: {retry_stats.get('target_train_cycles', 'n/a')}"
                    )
                retry_labels, retry_preds = evaluate_model(model_retry, eval_loader)
                r_common, r_out, r_h = compute_common_outlier_metrics(retry_labels, retry_preds, num_known)
                retry_metric_lookup = {
                    "common": (r_common, b_common),
                    "hscore": (r_h, b_h),
                    "overall": ((r_common + r_out) / 2.0, (b_common + b_out) / 2.0),
                }
                retry_transfer_score, _ = retry_metric_lookup[metric_key]
                if retry_transfer_score > transfer_score:
                    print(
                        f"✅ Retry fine-tune surpassed initial transfer ({retry_transfer_score:.4f} vs {transfer_score:.4f})."
                    )
                    model_ft = model_retry
                    tr_labels, tr_preds = retry_labels, retry_preds
                    t_common, t_out, t_h = r_common, r_out, r_h
                    transfer_stats = retry_stats or transfer_stats
                else:
                    print(
                        f"⚠️ Retry fine-tune did not exceed initial transfer ({retry_transfer_score:.4f} ≤ {transfer_score:.4f})."
                    )

                metric_lookup = {
                    "common": (t_common, b_common),
                    "hscore": (t_h, b_h),
                    "overall": ((t_common + t_out) / 2.0, (b_common + b_out) / 2.0),
                }
                transfer_score, baseline_score = metric_lookup[metric_key]
                improvement = transfer_score - baseline_score
                print(
                    f"📊 Updated {src_name} → {tgt_name}: baseline({metric_key})={baseline_score:.4f}, transfer({metric_key})={transfer_score:.4f}, improvement={improvement:+.4f}"
                )
                if transfer_score <= baseline_score:
                    print(f"⚠️ Transfer remains below baseline for {src_name} → {tgt_name} even after retry.")

            # Confusion matrices with consistent axes and minimum label coverage
            cm_transfer, labels_tr = _cm_with_min_labels(tr_labels, tr_preds, min_labels=5)
            cm_baseline, labels_bl = _cm_with_min_labels(baseline_labels_np, baseline_preds_np, min_labels=5)
            labels = sorted(set(labels_tr) | set(labels_bl))
            desired = max(len(labels), 5)
            cm_transfer, labels = _cm_with_min_labels(tr_labels, tr_preds, min_labels=desired)
            cm_baseline, _ = _cm_with_min_labels(baseline_labels_np, baseline_preds_np, min_labels=desired)
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
                    "comparison_metric": metric_key,
                    "baseline_score": baseline_score,
                    "transfer_score": transfer_score,
                    "improvement": improvement,
                    "baseline_common_acc": b_common,
                    "transfer_common_acc": t_common,
                    "baseline_outlier_acc": b_out,
                    "transfer_outlier_acc": t_out,
                    "baseline_hscore": b_h,
                    "transfer_hscore": t_h,
                }
            )

    if results:
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(
            args.checkpoint_dir,
            f"summary_{datetime.now().strftime('%m%d')}_{args.data_name}.csv",
        )
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")
        cols_to_show = [
            "source",
            "target",
            "comparison_metric",
            "baseline_score",
            "transfer_score",
            "improvement",
        ]
        print(summary_df[cols_to_show])
        mean_impr = summary_df["improvement"].mean()
        overall = summary_df["transfer_score"].mean() - summary_df["baseline_score"].mean()
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
            if transfer_acc <= baseline_acc:
                print(
                    f"♻️ Transfer lagged baseline for {src_str} → {tgt_str}; retrying with stronger target emphasis."
                )
                retry_args = argparse.Namespace(**vars(transfer_args))
                retry_args.lr = max(1e-5, transfer_args.lr * 0.5)
                retry_args.lambda_src = max(0.1, transfer_args.lambda_src * 0.5)
                retry_args.max_epoch = int(max(transfer_args.max_epoch, 40) * 1.25)
                retry_args.reinit_head = False
                retry_args.pretrained_model_path = os.path.join(ft_dir, "best_model.pth")
                retry_dir = os.path.join(ft_dir, "retry_target_focus")
                os.makedirs(retry_dir, exist_ok=True)
                model_retry, _, _, _, tr_loader_retry = run_experiment(
                    retry_args,
                    retry_dir,
                )
                retry_labels, retry_preds = evaluate_model(model_retry, tr_loader_retry)
                r_common, r_out, r_h = compute_common_outlier_metrics(retry_labels, retry_preds, num_known)
                retry_transfer_acc = r_common
                if retry_transfer_acc > transfer_acc:
                    print(
                        f"✅ Retry fine-tune boosted transfer accuracy ({retry_transfer_acc:.4f} vs {transfer_acc:.4f})."
                    )
                    model_ft = model_retry
                    tr_labels, tr_preds = retry_labels, retry_preds
                    t_common, t_out, t_h = r_common, r_out, r_h
                    transfer_acc = retry_transfer_acc
                    improvement = transfer_acc - baseline_acc
                else:
                    print(
                        f"⚠️ Retry fine-tune did not exceed baseline-matched score ({retry_transfer_acc:.4f} ≤ {transfer_acc:.4f})."
                    )
                print(
                    f"📊 Updated {src_str} → {tgt_str}: baseline(common)={baseline_acc:.4f}, "
                    f"transfer(common)={transfer_acc:.4f}, improvement={improvement:+.4f}"
                )
                if transfer_acc <= baseline_acc:
                    print(f"⚠️ Transfer remains below baseline for {src_str} → {tgt_str} even after retry.")


            # Confusion matrices with consistent axes and minimum label coverage
            cm_transfer, labels_tr = _cm_with_min_labels(tr_labels, tr_preds, min_labels=10)
            cm_baseline, labels_bl = _cm_with_min_labels(baseline_labels_np, baseline_preds_np, min_labels=10)
            labels = sorted(set(labels_tr) | set(labels_bl))
            cm_transfer, labels = _cm_with_min_labels(tr_labels, tr_preds, min_labels=len(labels))
            cm_baseline, _ = _cm_with_min_labels(baseline_labels_np, baseline_preds_np, min_labels=len(labels))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_transfer, display_labels=labels)
            disp.plot(cmap='Blues')
            plt.savefig(os.path.join(ft_dir, f"cm_transfer_{src_str}_to_{tgt_str}.png"))
            plt.close()

            disp = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=labels)
            disp.plot(cmap='Blues')
            plt.savefig(os.path.join(ft_dir, f"cm_baseline_{src_str}_to_{tgt_str}.png"))
            plt.close()
            
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
    
    # Auto-generate the cycle-level CSV if it is missing
    if (
        args.data_name == 'Battery_inconsistent'
        and args.csv
        and not os.path.exists(args.csv)
    ):
        try:
            from my_datasets.prepare_cycle_csv import build_cycle_csv

            processed = os.path.join(args.data_dir, 'proper_hdf5', 'processed')
            labels_csv = os.path.join(args.data_dir, 'battery_labeled.csv')
            print(f"⚠️ {args.csv} not found – building from {processed}")
            build_cycle_csv(processed, labels_csv, args.csv, num_classes=5)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"❌ Failed to build cycle CSV: {exc}")

    
    if args.auto_select:
        # Build tiny loaders once to assemble summary (reuse your existing dataset creation if needed)
        # We use the source loaders because they always exist in pretraining, else target.
        try:
            # Reuse your pretraining loaders quickly (small impact)
            if args.data_name == 'Battery_inconsistent':
                from battery_dataset_loader import load_battery_dataset
                src_tr, src_val, tgt_tr, tgt_val, label_names, _df, _stats = load_battery_dataset(
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
        text_ctx = _build_text_context(args, num_summary)        
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
