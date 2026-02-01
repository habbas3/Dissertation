#!/usr/bin/python
# -*- coding:utf-8 -*-

#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pickle
import shutil
import os
import gc
from typing import Optional
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
from llm_selector import select_config, run_ablation_suite
from utils.experiment_runner import _cm_with_min_labels
import os as _os
from scipy.spatial.distance import jensenshannon


def _cleanup_memory(tag: str = ""):
    """Best-effort memory cleanup to avoid kernel restarts on long runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()
    if tag:
        print(f"🧹 Memory cleanup: {tag}")


def _save_highres_confusion(cm, labels, path, title, normalize=True):
    labels = list(labels)
    fig, ax = plt.subplots(
        figsize=(4 + 0.45 * len(labels), 4 + 0.45 * len(labels)), dpi=320
    )
    cmap = plt.cm.Blues
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(len(labels)),
        yticks=range(len(labels)),
    )
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            _ = np.divide(cm.astype(float), row_sums, where=row_sums != 0)

    annot = [[str(int(cm[i, j])) for j in range(cm.shape[1])] for i in range(cm.shape[0])]

    thresh = cm.max() / 2.0 if cm.size else 0
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                annot[i][j],
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)



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


BASELINE_CWRU_TRANSFER_TASKS = [
    [[0], [1]],
    [[0], [2]],
    [[0], [3]],
    [[1], [0]],
    [[1], [2]],
    [[1], [3]],
    [[2], [0]],
    [[2], [1]],
    [[2], [3]],
    [[3], [0]],
    [[3], [1]],
    [[3], [2]],
]


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
    parser.add_argument('--method', type=str, default='auto',
                        choices=['deterministic', 'sngp', 'auto'],
                        help="Uncertainty head to use. 'auto' selects SNGP only when the dataset"
                             " exhibits open-set/outlier structure.")
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
    parser.add_argument('--warmup_epochs', type=int, default=3, help='linear-probe epochs before full fine-tuning')
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
    parser.add_argument('--no_confmat', action='store_true',
                        help='Skip confusion-matrix generation to reduce post-training memory usage')
    parser.add_argument('--confmat_dir', type=str, default='',
                        help='Optional directory to save confusion matrices (defaults to the run checkpoint folder)')
    parser.add_argument('--inconsistent', type=str, default='UAN')
    parser.add_argument('--model_name', type=str, default='cnn_features_1d')
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument('--input_channels', type=int, default=7)
    parser.add_argument('--classification_label', type=str, default='eol_class')
    parser.add_argument('--sequence_length', type=int, default=32)
    parser.add_argument('--cycles_per_file', type=int, default=15,
                        help='Number of early-life cycles to use for target-side baseline/transfer comparisons')
    parser.add_argument('--source_cycles_per_file', type=int, default=None,
                        help='Optional limit for source cathodes (defaults to all cycles when omitted)')
    parser.add_argument('--target_cycles_per_file', type=int, default=None,
                        help='Optional target limit; falls back to --cycles_per_file when omitted')
    parser.add_argument('--cycle_ablation', action='store_true',
                        help='Sweep cycle limits starting at cycle_ablation_start and increasing by cycle_ablation_step until'
                             ' validation accuracy no longer improves')
    parser.add_argument('--cycle_ablation_start', type=int, default=5,
                        help='Initial number of early cycles to use when --cycle_ablation is enabled')
    parser.add_argument('--cycle_ablation_step', type=int, default=10,
                        help='Increment to apply to the cycle horizon between ablation trials')
    parser.add_argument('--cycle_ablation_max', type=int, default=None,
                        help='Optional maximum cycle horizon to consider during ablation sweeps')
    parser.add_argument('--literature_context_file', type=str,
                        default='./references/joule_s2542-4351-22-00409-3.md',
                        help='Optional markdown/text snippet (e.g., Joule S2542-4351(22)00409-3) to provide chemistry-specific '
                             'guidance to the LLM prompt when using the Battery dataset')
    parser.add_argument('--sample_random_state', type=int, default=42,
                        help='Random seed used when sampling cycles')
    parser.add_argument('--transfer_task', type=str, default=json.dumps(BASELINE_CWRU_TRANSFER_TASKS),
                        help='CWRU transfer task as [[source],[target]] or list of such pairs (default: all 12 baseline combinations)')
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
    parser.add_argument('--lambda_src_decay_patience', type=int, default=5,
                        help='Epochs without target-val improvement before lambda_src is decayed')
    parser.add_argument('--lambda_src_decay_factor', type=float, default=0.5,
                        help='Multiplicative factor applied to lambda_src after each decay')
    parser.add_argument('--lambda_src_min', type=float, default=0.0,
                        help='Lower bound for lambda_src when decay scheduling is active')
    parser.add_argument('--lambda_src_warmup', type=int, default=0,
                        help='Number of epochs to wait before counting plateau epochs for lambda_src decay')
    parser.add_argument(
        '--improvement_metric',
        choices=['accuracy', 'common', 'hscore', 'overall'],
        default='accuracy',
        help=(
            "Metric used to compare transfer vs baseline on target_val. "
            "Accuracy is the default to keep transfer scores aligned with the observed confusion-matrix accuracy."
        ),
    )
    parser.add_argument(
        '--skip_retry',
        action='store_true',
        help='Deprecated (retries disabled for fairness comparisons).',
    )
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
    parser.add_argument('--llm_ablation', action='store_true', default=True,
                        help='Collect ablation configs (history on/off, limited-cycle prompts) for closed-loop evidence')
    parser.add_argument('--no-llm_ablation', dest='llm_ablation', action='store_false',
                        help='Disable ablation prompt collection and comparison candidates')
    parser.add_argument(
        '--compare_cycles',
        type=int,
        default=50,
        help='Fixed cycle horizon for non-cycle LLM comparisons (used in llm_compare runs).',
    )
    parser.add_argument(
        '--no_llm_history',
        dest='llm_allow_history',
        action='store_false',
        default=True,
        help='Disable leaderboard/history context when prompting the LLM.',
    )
    parser.add_argument(
        '--no_llm_chemistry',
        dest='llm_chemistry_feedback',
        action='store_false',
        default=True,
        help='Disable cathode chemistry hints when prompting the LLM.',
    )
    parser.add_argument('--llm_per_transfer', action='store_true', default=True,
                        help='Request an LLM pick for every transfer task/cathode pair instead of one global config')
    parser.add_argument('--ablation_cycle_limits', type=str, default='5,15,30,50,100',
                        help='Comma-separated early-cycle horizons (e.g., "5,15,30,50,100") to probe EOL prediction sensitivity')
    
        
    args = parser.parse_args()
    if args.data_name == 'CWRU_inconsistent' and args.data_dir == './my_datasets/Battery':
        args.data_dir = './my_datasets/CWRU_dataset'
    if isinstance(args.transfer_task, str):
        try:
            args.transfer_task = eval(args.transfer_task)
        except Exception:
            pass
    if args.auto_select:
        # Auto-selection implies we want per-transfer reasoning unless explicitly disabled.
        args.llm_per_transfer = args.llm_per_transfer or True
    return args

ARCHITECTURE_ORDER = [
    "cnn_features_1d",
    "cnn_features_1d_sa",
    "cnn_openmax",
    "WideResNet",
    "WideResNet_sa",
    "WideResNet_edited",
]


def _unique_preserve(seq):
    seen = set()
    ordered = []
    for item in seq:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _dataset_profile_from_dataset(ds, num_classes, sequence_length_hint=None):
    profile = {
        "total_samples": 0,
        "sequence_length": None,
        "channels": None,
        "known_fraction": 0.0,
        "outlier_fraction": 0.0,
        "unknown_count": 0,
        "class_counts": {},
    }

    if ds is None:
        return profile

    try:
        length = len(ds)
    except TypeError:
        length = 0
    profile["total_samples"] = int(max(length, 0))

    if length > 0:
        try:
            sample_x, _ = ds[0]
            if isinstance(sample_x, torch.Tensor):
                profile["channels"] = int(sample_x.shape[0])
                profile["sequence_length"] = int(sample_x.shape[-1])
            else:
                sample_x = np.asarray(sample_x)
                if sample_x.ndim >= 2:
                    profile["channels"] = int(sample_x.shape[0])
                    profile["sequence_length"] = int(sample_x.shape[-1])
        except Exception:
            pass

    labels = getattr(ds, "labels", None)
    if labels is None:
        return profile

    labels_np = np.asarray(labels)
    seq_len = getattr(ds, "sequence_length", None) or sequence_length_hint or 1
    seq_len = max(int(seq_len), 1)

    if labels_np.size >= seq_len:
        effective = labels_np[seq_len - 1: seq_len - 1 + profile["total_samples"]]
    else:
        effective = labels_np[-profile["total_samples"]:] if profile["total_samples"] > 0 else labels_np

    if effective.size == 0:
        return profile

    effective = effective.astype(int, copy=False)
    known_mask = effective < num_classes
    unknown_mask = ~known_mask

    profile["unknown_count"] = int(unknown_mask.sum())
    profile["outlier_fraction"] = float(profile["unknown_count"] / effective.size)
    profile["known_fraction"] = float(known_mask.sum() / effective.size)

    if known_mask.any():
        class_ids, counts = np.unique(effective[known_mask], return_counts=True)
        profile["class_counts"] = {int(c): int(n) for c, n in zip(class_ids, counts)}
    return profile


def _dataset_profile_from_loader(loader, num_classes, sequence_length_hint=None):
    ds = getattr(loader, "dataset", None) if loader is not None else None
    return _dataset_profile_from_dataset(ds, num_classes, sequence_length_hint)


def _decide_uncertainty_method(requested, profile, dataset_name=None):
    if requested != 'auto':
        return requested
    
    if dataset_name and 'cwru' in dataset_name.lower():
        return 'sngp'
    
    if profile.get("unknown_count", 0) > 0 or profile.get("outlier_fraction", 0.0) >= 0.05:
        return 'sngp'
    return 'deterministic'


def _choose_architectures(profile):
    if profile is None:
        return ARCHITECTURE_ORDER

    seq_len = profile.get("sequence_length") or 0
    channels = profile.get("channels") or 1
    total = profile.get("total_samples", 0)
    outlier_frac = profile.get("outlier_fraction", 0.0)

    preferred = []

    if total and total < 300:
        preferred.extend(["cnn_features_1d", "cnn_features_1d_sa"])
    elif seq_len and seq_len >= 256:
        preferred.extend(["WideResNet", "WideResNet_sa"])
    else:
        preferred.extend(["cnn_features_1d", "cnn_features_1d_sa", "WideResNet", "WideResNet_sa"])

    if outlier_frac > 0.02 or profile.get("unknown_count", 0) > 0:
        preferred.extend(["cnn_openmax", "WideResNet_edited"])
    else:
        preferred.append("cnn_openmax")

    if channels and channels > 3:
        preferred.insert(0, "WideResNet")

    return _unique_preserve(preferred + ARCHITECTURE_ORDER)


def _log_profile(prefix, profile):
    if not profile or profile.get("total_samples", 0) == 0:
        print(f"ℹ️ {prefix}: no samples available for profiling.")
        return
    msg = (
        f"{prefix}: {profile['total_samples']} samples · seq_len={profile.get('sequence_length', 'n/a')} · "
        f"channels={profile.get('channels', 'n/a')} · known={(profile.get('known_fraction', 0.0) * 100):.1f}% · "
        f"outliers={(profile.get('outlier_fraction', 0.0) * 100):.1f}%"
    )
    print(f"📊 {msg}")


# --- Chemistry-aware cathode grouping and compatibility ---
    
def build_cathode_groups(csv_path):
    """
    Group cathodes by approximate chemistry family and return a compatibility map.
    Families are intentionally tight to avoid harmful transfers.
    """

    df = pd.read_csv(csv_path)
    if "cathode" not in df.columns:
        raise ValueError(f"'cathode' column not found in {csv_path}")

    # Trim whitespace once so the regex patterns below see canonical names
    # (e.g. "NMC532" instead of " NMC532").
    df["cathode"] = df["cathode"].astype(str).str.strip()
    cath = df["cathode"]

    groups = {
        # NMC family (+ HE5050 bucketed here as NMC-like)
        "nmc": sorted(
            cath[cath.str.contains(r"^(?:NMC|HE5050)", case=False, regex=True)]
            .unique().tolist()
        ),
        # Li-rich layered compositions
        "lirich": sorted(
            cath[cath.str.contains(r"^Li1\.", case=False, regex=True)]
            .unique().tolist()
        ),
        # 5V spinel
        "spinel5v": sorted(
            cath[cath.str.contains("5Vspinel", case=False, regex=True)]
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
    scoped_cycles = shared_stats.get("cycles_per_cell_per_cathode")
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
        "cycles_per_cell_per_cathode": scoped_cycles if scoped_cycles is not None else {},
    }

def _cwru_shared_stats(src_train, src_val, tgt_train, tgt_val):
    def _len(ds):
        try:
            return len(ds)
        except Exception:
            return 0

    return {
        "source_train_sequences": _len(src_train),
        "source_val_sequences": _len(src_val),
        "target_train_sequences": _len(tgt_train),
        "target_val_sequences": _len(tgt_val),
    }


def _cwru_baseline_stats(shared_stats: dict[str, int]):
    if not shared_stats:
        return {}
    return {
        "source_train_sequences": shared_stats.get("target_train_sequences", 0),
        "source_val_sequences": shared_stats.get("target_val_sequences", 0),
        "target_train_sequences": 0,
        "target_val_sequences": shared_stats.get("target_val_sequences", 0),
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
        chem_ctx = cycle_stats.get("chemistry_context")
        if chem_ctx:
            summary["chemistry_context"] = chem_ctx

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


def _load_literature_context(path: str | None, limit: int = 1200) -> str:
    """Best-effort read of a local literature note to enrich the LLM prompt."""

    if not path:
        return ""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read().strip()
                if len(text) > limit:
                    text = text[:limit] + "…"
                return text
    except Exception:
        return ""
    return ""


def _build_text_context(args, num_summary):
    import json

    lines = []
    dataset_name = getattr(args, 'data_name', 'unknown')

    if dataset_name == 'Battery_inconsistent':
        feature_names = num_summary.get('feature_names', [])
        lines.append("Dataset: Argonne National Laboratory battery aging time-series with partial cycle windows.")
        chem_ctx = num_summary.get('chemistry_context', {})
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
        if chem_ctx:
            src_chem = chem_ctx.get('source') or {}
            tgt_chem = chem_ctx.get('target') or {}
            def _fmt_chem(bucket):
                parts = []
                for name, info in bucket.items():
                    an = info.get('anode') or '?'
                    cat = info.get('cathode') or name
                    ele = info.get('electrolyte') or 'electrolyte ?'
                    parts.append(f"{cat} (anode {an}, {ele})")
                return '; '.join(parts) if parts else '(unspecified chemistry)'
            lines.append(f"Source chemistry: {_fmt_chem(src_chem)}")
            lines.append(f"Target chemistry: {_fmt_chem(tgt_chem)}")
        lit_ctx = _load_literature_context(getattr(args, 'literature_context_file', None))
        if lit_ctx:
            lines.append("Literature cues (Joule S2542-4351(22)00409-3, chemistry-sensitive ablations):")
            lines.append(lit_ctx)
        if num_summary.get('cycle_limit_hint'):
            lines.append(f"Ablation: only first {num_summary['cycle_limit_hint']} cycles exposed to the model.")
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

def _parse_cycle_limits(raw: str, num_summary: dict, cycles_per_file: int) -> list[int]:
    """Parse user-provided cycle limits or fall back to a rich default battery sweep."""

    cycle_limits: list[int] = []
    for token in (raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = int(token)
            if val > 0:
                cycle_limits.append(val)
        except Exception:
            continue
        
    def _unique_in_order(values: list[int]) -> list[int]:
        seen: set[int] = set()
        ordered: list[int] = []
        for value in values:
            if value <= 0 or value in seen:
                continue
            ordered.append(value)
            seen.add(value)
        return ordered

    if cycle_limits:
        return _unique_in_order(cycle_limits)

    cycle_hint = (num_summary.get("cycle_stats") or {}).get("target_train_cycles") or 0
    defaults = [5, 15, 30, 50, 100, int(cycles_per_file)]
    if cycle_hint:
        defaults.append(int(min(cycle_hint, max(defaults))))
    return _unique_in_order(defaults)


def _resolve_compare_cycles(args, num_summary: Optional[dict] = None) -> int:
    """Pick a fixed cycle budget for non-cycle comparisons in llm_compare runs."""

    explicit = getattr(args, "compare_cycles", None)
    if explicit is not None and explicit > 0:
        reference = int(explicit)
    else:
        reference = 30

    cycle_hint = None
    if num_summary:
        cycle_hint = (num_summary.get("cycle_stats") or {}).get("target_train_cycles")
    if not cycle_hint:
        cycle_hint = (getattr(args, "dataset_cycle_stats", {}) or {}).get("target_train_cycles")
    if cycle_hint:
        reference = min(reference, int(cycle_hint))
    return max(1, int(reference))


def _apply_cycle_budget(target_args, cycles: int) -> None:
    target_args.cycles_per_file = int(cycles)
    target_args.source_cycles_per_file = int(cycles)
    target_args.target_cycles_per_file = int(cycles)

def _llm_pick_for_transfer(
    args,
    dls_for_peek: dict,
    debug_root: Optional[str] = None,
    *,
    allow_history: Optional[bool] = None,
    chemistry_feedback: Optional[bool] = None,
    compare_chemistry: Optional[bool] = None,
):
    """Build a per-transfer prompt and fetch configs + ablations."""

    stamp = getattr(args, "llm_cfg_stamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    setattr(args, "llm_cfg_stamp", stamp)
    debug_dir = None
    if debug_root:
        os.makedirs(debug_root, exist_ok=True)
        debug_dir = os.path.join(debug_root, f"transfer_{stamp}_{len(os.listdir(debug_root))}")
        os.makedirs(debug_dir, exist_ok=True)

    num_summary = _build_numeric_summary(dls_for_peek, args)
    text_ctx = _build_text_context(args, num_summary)
    
    if allow_history is None:
        allow_history = getattr(args, "llm_allow_history", True)
    if chemistry_feedback is None:
        chemistry_feedback = getattr(args, "llm_chemistry_feedback", True)
    if compare_chemistry is None:
        compare_chemistry = chemistry_feedback

    llm_cfg = select_config(
        text_context=text_ctx,
        num_summary=num_summary,
        backend=args.llm_backend,
        model=args.llm_model,
        debug_dir=debug_dir,
        allow_history=allow_history,
        chemistry_feedback=chemistry_feedback,
    )

    cycle_limits = _parse_cycle_limits(
        getattr(args, "ablation_cycle_limits", ""),
        num_summary,
        getattr(args, "cycles_per_file", 0) or 0,
    )
    ablation_records: list[dict] = []
    if args.llm_ablation:
        ablation_records = run_ablation_suite(
            text_ctx,
            num_summary,
            backend=args.llm_backend,
            model=args.llm_model,
            debug_dir=debug_dir,
            cycle_horizons=cycle_limits,
            base_config=llm_cfg,
            lock_hyperparams=True,
            chemistry_feedback=chemistry_feedback,
            compare_chemistry=compare_chemistry,
        )

    return llm_cfg, num_summary, text_ctx, ablation_records

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
            source_cycles_per_file=args.source_cycles_per_file,
            target_cycles_per_file=args.target_cycles_per_file,
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
            source_train_dataset, source_val_dataset, _, target_val_dataset, num_classes = cwru_dataset.data_split(transfer_learning=True)
            target_train_dataset = None
            args.num_classes = num_classes
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

    profile_for_method = _dataset_profile_from_loader(target_val_loader, args.num_classes or 0, getattr(args, 'sequence_length', None))
    chosen_method = _decide_uncertainty_method(
        getattr(args, 'method', 'deterministic'), profile_for_method, dataset_name=args.data_name
    )
    if chosen_method != getattr(args, 'method', None):
        print(f"⚖️  Auto-selected method: {chosen_method} (requested={getattr(args, 'method', None)})")
        args.method = chosen_method


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


def _safe_accuracy(labels, preds):
    if labels is None or preds is None:
        return 0.0
    labels_arr = np.asarray(labels)
    preds_arr = np.asarray(preds)
    if labels_arr.size == 0:
        return 0.0
    return float(accuracy_score(labels_arr, preds_arr))


def _load_sngp_uncertainty_mean(save_dir: str):
    summary_path = os.path.join(save_dir, "sngp_uncertainty_target_val_summary.json")
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r") as fh:
            summary = json.load(fh)
        return float(summary.get("mean_entropy"))
    except Exception as err:
        print(f"⚠️  Unable to read SNGP uncertainty summary at {summary_path}: {err}")
        return None

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

    groups, compat = build_cathode_groups(args.csv)
    
    print("🔍 Cathode families discovered:")
    for fam, items in groups.items():
        preview = ', '.join(items[:5])
        suffix = '…' if len(items) > 5 else ''
        print(f"   - {fam}: {len(items)} cathodes → {preview}{suffix}")
        
    print("🔁 Allowed transfer directions (data-driven):")
    for fam, targets in sorted(compat.items()):
        arrow = ', '.join(sorted(targets)) if targets else '(none)'
        print(f"   - {fam} → {arrow}")


    experiment_configs = []
    for src_name, src_cathodes in groups.items():
        for tgt_name, tgt_cathodes in groups.items():
            if src_name == tgt_name:
                continue
            if tgt_name not in compat.get(src_name, set()):
                print(f"⏭️  Skipping {src_name} → {tgt_name} (incompatible families: {src_name} → {tgt_name})")
                continue
            experiment_configs.append((src_name, tgt_name, src_cathodes, tgt_cathodes))

    if args.source_cathode and args.target_cathode:
        experiment_configs = [("custom", "custom", args.source_cathode, args.target_cathode)]
    forced_architectures = [args.model_name] if args.model_name else None
    original_method = args.method

    results = []
    cycle_details = []
    llm_transfer_records = []
    for src_name, tgt_name, source_cathodes, target_cathodes in experiment_configs:
        shared_tuple = load_battery_dataset(
            csv_path=args.csv,
            source_cathodes=source_cathodes,
            target_cathodes=target_cathodes,
            classification_label=args.classification_label,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_classes=args.num_classes,
            cycles_per_file=args.cycles_per_file,
            source_cycles_per_file=args.source_cycles_per_file,
            target_cycles_per_file=args.target_cycles_per_file,
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
        
        if args.llm_per_transfer:
            dls_for_peek = {
                'source_train': shared_src_train_loader,
                'source_val': shared_src_val_loader,
                'target_train': shared_tgt_train_loader,
                'target_val': shared_tgt_val_loader,
            }
            args.source_cathode = source_cathodes
            args.target_cathode = target_cathodes
            llm_cfg, num_summary, text_ctx, ablations = _llm_pick_for_transfer(
                args,
                dls_for_peek,
                debug_root=os.path.join(
                    args.checkpoint_dir,
                    f"llm_run_{getattr(args, 'llm_cfg_stamp', datetime.now().strftime('%Y%m%d_%H%M%S'))}",
                ),
            )
            llm_transfer_records.append({
                "dataset": args.data_name,
                "source_cathodes": list(source_cathodes),
                "target_cathodes": list(target_cathodes),
                "llm_cfg": llm_cfg,
                "numeric_summary": num_summary,
                "text_context": text_ctx,
                "ablation": ablations,
            })
            args.llm_cfg_inputs = {"text_context": text_ctx, "numeric_summary": num_summary}
            args.llm_cfg = llm_cfg
            forced_architectures = [llm_cfg["model_name"]]
            original_method = 'sngp' if llm_cfg.get("sngp", False) else 'deterministic'
            args.droprate = llm_cfg.get("dropout", getattr(args, "droprate", 0.3))
            args.lr = llm_cfg.get("learning_rate", getattr(args, "lr", 1e-3))
            args.batch_size = int(llm_cfg.get("batch_size", getattr(args, "batch_size", 64)))
            args.lambda_src = float(llm_cfg.get("lambda_src", getattr(args, "lambda_src", 1.0)))
            if hasattr(args, "bottleneck_num"):
                args.bottleneck_num = int(llm_cfg.get("bottleneck", getattr(args, "bottleneck_num", 256)))


        target_profile = _dataset_profile_from_loader(shared_tgt_val_loader, len(label_names), args.sequence_length)
        _log_profile(f"Battery target {src_name}→{tgt_name}", target_profile)

        arch_sequence = forced_architectures or _choose_architectures(target_profile)
        print(f"🧠 Architecture order for {src_name}→{tgt_name}: {', '.join(arch_sequence)}")

        method_choice = _decide_uncertainty_method(
            original_method, target_profile, dataset_name=args.data_name
        )
        if method_choice != original_method:
            print(f"⚖️  Selected method '{method_choice}' for {src_name}→{tgt_name} (requested={original_method}).")
        else:
            print(f"⚖️  Using method '{method_choice}' for {src_name}→{tgt_name}.")

        for model_name in arch_sequence:
            global_habbas3.init()
            args.model_name = model_name
            args.method = method_choice
            args.source_cathode = source_cathodes
            
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
                "target_val_loader": _clone_loader(shared_tgt_val_loader, force_shuffle=False),
                "label_names": label_names,
                "num_classes": len(label_names),
                "cycle_stats": _baseline_cycle_stats(shared_stats),
            }

            baseline_args = argparse.Namespace(**vars(args))
            baseline_args.source_cathode = target_cathodes
            baseline_args.target_cathode = []
            baseline_args.pretrained = False
            baseline_args.model_name = "cnn_features_1d"
            baseline_args.method = "deterministic"
            baseline_args.droprate = 0.3
            baseline_args.lr = getattr(args, "baseline_lr", 3e-4)
            baseline_args.lambda_src = 0.0
            baseline_args.sngp = False
            baseline_args.openmax = False
            baseline_args.use_unknown_head = False
            baseline_args.pretrained_model_path = None
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
                print(f"🧵 Zhao CNN baseline training cycles: {baseline_stats.get('source_train_cycles', 'n/a')}")

            transfer_args = argparse.Namespace(**vars(args))
            transfer_args.pretrained = True
            transfer_args.pretrained_model_path = os.path.join(pre_dir, "best_model.pth")
            transfer_args.source_cathode = source_cathodes
            transfer_args.target_cathode = target_cathodes
            
            # Cross-family transfers (e.g., nmc → spinel5v) have struggled; bias toward
            # quicker target adaptation by tempering source supervision and extending
            # the warmup/epoch budget.
            cross_family = src_name != tgt_name
            hard_spinel = cross_family and "nmc" in src_name.lower() and "spinel" in tgt_name.lower()
            if cross_family:
                transfer_args.lambda_src = min(getattr(transfer_args, "lambda_src", 1.0), 0.6)
                transfer_args.lambda_src_decay_patience = max(
                    1, getattr(transfer_args, "lambda_src_decay_patience", 5) // 2
                )
                transfer_args.lambda_src_warmup = max(
                    getattr(transfer_args, "lambda_src_warmup", 0), getattr(transfer_args, "warmup_epochs", 3)
                )
                transfer_args.max_epoch = int(max(getattr(transfer_args, "max_epoch", 50), 60) * 1.1)
                transfer_args.warmup_epochs = max(getattr(transfer_args, "warmup_epochs", 3), 5 if hard_spinel else 4)
                transfer_args.droprate = max(getattr(transfer_args, "droprate", 0.3), 0.35 if hard_spinel else 0.3)
                transfer_args.lr = min(getattr(transfer_args, "lr", args.lr), 5e-4)
                
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
                print(f"🔁 Transfer target training cycles used: {transfer_stats.get('target_train_cycles', 'n/a')}")
                
            cycle_details.append(
                {
                    "source": src_name,
                    "target": tgt_name,
                    "cycles": transfer_stats.get("cycles_per_cell_per_cathode", {}),
                }
            )
                
            eval_loader = transfer_override.get("target_val_loader")
            tr_labels, tr_preds = evaluate_model(model_ft, eval_loader)
            
            bl_labels, bl_preds = evaluate_model(model_bl, eval_loader)
            baseline_labels_np = np.array(bl_labels)
            baseline_preds_np = np.array(bl_preds)
            
            num_known = transfer_args.num_classes
            
            t_common, t_out, t_h = compute_common_outlier_metrics(tr_labels, tr_preds, num_known)
            b_common, b_out, b_h = compute_common_outlier_metrics(baseline_labels_np, baseline_preds_np, num_known)
            transfer_acc_metric = _safe_accuracy(tr_labels, tr_preds)
            baseline_acc_metric = _safe_accuracy(baseline_labels_np, baseline_preds_np)

            metric_key = args.improvement_metric
            
            metric_lookup = {
                "accuracy": (transfer_acc_metric, baseline_acc_metric),
                "common": (t_common, b_common),
                "hscore": (t_h, b_h),
                "overall": ((t_common + t_out) / 2.0, (b_common + b_out) / 2.0),
            }
           
            transfer_score, baseline_score = metric_lookup[metric_key]
            
            
            improvement = transfer_score - baseline_score
            print(
                f"📊 {src_name} → {tgt_name}: Zhao-baseline({metric_key})={baseline_score:.4f}, "
                f"transfer({metric_key})={transfer_score:.4f}, improvement={improvement:+.4f}"
            )
            
            final_model_label = 'transfer'
            selection_note = ''
            
            if transfer_score <= baseline_score:
                print(
                    f"⚠️ Transfer below Zhao CNN baseline for {src_name} → {tgt_name}; "
                    "no retries or booster passes run (fair single-shot evaluation)."
                )
                
                    
            if transfer_score <= baseline_score:
                final_model_label = 'baseline'
                selection_note = 'baseline_kept'
                print(
                    f"↩️  Using Zhao CNN baseline weights for {src_name} → {tgt_name}; improvement recorded as 0."
                )
                tr_labels, tr_preds = baseline_labels_np, baseline_preds_np
                t_common, t_out, t_h = b_common, b_out, b_h
                transfer_acc_metric = baseline_acc_metric
                transfer_score = baseline_score
                improvement = 0.0
                transfer_stats = baseline_stats or transfer_stats
                model_ft = model_bl
                baseline_ckpt = os.path.join(baseline_dir, "best_model.pth")
                transfer_ckpt = os.path.join(ft_dir, "best_model.pth")
                try:
                    shutil.copy2(baseline_ckpt, transfer_ckpt)
                    print(f"↩️  Copied baseline checkpoint to {transfer_ckpt}")
                except Exception as copy_err:
                    print(f"⚠️ Unable to copy baseline checkpoint: {copy_err}")

            print(
                f"🎯 Final selection for {src_name} → {tgt_name}: {final_model_label} (score={transfer_score:.4f})."
            )
            
            transfer_uncertainty = None
            if getattr(transfer_args, "method", "") == "sngp":
                transfer_uncertainty = _load_sngp_uncertainty_mean(ft_dir)
                if transfer_uncertainty is not None:
                    print(f"SNGP mean entropy (target_val): {transfer_uncertainty:.4f}")

            cm_transfer, labels_tr = _cm_with_min_labels(tr_labels, tr_preds, min_labels=5)
            cm_baseline, labels_bl = _cm_with_min_labels(baseline_labels_np, baseline_preds_np, min_labels=5)
            labels = sorted(set(labels_tr) | set(labels_bl))
            desired = max(len(labels), 5)
            cm_transfer, labels = _cm_with_min_labels(tr_labels, tr_preds, min_labels=desired)
            cm_baseline, _ = _cm_with_min_labels(baseline_labels_np, baseline_preds_np, min_labels=desired)
            transfer_acc_cm = float(np.trace(cm_transfer) / cm_transfer.sum()) if cm_transfer.sum() else 0.0
            baseline_acc_cm = float(np.trace(cm_baseline) / cm_baseline.sum()) if cm_baseline.sum() else 0.0

            transfer_acc = transfer_acc_metric
            baseline_acc = baseline_acc_metric

            if abs(transfer_acc_cm - transfer_acc_metric) > 1e-3:
                print(
                    f"⚠️  Transfer accuracy mismatch: cm_acc={transfer_acc_cm:.4f} vs metric={transfer_acc_metric:.4f}. Using metric-based accuracy."
                )
            if abs(baseline_acc_cm - baseline_acc_metric) > 1e-3:
                print(
                    f"⚠️  Baseline accuracy mismatch: cm_acc={baseline_acc_cm:.4f} vs metric={baseline_acc_metric:.4f}."
                )

            if not args.no_confmat:
                confmat_dir = args.confmat_dir or ft_dir
                transfer_cm_path = os.path.join(
                    confmat_dir, f"cm_transfer_{src_name}_to_{tgt_name}.png"
                )
                baseline_cm_path = os.path.join(confmat_dir, f"cm_baseline_{tgt_name}.png")
                _save_highres_confusion(
                    cm_transfer,
                    labels,
                    transfer_cm_path,
                    title=f"Transfer {src_name}→{tgt_name} | acc={transfer_acc*100:.2f}%",
                )
                _save_highres_confusion(
                    cm_baseline,
                    labels,
                    baseline_cm_path,
                    title=f"Baseline {tgt_name} | acc={baseline_acc*100:.2f}%",
                )
            
            results.append({
                "model": model_name,
                "source": src_name,
                "target": tgt_name,
                "comparison_metric": metric_key,
                "baseline_score": baseline_score,
                "transfer_score": transfer_score,
                "improvement": improvement,
                "baseline_accuracy": baseline_acc,
                "transfer_accuracy": transfer_acc,
                "baseline_common_acc": b_common,
                "transfer_common_acc": t_common,
                "baseline_outlier_acc": b_out,
                "transfer_outlier_acc": t_out,
                "baseline_hscore": b_h,
                "transfer_hscore": t_h,
                "transfer_uncertainty_mean_entropy": transfer_uncertainty,
                "final_model": final_model_label,
                "note": selection_note,
            })
            
            
            del model_ft
            del model_bl
            del eval_loader
            del transfer_override
            del baseline_override
            del transfer_args
            del baseline_args
            _cleanup_memory(f"battery {src_name}->{tgt_name} {model_name}")
            
            args.method = original_method
        
        del shared_src_train_loader
        del shared_src_val_loader
        del shared_tgt_train_loader
        del shared_tgt_val_loader
        _cleanup_memory(f"battery loaders {src_name}->{tgt_name}")

    if results:
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(
            args.checkpoint_dir,
            f"summary_{datetime.now().strftime('%m%d_%H%M%S')}_{args.data_name}.csv",
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
            "final_model",
        ]
        print(summary_df[cols_to_show])
        print("Baseline metrics reflect Zhao et al.'s CNN without transfer learning.")
        mean_impr = summary_df["improvement"].mean()
        overall = summary_df["transfer_score"].mean() - summary_df["baseline_score"].mean()
        print(f"Average improvement across experiments: {mean_impr:+.4f}")
        print(f"Overall transfer vs baseline: {overall:+.4f}")
        
        if llm_transfer_records:
            stamp = getattr(args, "llm_cfg_stamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            llm_path = os.path.join(
                args.checkpoint_dir,
                f"llm_transfer_choices_{args.data_name}_{stamp}.json",
            )
            with open(llm_path, "w") as fh:
                json.dump(llm_transfer_records, fh, indent=2)
            print(f"🧾 Saved per-transfer LLM picks to {llm_path}")
        
        cycle_summary: dict[str, dict[str, dict[str, int | dict[str, int]]]] = {}
        for entry in cycle_details:
            scoped = entry.get("cycles") or {}
            for scope, per_cathode in scoped.items():
                if not per_cathode:
                    continue
                scope_bucket = cycle_summary.setdefault(scope, {})
                for cathode, cell_list in per_cathode.items():
                    if not cell_list:
                        continue
                    cathode_bucket = scope_bucket.setdefault(cathode, {})
                    for cell_info in cell_list:
                        name = cell_info.get("filename")
                        cycles = cell_info.get("cycles")
                        total_cycles = cell_info.get("total_cycles")
                        if not name or cycles is None:
                            continue
                        existing = cathode_bucket.get(name)
                        if isinstance(existing, dict):
                            entry_dict = existing
                        elif existing is not None:
                            entry_dict = {"cycles": int(existing)}
                        else:
                            entry_dict = {}
                        entry_dict["cycles"] = int(cycles)
                        if total_cycles is not None:
                            entry_dict["total_cycles"] = int(total_cycles)
                        cathode_bucket[name] = entry_dict

        if cycle_summary:
            print("🧮 Cycles per cell per cathode used (by split):")
            for scope in sorted(cycle_summary):
                print(f"   [{scope}]")
                for cathode in sorted(cycle_summary[scope]):
                    cells = dict(sorted(cycle_summary[scope][cathode].items()))
                    print(f"      - {cathode} ({len(cells)} cells)")
                    for cell_name, info in cells.items():
                        if isinstance(info, dict):
                            used = info.get("cycles")
                            total = info.get("total_cycles")
                        else:
                            used = info
                            total = None
                        if used is None:
                            continue
                        if total is not None and total != used:
                            print(
                                f"           {cell_name}: {used} used / {total} total cycles"
                            )
                        elif total is not None:
                            print(f"           {cell_name}: {used} cycles (full coverage)")
                        else:
                            print(f"           {cell_name}: {used} cycles")
        
        
def run_cwru_experiments(args):
    raw_task = args.transfer_task

    if isinstance(raw_task, str):
        try:
            normalized = json.loads(raw_task)
        except Exception:
            normalized = eval(raw_task)
    else:
        normalized = raw_task

    if isinstance(normalized, list) and normalized:
        first = normalized[0]
        is_pair = isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple))
        transfer_tasks = normalized if is_pair else [normalized]
    else:
        transfer_tasks = [normalized]

    forced_architectures = [args.model_name] if args.model_name else None
    original_method = args.method
    results = []
    llm_transfer_records = []
    for transfer_task in transfer_tasks:
        if not transfer_task:
            continue

        src_ids = transfer_task[0]
        tgt_ids = transfer_task[1] if len(transfer_task) > 1 else []
        src_str = '-'.join(map(str, src_ids))
        tgt_str = '-'.join(map(str, tgt_ids))

        cwru_dataset = CWRU_inconsistent(
            args.data_dir, transfer_task, args.inconsistent, args.normlizetype
        )
        (
            shared_src_train_dataset,
            shared_src_val_dataset,
            shared_tgt_train_dataset,
            shared_tgt_val_dataset,
            num_classes,
        ) = cwru_dataset.data_split(transfer_learning=True)

        shared_src_train_loader = DataLoader(
            shared_src_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        shared_src_val_loader = DataLoader(
            shared_src_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        shared_tgt_train_loader = DataLoader(
            shared_tgt_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        shared_tgt_val_loader = DataLoader(
            shared_tgt_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        shared_stats = _cwru_shared_stats(
            shared_src_train_dataset,
            shared_src_val_dataset,
            shared_tgt_train_dataset,
            shared_tgt_val_dataset,
        )
        
        if args.llm_per_transfer:
            dls_for_peek = {
                'source_train': shared_src_train_loader,
                'source_val': shared_src_val_loader,
                'target_train': shared_tgt_train_loader,
                'target_val': shared_tgt_val_loader,
            }
            args.transfer_task = transfer_task
            llm_cfg, num_summary, text_ctx, ablations = _llm_pick_for_transfer(
                args,
                dls_for_peek,
                debug_root=os.path.join(
                    args.checkpoint_dir,
                    f"llm_run_{getattr(args, 'llm_cfg_stamp', datetime.now().strftime('%Y%m%d_%H%M%S'))}",
                ),
            )
            llm_transfer_records.append({
                "dataset": args.data_name,
                "source_ids": list(src_ids),
                "target_ids": list(tgt_ids),
                "llm_cfg": llm_cfg,
                "numeric_summary": num_summary,
                "text_context": text_ctx,
                "ablation": ablations,
            })
            args.llm_cfg_inputs = {"text_context": text_ctx, "numeric_summary": num_summary}
            args.llm_cfg = llm_cfg
            forced_architectures = [llm_cfg["model_name"]]
            original_method = 'sngp' if llm_cfg.get("sngp", False) else 'deterministic'
            args.droprate = llm_cfg.get("dropout", getattr(args, "droprate", 0.3))
            args.lr = llm_cfg.get("learning_rate", getattr(args, "lr", 1e-3))
            args.batch_size = int(llm_cfg.get("batch_size", getattr(args, "batch_size", 64)))
            args.lambda_src = float(llm_cfg.get("lambda_src", getattr(args, "lambda_src", 1.0)))
            if hasattr(args, "bottleneck_num"):
                args.bottleneck_num = int(llm_cfg.get("bottleneck", getattr(args, "bottleneck_num", 256)))


        label_names = list(range(num_classes))
        target_profile = _dataset_profile_from_loader(
            shared_tgt_val_loader,
            num_classes,
            getattr(shared_tgt_val_dataset, 'sequence_length', None),
        )
        _log_profile(f"CWRU target {src_str}→{tgt_str}", target_profile)

        arch_sequence = forced_architectures or _choose_architectures(target_profile)
        print(f"🧠 Architecture order for {src_str}→{tgt_str}: {', '.join(arch_sequence)}")

        method_choice = _decide_uncertainty_method(
            original_method, target_profile, dataset_name=args.data_name
        )
        if method_choice != original_method:
            print(f"⚖️  Selected method '{method_choice}' for {src_str}→{tgt_str} (requested={original_method}).")
        else:
            print(f"⚖️  Using method '{method_choice}' for {src_str}→{tgt_str}.")

        for model_name in arch_sequence:
            global_habbas3.init()
            args.model_name = model_name
            args.method = method_choice
            args.transfer_task = transfer_task
            args.num_classes = num_classes


            pre_args = argparse.Namespace(**vars(args))
            pre_args.pretrained = False
            pre_dir = os.path.join(
                args.checkpoint_dir,
                f"pretrain_{model_name}_{src_str}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(pre_dir, exist_ok=True)
            pre_override = {
                "source_train_loader": _clone_loader(shared_src_train_loader, force_shuffle=True),
                "source_val_loader": _clone_loader(shared_src_val_loader, force_shuffle=False),
                "target_train_loader": None,
                "target_val_loader": None,
                "label_names": label_names,
                "num_classes": num_classes,
                "cycle_stats": shared_stats,
            }
            run_experiment(pre_args, pre_dir, override_data=pre_override)

            transfer_override = {
                "source_train_loader": _clone_loader(shared_src_train_loader, force_shuffle=True),
                "source_val_loader": _clone_loader(shared_src_val_loader, force_shuffle=False),
                "target_train_loader": _clone_loader(shared_tgt_train_loader, force_shuffle=True),
                "target_val_loader": _clone_loader(shared_tgt_val_loader, force_shuffle=False),
                "label_names": label_names,
                "num_classes": num_classes,
                "cycle_stats": shared_stats,
            }

            baseline_override = {
                "source_train_loader": _clone_loader(shared_tgt_train_loader, force_shuffle=True),
                "source_val_loader": _clone_loader(shared_tgt_val_loader, force_shuffle=False),
                "target_train_loader": None,
                "target_val_loader": _clone_loader(shared_tgt_val_loader, force_shuffle=False),
                "label_names": label_names,
                "num_classes": num_classes,
                "cycle_stats": _cwru_baseline_stats(shared_stats),
            }

            baseline_args = argparse.Namespace(**vars(args))
            baseline_args.pretrained = False
            baseline_args.transfer_task = [tgt_ids, tgt_ids]
            baseline_args.model_name = "cnn_features_1d"
            baseline_args.method = "deterministic"
            baseline_args.droprate = 0.3
            baseline_args.lr = getattr(args, "baseline_lr", 3e-4)
            baseline_args.lambda_src = 0.0
            baseline_args.sngp = False
            baseline_args.openmax = False
            baseline_args.use_unknown_head = False
            baseline_args.pretrained_model_path = None
            baseline_dir = os.path.join(
                args.checkpoint_dir,
                f"baseline_{model_name}_{tgt_str}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(baseline_dir, exist_ok=True)
            model_bl, _, _, _, _ = run_experiment(
                baseline_args,
                baseline_dir,
                baseline=True,
                override_data=baseline_override,
            )
            
            transfer_args = argparse.Namespace(**vars(args))
            transfer_args.pretrained = True
            transfer_args.transfer_task = transfer_task
            transfer_args.pretrained_model_path = os.path.join(pre_dir, "best_model.pth")

            ft_dir = os.path.join(
                args.checkpoint_dir,
                f"transfer_{model_name}_{src_str}_to_{tgt_str}_{datetime.now().strftime('%m%d')}",
            )
            os.makedirs(ft_dir, exist_ok=True)
            model_ft, _, _, _, _ = run_experiment(
                transfer_args,
                ft_dir,
                override_data=transfer_override,
            )

            eval_loader = transfer_override["target_val_loader"]
            tr_labels, tr_preds = evaluate_model(model_ft, eval_loader)
            bl_labels, bl_preds = evaluate_model(model_bl, eval_loader)

            baseline_labels_np = np.array(bl_labels)
            baseline_preds_np = np.array(bl_preds)
            
            num_known = transfer_args.num_classes
            
            t_common, t_out, t_h = compute_common_outlier_metrics(tr_labels, tr_preds, num_known)
            b_common, b_out, b_h = compute_common_outlier_metrics(baseline_labels_np, baseline_preds_np, num_known)
            
            transfer_acc_metric = _safe_accuracy(tr_labels, tr_preds)
            baseline_acc_metric = _safe_accuracy(baseline_labels_np, baseline_preds_np)
            
            metric_key = args.improvement_metric
            metric_lookup = {
                "accuracy": (transfer_acc_metric, baseline_acc_metric),
                "common": (t_common, b_common),
                "hscore": (t_h, b_h),
                "overall": ((t_common + t_out) / 2.0, (b_common + b_out) / 2.0),
            }
            transfer_score, baseline_score = metric_lookup[metric_key]
            improvement = transfer_score - baseline_score
            
            print(
                f"✅ Transfer {src_str} → {tgt_str}: common={t_common:.4f}, outlier={t_out:.4f}, hscore={t_h:.4f}"
            )
            print(
                f"🧪 Zhao CNN baseline (no transfer): common={b_common:.4f}, outlier={b_out:.4f}, hscore={b_h:.4f}"
            )
            
            print(
                f"📊 {src_str} → {tgt_str}: Zhao-baseline({metric_key})={baseline_score:.4f}, "
                f"transfer({metric_key})={transfer_score:.4f}, improvement={improvement:+.4f}"
            )
            
            final_model_label = 'transfer'
            selection_note = ''
            
            if transfer_score <= baseline_score:
                
                print(
                    f"⚠️ Transfer below Zhao CNN baseline for {src_str} → {tgt_str}; "
                    "no retries or booster passes run (fair single-shot evaluation)."
                )
                
            
            if transfer_score <= baseline_score:
                final_model_label = 'baseline'
                selection_note = 'baseline_kept'
                print(
                    f"↩️  Using Zhao CNN baseline weights for {src_str} → {tgt_str}; improvement recorded as 0."
                )
                tr_labels, tr_preds = baseline_labels_np, baseline_preds_np
                t_common, t_out, t_h = b_common, b_out, b_h
                transfer_acc_metric = baseline_acc_metric
                transfer_score = baseline_score
                improvement = 0.0
                model_ft = model_bl
                baseline_ckpt = os.path.join(baseline_dir, "best_model.pth")
                transfer_ckpt = os.path.join(ft_dir, "best_model.pth")
                try:
                    shutil.copy2(baseline_ckpt, transfer_ckpt)
                    print(f"↩️  Copied baseline checkpoint to {transfer_ckpt}")
                except Exception as copy_err:
                    print(f"⚠️ Unable to copy baseline checkpoint: {copy_err}")

            print(
                f"🎯 Final selection for {src_str} → {tgt_str}: {final_model_label} (score={transfer_score:.4f})."
            )
            
            transfer_uncertainty = None
            if getattr(transfer_args, "method", "") == "sngp":
                transfer_uncertainty = _load_sngp_uncertainty_mean(ft_dir)
                if transfer_uncertainty is not None:
                    print(f"SNGP mean entropy (target_val): {transfer_uncertainty:.4f}")

            cm_transfer, labels_tr = _cm_with_min_labels(tr_labels, tr_preds, min_labels=10)
            cm_baseline, labels_bl = _cm_with_min_labels(baseline_labels_np, baseline_preds_np, min_labels=10)
            labels = sorted(set(labels_tr) | set(labels_bl))
            cm_transfer, labels = _cm_with_min_labels(tr_labels, tr_preds, min_labels=len(labels))
            cm_baseline, _ = _cm_with_min_labels(baseline_labels_np, baseline_preds_np, min_labels=len(labels))
            transfer_acc_cm = float(np.trace(cm_transfer) / cm_transfer.sum()) if cm_transfer.sum() else 0.0
            baseline_acc_cm = float(np.trace(cm_baseline) / cm_baseline.sum()) if cm_baseline.sum() else 0.0

            transfer_acc = transfer_acc_metric
            baseline_acc = baseline_acc_metric

            if abs(transfer_acc_cm - transfer_acc_metric) > 1e-3:
                print(
                    f"⚠️  Transfer accuracy mismatch: cm_acc={transfer_acc_cm:.4f} vs metric={transfer_acc_metric:.4f}. Using metric-based accuracy."  # noqa: E501
                )
            if abs(baseline_acc_cm - baseline_acc_metric) > 1e-3:
                print(
                    f"⚠️  Baseline accuracy mismatch: cm_acc={baseline_acc_cm:.4f} vs metric={baseline_acc_metric:.4f}."
                )

            if not args.no_confmat:
                confmat_dir = args.confmat_dir or ft_dir
                transfer_cm_path = os.path.join(
                    confmat_dir, f"cm_transfer_{src_str}_to_{tgt_str}.png"
                )
                baseline_cm_path = os.path.join(confmat_dir, f"cm_baseline_{tgt_str}.png")
                _save_highres_confusion(
                    cm_transfer,
                    labels,
                    transfer_cm_path,
                    title=f"Transfer {src_str}→{tgt_str} | acc={transfer_acc*100:.2f}%",
                )
                _save_highres_confusion(
                    cm_baseline,
                    labels,
                    baseline_cm_path,
                    title=f"Baseline {tgt_str} | acc={baseline_acc*100:.2f}%",
                )

            
            results.append(
                {
                    "model": model_name,
                    "source": src_str,
                    "target": tgt_str,
                    "comparison_metric": metric_key,
                    "baseline_score": baseline_score,
                    "transfer_score": transfer_score,
                    "improvement": improvement,
                    "baseline_accuracy": baseline_acc,
                    "transfer_accuracy": transfer_acc,
                    "baseline_common_acc": b_common,
                    "transfer_common_acc": t_common,
                    "baseline_outlier_acc": b_out,
                    "transfer_outlier_acc": t_out,
                    "baseline_hscore": b_h,
                    "transfer_hscore": t_h,
                    "transfer_uncertainty_mean_entropy": transfer_uncertainty,
                    "final_model": final_model_label,
                    "note": selection_note,
                }
            )
            
            
            del model_ft
            del model_bl
            del eval_loader
            del transfer_override
            del baseline_override
            del transfer_args
            del baseline_args
            _cleanup_memory(f"cwru {src_str}->{tgt_str} {model_name}")
            
        args.method = original_method
        
        del shared_src_train_loader
        del shared_src_val_loader
        del shared_tgt_train_loader
        del shared_tgt_val_loader
        _cleanup_memory(f"cwru loaders {src_str}->{tgt_str}")

    if results:
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(
            args.checkpoint_dir,
            f"summary_{datetime.now().strftime('%m%d_%H%M%S')}_{args.data_name}.csv",
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
            "final_model",
        ]
        print(summary_df[cols_to_show])
        print("Baseline metrics reflect Zhao et al.'s CNN without transfer learning.")
        mean_impr = summary_df["improvement"].mean()
        overall = summary_df["transfer_score"].mean() - summary_df["baseline_score"].mean()
        print(f"Average improvement across experiments: {mean_impr:+.4f}")
        print(f"Overall transfer vs baseline: {overall:+.4f}")


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
            print(f"Failed to build cycle CSV: {exc}")

    
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
                    cycles_per_file=args.cycles_per_file,
                    source_cycles_per_file=args.source_cycles_per_file,
                    target_cycles_per_file=args.target_cycles_per_file,
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
        ablation_records = []
        cycle_ablation_cfgs = []
        coldstart_cfg = None
        chemistry_off_cfg = None
        load_off_cfg = None
        if args.llm_ablation:
            raw_limits = (getattr(args, "ablation_cycle_limits", "") or "")
            cycle_limits: list[int] = []
            for token in raw_limits.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    val = int(token)
                    if val > 0:
                        cycle_limits.append(val)
                except Exception:
                    continue

            if not cycle_limits:
                cycle_hint = (num_summary.get("cycle_stats") or {}).get("target_train_cycles") or 0
                defaults = {5, 15, 30, 50, 100, int(args.cycles_per_file)}
                if cycle_hint:
                    defaults.add(min(int(cycle_hint), max(defaults)))
                cycle_limits = sorted([v for v in defaults if v > 0])

            ablation_records = run_ablation_suite(
                text_ctx,
                num_summary,
                backend=args.llm_backend,
                model=args.llm_model,
                debug_dir=_llm_root,
                cycle_horizons=cycle_limits,
                base_config=llm_cfg,
                lock_hyperparams=True,
                chemistry_feedback=True,
                compare_chemistry=True,
                include_transfer_context=True,
                compare_transfer_context=True,
            )

            for record in ablation_records:
                if record.get("tag") == "history_off":
                    coldstart_cfg = record.get("config")
                if record.get("tag") == "chemistry_off":
                    chemistry_off_cfg = record.get("config")
                if record.get("tag") == "load_off":
                    load_off_cfg = record.get("config")
                if record.get("cycle_limit"):
                    cycle_ablation_cfgs.append(record)

            ablation_path = _os.path.join(_llm_root, "llm_ablation.json")
            with open(ablation_path, "w") as _f:
                _json.dump(ablation_records, _f, indent=2)
            print(f"🔎 Ablation prompts saved to {ablation_path}")

        

    if args.auto_select and args.llm_compare:
        import copy, glob, time, shutil, pandas as _pd, json as _json, os as _os

        base_args = copy.deepcopy(args)
        candidates = []
        
        reference_cycles = _resolve_compare_cycles(args, locals().get("num_summary"))

        def _configure_llm_prompting(target_args, tag: str) -> None:
            target_args.llm_per_transfer = False

        def _apply_non_cycle_budget(target_args) -> None:
            _apply_cycle_budget(target_args, reference_cycles)
        
        def _apply_llm_cfg_to_args(cfg_obj, target_args):
            target_args.model_name = cfg_obj.get("model_name", target_args.model_name)
            target_args.method = 'sngp' if cfg_obj.get("sngp", False) else 'deterministic'
            target_args.droprate = cfg_obj.get("dropout", getattr(target_args, "droprate", 0.3))
            target_args.lr = cfg_obj.get("learning_rate", getattr(target_args, "lr", 1e-3))
            target_args.batch_size = int(cfg_obj.get("batch_size", getattr(target_args, "batch_size", 64)))
            target_args.lambda_src = float(cfg_obj.get("lambda_src", getattr(target_args, "lambda_src", 1.0)))
            if hasattr(target_args, "bottleneck_num"):
                target_args.bottleneck_num = int(cfg_obj.get("bottleneck", getattr(target_args, "bottleneck_num", 256)))
            return target_args

        # 1) The LLM pick (already applied to args)
        candidates.append(("llm_pick", copy.deepcopy(args)))
        
        
        
        if coldstart_cfg:
            cold_args = _apply_llm_cfg_to_args(coldstart_cfg, copy.deepcopy(base_args))
            cold_args.tag = (getattr(cold_args, "tag", "") + "_history_off_" + args.llm_cfg_stamp).strip("_")
            candidates.append(("history_off", cold_args))
            
        if chemistry_off_cfg:
            chem_args = _apply_llm_cfg_to_args(chemistry_off_cfg, copy.deepcopy(base_args))
            chem_args.tag = (getattr(chem_args, "tag", "") + "_chemistry_off_" + args.llm_cfg_stamp).strip("_")
            candidates.append(("chemistry_off", chem_args))

        if load_off_cfg:
            load_args = _apply_llm_cfg_to_args(load_off_cfg, copy.deepcopy(base_args))
            load_args.tag = (getattr(load_args, "tag", "") + "_load_off_" + args.llm_cfg_stamp).strip("_")
            candidates.append(("load_off", load_args))

        wrn_det = copy.deepcopy(base_args)
        wrn_det.model_name = "WideResNet"
        wrn_det.method = "deterministic"
        wrn_det.droprate = max(getattr(base_args, "droprate", 0.3), 0.25)
        wrn_det.tag = (getattr(wrn_det, "tag", "") + "_wrn_base_" + args.llm_cfg_stamp).strip("_")
        candidates.append(("wideresnet", wrn_det))

        

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

        def _collect_latest_summary(copy_prefix: str) -> tuple[str, dict]:
            summaries = sorted(glob.glob(_os.path.join("checkpoint", "summary_*.csv")), key=os.path.getmtime)
            if not summaries:
                return ("", {})
            latest = summaries[-1]
            dst = os.path.join(_cmp_dir, f"{copy_prefix}_{os.path.basename(latest)}")
            try:
                shutil.copy2(latest, dst)
            except Exception:
                dst = latest
            stats: dict[str, float] = {}
            try:
                df = _pd.read_csv(dst)
                acc_stats = {}
                
                def _mean_from_cols(columns, mask=None):
                    for col in columns:
                        if col not in df.columns:
                            continue
                        series = df[col]
                        if mask is not None:
                            series = series[mask]
                        vals = _pd.to_numeric(series, errors="coerce").dropna()
                        if not vals.empty:
                            return float(vals.mean())
                    return None

                acc_mask = None
                if "comparison_metric" in df.columns:
                    acc_mask = df["comparison_metric"].astype(str).str.lower().isin({"accuracy", "common"})

                baseline_acc = _mean_from_cols(["baseline_accuracy", "baseline_acc"], mask=acc_mask)
                transfer_acc = _mean_from_cols(["transfer_accuracy", "transfer_acc"], mask=acc_mask)
                if baseline_acc is None:
                    baseline_acc = _mean_from_cols(["baseline_score"], mask=acc_mask)
                if transfer_acc is None:
                    transfer_acc = _mean_from_cols(["transfer_score"], mask=acc_mask)

                if baseline_acc is not None:
                    acc_stats["baseline_accuracy"] = baseline_acc
                if transfer_acc is not None:
                    acc_stats["transfer_accuracy"] = transfer_acc
                if baseline_acc is not None and transfer_acc is not None:
                    acc_stats["accuracy_delta"] = transfer_acc - baseline_acc

                if "transfer_uncertainty_mean_entropy" in df.columns:
                    entropy_vals = _pd.to_numeric(
                        df["transfer_uncertainty_mean_entropy"], errors="coerce"
                    ).dropna()
                    if not entropy_vals.empty:
                        acc_stats["transfer_uncertainty_mean_entropy"] = float(entropy_vals.mean())
                if "improvement" in df.columns:
                    imp = _pd.to_numeric(df["improvement"], errors="coerce").dropna()
                    if not imp.empty:
                        avg_imp = float(imp.mean())
                        stats["mean"] = avg_imp
                        stats["median"] = float(imp.median())
                        stats["std"] = float(imp.std(ddof=0))
                        stats["count"] = float(len(imp))
                        stats["positive"] = float((imp > 0).sum())
                        stderr = float(imp.std(ddof=0) / max(len(imp) ** 0.5, 1e-9))
                        stats["ci95"] = float(1.96 * stderr)
                elif {"transfer_acc","baseline_acc"}.issubset(df.columns):
                    diffs = (
                        _pd.to_numeric(df["transfer_acc"], errors="coerce")
                        - _pd.to_numeric(df["baseline_acc"], errors="coerce")
                    ).dropna()
                    if not diffs.empty:
                        stats["mean"] = float(diffs.mean())
                        stats["median"] = float(diffs.median())
                        stats["std"] = float(diffs.std(ddof=0))
                        stats["count"] = float(len(diffs))
                        stats["positive"] = float((diffs > 0).sum())
                        stderr = float(diffs.std(ddof=0) / max(len(diffs) ** 0.5, 1e-9))
                        stats["ci95"] = float(1.96 * stderr)
                elif {"transfer_score", "baseline_score"}.issubset(df.columns):
                    diffs = (
                        _pd.to_numeric(df["transfer_score"], errors="coerce")
                        - _pd.to_numeric(df["baseline_score"], errors="coerce")
                    ).dropna()
                    if not diffs.empty:
                        stats["mean"] = float(diffs.mean())
                        stats["median"] = float(diffs.median())
                        stats["std"] = float(diffs.std(ddof=0))
                        stats["count"] = float(len(diffs))
                        stats["positive"] = float((diffs > 0).sum())
                        stderr = float(diffs.std(ddof=0) / max(len(diffs) ** 0.5, 1e-9))
                        stats["ci95"] = float(1.96 * stderr)
            except Exception:
                stats = {}
                stats.update(acc_stats)
            return (dst, stats)
        for tag, cfg in candidates:
            print(f"\n===== LLM comparison run: {tag} =====")
            if tag.startswith("cycles_"):
                try:
                    cyc_val = int(tag.split("_", 1)[1])
                    _apply_cycle_budget(cfg, cyc_val)
                except Exception:
                    _apply_cycle_budget(cfg, reference_cycles)
            else:
                _apply_non_cycle_budget(cfg)

            _configure_llm_prompting(cfg, tag)
            if cfg.data_name == 'Battery_inconsistent':
                run_battery_experiments(cfg)
            else:
                run_cwru_experiments(cfg)

            time.sleep(0.5)

            copied_path, stats = _collect_latest_summary(copy_prefix=tag)
            avg_imp = stats.get("mean", float("nan")) if stats else float("nan")
            cyc_lim = None
            if tag.startswith("cycles_"):
                try:
                    cyc_lim = int(tag.split("_", 1)[1])
                except Exception:
                    cyc_lim = None
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
                "improvement_median": stats.get("median") if stats else None,
                "improvement_std": stats.get("std") if stats else None,
                "improvement_count": stats.get("count") if stats else None,
                "improvement_positive": stats.get("positive") if stats else None,
                "improvement_ci95": stats.get("ci95") if stats else None,
                "baseline_accuracy_mean": stats.get("baseline_accuracy") if stats else None,
                "transfer_accuracy_mean": stats.get("transfer_accuracy") if stats else None,
                "target_accuracy_mean": stats.get("transfer_accuracy") if stats else None,
                "accuracy_delta_mean": stats.get("accuracy_delta") if stats else None,
                "transfer_uncertainty_mean_entropy": stats.get("transfer_uncertainty_mean_entropy") if stats else None,
                "cycle_limit": cyc_lim,
            })

        _llm_root = _os.path.join("checkpoint", f"llm_run_{args.llm_cfg_stamp}")
        _leader_csv = _os.path.join(_llm_root, "llm_leaderboard.csv")
        _leader_json = _os.path.join(_llm_root, "llm_leaderboard.json")
        _pd.DataFrame(leaderboard_rows).to_csv(_leader_csv, index=False)
        with open(_leader_json, "w") as _f:
            _json.dump(leaderboard_rows, _f, indent=2)
            
        _manifest_path = _os.path.join(_llm_root, "llm_compare_manifest.json")
        tag_defs = {
            "llm_pick": "Single-shot LLM configuration (history + transfer context enabled).",
            "history_off": "LLM configuration without leaderboard/history context (cold-start prompt).",
            "chemistry_off": "LLM configuration without battery chemistry hints.",
            "load_off": "LLM configuration without CWRU load/HP/rpm transfer metadata.",
            "deterministic_cnn": "Zhao CNN baseline (deterministic, no transfer head).",
            "wideresnet": "Deterministic WideResNet capacity baseline (no SNGP).",
            "sngp_wrn_sa": "WideResNet+SA with SNGP head (calibrated baseline).",
        }
        manifest = {
            "baseline_definition": "Zhao et al. deterministic 1-D CNN trained on target-only (no transfer).",
            "single_shot_evaluation": True,
            "retries_disabled": True,
            "parameter_definitions": {
                "droprate": "Dropout probability applied to the classifier head.",
                "lr": "Learning rate for optimizer.",
                "batch_size": "Mini-batch size for training/transfer.",
                "lambda_src": "Weight on source-domain loss during transfer.",
                "method": "Training head style (deterministic or SNGP).",
            },
            "candidates": [
                {"tag": tag, "definition": tag_defs.get(tag, "")}
                for tag, _cfg in candidates
            ],
        }
        with open(_manifest_path, "w") as _f:
            _json.dump(manifest, _f, indent=2)


        _valid = [r for r in leaderboard_rows if not (r["avg_improvement"] != r["avg_improvement"])]
        if _valid:
            best = max(_valid, key=lambda r: r["avg_improvement"])
            print("\n🏆 Leaderboard (avg improvement over Zhao CNN baseline):")
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
