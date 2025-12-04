#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
from torch import optim
import my_datasets.global_habbas3
import models
# from models.cnn_1d_original import CNN
# from models.cnn_1d_habbas import CNN
# from models.cnn_1d_habbas_hyperparstudy import CNN as cnn_features_1d_hyperparstudy
from models.wideresnet_habbas import WideResNet
from models.wideresnet_self_attention_habbas import WideResNet_sa
from models.wideresnet_multihead_attention import WideResNet_mh
from models.wideresnet_edited_habbas import WideResNet_edited
# from models.cnn_1d_selfattention_habbas import cnn_features as cnn_features_1d_sa
from models.cnn_sa_openmax_habbas import CNN_OpenMax as cnn_openmax
import datasets
from utils.counter import AccuracyCounter
import torch.nn.functional as F
from utils.lib import *
from models.sngp import Deterministic as deterministic
from models.sngp import SNGP as sngp
from utils.sngp_utils import to_numpy, Accumulator, mean_field_logits
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score
import random
import torch
import pandas as pd
import numpy
import optuna
from my_datasets.Battery_label_inconsistent import load_battery_dataset
import global_habbas3
from itertools import cycle
import traceback
from sklearn.utils.class_weight import compute_class_weight
import torch
import copy
from datetime import datetime
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import json
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.sampler import RandomSampler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import RandomSampler





SEED = 123  # Choose your own seed

torch.manual_seed(SEED)
numpy.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(SEED)
#Adapted from https://github.com/YU1ut/openset-DA and https://github.com/thuml/Universal-Domain-Adaptation


def _find_last_linear(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last


def _kaiming_reset_linear(lin: nn.Linear):
    if hasattr(nn.init, "kaiming_uniform_"):
        nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
    else:
        nn.init.xavier_uniform_(lin.weight)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)
        
        
def _wrap_with_class_balanced_sampler(loader, batch_size, num_workers, device):
    """
    Replace a DataLoader with a class-balanced sampler so minority classes are not ignored.
    Works even if the dataset doesn't expose .targets; we scan labels once.
    """
    if loader is None:
        return loader
    ds = loader.dataset

    # Try to get labels efficiently; fall back to a one-pass scan
    labels = None
    for attr in ("targets", "labels", "y", "ys"):
        if hasattr(ds, attr):
            arr = getattr(ds, attr)
            labels = np.asarray(arr if isinstance(arr, (list, np.ndarray)) else list(arr))
            break
    if labels is None:
        # One pass through the dataset to read labels
        labels = np.array([ds[i][1] for i in range(len(ds))])

    classes, counts = np.unique(labels, return_counts=True)
    class_to_weight = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
    sample_weights = np.array([class_to_weight[int(y)] for y in labels], dtype=np.float32)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    # Rebuild the loader with the sampler (no shuffle)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(True if str(device) == 'cuda' else False),
        drop_last=False
    )



@torch.no_grad()
def calibrate_bn(model: nn.Module, loader, device, max_batches: int = 50):
    """Recompute BN running stats on target data; weights frozen."""
    if loader is None:
        return
    was_training = model.training
    model.train()  # BN updates running stats in train mode
    cnt = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        model(x)
        cnt += 1
        if cnt >= max_batches:
            break
    model.train(was_training)
    print(f"📏 BN calibrated on {cnt} target batches.")
    
@torch.no_grad()
def _save_confusion_outputs(model: nn.Module,
                            loader,
                            device,
                            num_classes: int,
                            out_dir: str,
                            split_name: str,
                            labels_override=None):
    """
    Saves confusion matrix (PNG + CSV) and classification report (JSON).
    Handles models that return (logits, features, ...) and maps any
    predicted/true labels outside [0..num_classes-1] safely.
    """
    import os
    if loader is None:
        return

    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        # logits -> preds
        preds = torch.argmax(out, dim=1).detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)

    if not all_labels:
        return

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # Keep only known-class samples, then clip preds to known-class range
    known_mask = y_true < num_classes
    y_true = y_true[known_mask]
    y_pred = y_pred[known_mask]
    y_pred = np.clip(y_pred, 0, num_classes - 1)

    # Label set for axes
    labels = list(labels_override) if labels_override is not None else list(range(num_classes))
    
    # Ensure Battery confusion matrices always expose the same five classes
    # (0-4) even if a particular split is missing later-life classes.  This
    # keeps figure layouts consistent across experiments.
    try:
        numeric_labels = [int(l) for l in labels]
    except Exception:
        numeric_labels = []
    if numeric_labels:
        max_label = max(numeric_labels)
        desired_max = max(max_label, 4)
        labels = list(range(desired_max + 1))
    elif labels_override is None and num_classes < 5:
        labels = list(range(5))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    os.makedirs(out_dir, exist_ok=True)
    # CSV
    np.savetxt(os.path.join(out_dir, f"confmat_{split_name}.csv"), cm, delimiter=",", fmt="%d")

    # PNG
    fig = plt.figure(figsize=(4 + 0.3*len(labels), 4 + 0.3*len(labels)), dpi=150)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(title=f"Confusion Matrix - {split_name}", xlabel="Predicted", ylabel="True")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    thresh = cm.max()/2 if cm.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = int(cm[i, j]) if cm.size else 0
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if val > thresh else "black")
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"confmat_{split_name}.png")
    fig.savefig(png_path)
    plt.close(fig)

    # Classification report
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    with open(os.path.join(out_dir, f"classification_report_{split_name}.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"🧾 Saved confusion matrix & report for {split_name} → {png_path}")




class train_utils_open_univ(object):
    def __init__(self, args, save_dir,
                 source_train_loader, source_val_loader,
                 target_train_loader=None, target_val_loader=None,
                 source_train_dataset=None, target_val_dataset=None):

        self.args = args
        self.save_dir = save_dir
        self.best_hscore = 0.0

        # ✅ Assign loaders
        self.source_train_loader = source_train_loader
        self.source_val_loader = source_val_loader
        self.target_train_loader = target_train_loader
        self.target_val_loader = target_val_loader

        # ✅ Assign datasets
        self.source_train_dataset = source_train_dataset
        self.target_val_dataset = target_val_dataset

        # ✅ Build dataloaders dictionary dynamically
        self.dataloaders = {
            'source_train': self.source_train_loader,
            'source_val': self.source_val_loader
        }

        if self.target_train_loader is not None:
            self.dataloaders['target_train'] = self.target_train_loader
        if self.target_val_loader is not None:
            self.dataloaders['target_val'] = self.target_val_loader
            
        
        # --- Adaptive source-loss schedule knobs ---
        self.lambda_src_init = float(getattr(self.args, 'lambda_src', 1.0))
        self.lambda_src_current = self.lambda_src_init
        self.lambda_src_decay_patience = int(getattr(self.args, 'lambda_src_decay_patience', 5))
        self.lambda_src_decay_factor = float(getattr(self.args, 'lambda_src_decay_factor', 0.5))
        self.lambda_src_min = float(getattr(self.args, 'lambda_src_min', 0.0))
        self.lambda_src_warmup = int(getattr(self.args, 'lambda_src_warmup', 0))
        self._lambda_plateau_epochs = 0
        

        
    def collect_activation_vectors(self):
        activation_vectors = []
        labels_vector = []
        
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.dataloaders['source_train']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if torch.isnan(inputs).any() or torch.isnan(labels).any():
                    print("NaN values detected in inputs or labels")
                    continue  # Skip this batch or handle NaNs appropriately
    
                # Extract features before OpenMax
                _, features = self.model.forward_before_openmax(inputs)
                activation_vectors.append(features.cpu().numpy())
                labels_vector.append(labels.cpu().numpy())
    
        # Convert lists to numpy arrays after collecting all data
        activation_vectors = np.concatenate(activation_vectors, axis=0)
        labels_vector = np.concatenate(labels_vector, axis=0)
        return activation_vectors, labels_vector
    
    def _export_sngp_uncertainty(self, loader, split_name: str):
        if loader is None or getattr(self.args, 'method', '') != 'sngp':
            return
        if not hasattr(self, 'sngp_model'):
            return

        self.model.eval()
        self.sngp_model.eval()
        records = []
        probs_batches = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    backbone_logits = outputs[0]
                    features = outputs[1] if len(outputs) > 1 else None
                else:
                    backbone_logits, features = outputs, None

                if features is not None:
                    feats = features
                    if self.args.bottleneck and hasattr(self, 'bottleneck_layer') and not isinstance(self.bottleneck_layer, nn.Identity):
                        feats = self.bottleneck_layer(feats)
                    gp_feature, gp_logits = self.sngp_model.gp_layer(feats, update_cov=False)
                    cov = self.sngp_model.compute_predictive_covariance(gp_feature)
                    logits = mean_field_logits(gp_logits, cov)
                else:
                    logits = backbone_logits

                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
                max_prob, preds = torch.max(probs, dim=1)

                for idx in range(labels.size(0)):
                    label_val = int(labels[idx].item())
                    records.append({
                        "split": split_name,
                        "index": len(records),
                        "label": label_val,
                        "pred": int(preds[idx].item()),
                        "max_prob": float(max_prob[idx].item()),
                        "entropy": float(entropy[idx].item()),
                        "is_outlier": bool(label_val >= self.num_classes),
                    })

                probs_batches.append(probs.detach().cpu().numpy())

        if not records:
            return

        df = pd.DataFrame.from_records(records)
        known_mask = ~df["is_outlier"]
        summary = {
            "mean_entropy": float(df["entropy"].mean()),
            "mean_entropy_known": float(df.loc[known_mask, "entropy"].mean()) if known_mask.any() else float('nan'),
            "mean_entropy_outlier": float(df.loc[~known_mask, "entropy"].mean()) if (~known_mask).any() else float('nan'),
            "mean_max_prob_known": float(df.loc[known_mask, "max_prob"].mean()) if known_mask.any() else float('nan'),
            "mean_max_prob_outlier": float(df.loc[~known_mask, "max_prob"].mean()) if (~known_mask).any() else float('nan'),
        }

        os.makedirs(self.save_dir, exist_ok=True)
        csv_path = os.path.join(self.save_dir, f"sngp_uncertainty_{split_name}.csv")
        df.to_csv(csv_path, index=False)
        with open(os.path.join(self.save_dir, f"sngp_uncertainty_{split_name}_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

        try:
            stacked = numpy.concatenate(probs_batches, axis=0)
            global_habbas3.probs_list = stacked
        except Exception:
            global_habbas3.probs_list = probs_batches

        print(f"📈 SNGP uncertainty saved to {csv_path}")
        
    def _rebalance_training_loaders(self):
        """Attach class-balanced samplers to imbalanced source/target train loaders."""

        def _imbalance_ratio(labels: np.ndarray):
            if labels.size == 0:
                return None
            labels = labels[labels >= 0]
            if labels.size == 0:
                return None
            counts = np.bincount(labels.astype(int))
            counts = counts[counts > 0]
            if counts.size == 0:
                return None
            ratio = counts.max() / counts.min()
            threshold = float(getattr(self.args, 'rebalance_threshold', 1.6))
            return ratio if ratio >= threshold else None

        for name in ['source_train', 'target_train']:
            loader = self.dataloaders.get(name)
            if loader is None:
                continue

            sampler = getattr(loader, 'sampler', None)
            if isinstance(sampler, WeightedRandomSampler) or not isinstance(sampler, RandomSampler):
                # Custom samplers (e.g., WeightedRandomSampler) already handle class imbalance.
                continue

            dataset = loader.dataset
            labels = None
            for attr in ('targets', 'labels', 'y', 'ys'):
                if hasattr(dataset, attr):
                    data_attr = getattr(dataset, attr)
                    if torch.is_tensor(data_attr):
                        labels = data_attr.detach().cpu().numpy()
                    else:
                        labels = np.asarray(data_attr if isinstance(data_attr, (list, np.ndarray)) else list(data_attr))
                    break
            if labels is None:
                try:
                    max_scan = int(getattr(self.args, 'rebalance_scan_limit', 5000))
                    total_len = len(dataset)
                    if total_len > max_scan:
                        rng = np.random.RandomState(0)
                        indices = rng.choice(total_len, size=max_scan, replace=False)
                    else:
                        indices = range(total_len)
                    labels = np.array([
                        int(dataset[i][1].item()) if torch.is_tensor(dataset[i][1]) else int(dataset[i][1])
                        for i in indices
                    ])
                except Exception:
                    labels = np.array([])

            imbalance_ratio = _imbalance_ratio(labels)
            if imbalance_ratio is None:
                continue

            new_loader = _wrap_with_class_balanced_sampler(
                loader,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                device=self.device,
            )

            if new_loader is None:
                continue

            for attr in ('sequence_count', 'cycle_count'):
                if hasattr(loader, attr):
                    setattr(new_loader, attr, getattr(loader, attr))

            self.dataloaders[name] = new_loader
            if name == 'source_train':
                self.source_train_loader = new_loader
            else:
                self.target_train_loader = new_loader

            print(f"⚖️ Applied class-balanced sampling to {name} (imbalance ratio {imbalance_ratio:.2f}).")
            
    def _rebalance_training_loaders(self):
        """Attach class-balanced samplers to imbalanced source/target train loaders."""

        def _imbalance_ratio(labels: np.ndarray):
            if labels.size == 0:
                return None
            labels = labels[labels >= 0]
            if labels.size == 0:
                return None
            counts = np.bincount(labels.astype(int))
            counts = counts[counts > 0]
            if counts.size == 0:
                return None
            ratio = counts.max() / counts.min()
            threshold = float(getattr(self.args, 'rebalance_threshold', 1.6))
            return ratio if ratio >= threshold else None

        for name in ['source_train', 'target_train']:
            loader = self.dataloaders.get(name)
            if loader is None:
                continue

            sampler = getattr(loader, 'sampler', None)
            if isinstance(sampler, WeightedRandomSampler) or not isinstance(sampler, RandomSampler):
                # Custom samplers (e.g., WeightedRandomSampler) already handle class imbalance.
                continue

            dataset = loader.dataset
            labels = None
            for attr in ('targets', 'labels', 'y', 'ys'):
                if hasattr(dataset, attr):
                    data_attr = getattr(dataset, attr)
                    if torch.is_tensor(data_attr):
                        labels = data_attr.detach().cpu().numpy()
                    else:
                        labels = np.asarray(data_attr if isinstance(data_attr, (list, np.ndarray)) else list(data_attr))
                    break
            if labels is None:
                try:
                    max_scan = int(getattr(self.args, 'rebalance_scan_limit', 5000))
                    total_len = len(dataset)
                    if total_len > max_scan:
                        rng = np.random.RandomState(0)
                        indices = rng.choice(total_len, size=max_scan, replace=False)
                    else:
                        indices = range(total_len)
                    labels = np.array([
                        int(dataset[i][1].item()) if torch.is_tensor(dataset[i][1]) else int(dataset[i][1])
                        for i in indices
                    ])
                except Exception:
                    labels = np.array([])

            imbalance_ratio = _imbalance_ratio(labels)
            if imbalance_ratio is None:
                continue

            new_loader = _wrap_with_class_balanced_sampler(
                loader,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                device=self.device,
            )

            if new_loader is None:
                continue

            for attr in ('sequence_count', 'cycle_count'):
                if hasattr(loader, attr):
                    setattr(new_loader, attr, getattr(loader, attr))

            self.dataloaders[name] = new_loader
            if name == 'source_train':
                self.source_train_loader = new_loader
            else:
                self.target_train_loader = new_loader

            print(f"⚖️ Applied class-balanced sampling to {name} (imbalance ratio {imbalance_ratio:.2f}).")
        
    
    def _load_pretrained_weights(self, pretrained_path):
        print(f"🔁 Loading pretrained model from: {pretrained_path}")
        raw_state = torch.load(pretrained_path, map_location=self.device)

        # Some checkpoints wrap the actual state dict in another dict
        # (e.g. {"model_state_dict": ..., "optimizer": ...}).  When this is
        # the case we try to extract the state dict; otherwise we assume the
        # loaded object already represents a state dict.
        if isinstance(raw_state, dict) and "state_dict" in raw_state:
            state_dict = raw_state["state_dict"]
        elif isinstance(raw_state, dict) and "model_state_dict" in raw_state:
            state_dict = raw_state["model_state_dict"]
        else:
            state_dict = raw_state
        model_state = self.model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                skipped_keys.append(k)
        if skipped_keys:
            print(f"⚠️ Skipped keys due to size mismatch: {skipped_keys}")
        incompatible = self.model.load_state_dict(filtered_state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"⚠️ Missing keys when loading pretrained model: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"⚠️ Unexpected keys when loading pretrained model: {incompatible.unexpected_keys}")
            
        # --- Reset optimizer state after loading pretrained weights (prevents stale momentum) ---
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            try:
                # Clear momentum/adam moments
                self.optimizer.state.clear()
                # Reset LR on all param groups to current args.lr
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.args.lr
                print("🧽 Cleared optimizer state and reset learning rate for transfer fine-tuning.")
            except Exception as _e:
                print(f"⚠️ Optimizer reset skipped: {_e}")

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args
        

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))


        # Load datasets only if not already provided
        # Load datasets only if not already provided
        if self.source_train_loader is None or self.source_val_loader is None:
            if args.data_name == 'Battery_inconsistent':
                self.datasets = {}
                if hasattr(args, 'target_cathode') and args.target_cathode:
                    target_label = args.target_cathode[0] if isinstance(args.target_cathode, list) else args.target_cathode
                    print("Target Labels Sample:", str(target_label)[:5])
                else:
                    print("No target cathode provided — pretraining mode.")
        
                (
                    source_train,
                    source_val,
                    target_train,
                    target_val,
                    label_names,
                    df,
                    cycle_stats,
                ) = load_battery_dataset(
                    csv_path=self.args.csv,
                    source_cathodes=self.args.source_cathode,
                    target_cathodes=self.args.target_cathode,
                    classification_label=self.args.classification_label,
                    batch_size=self.args.batch_size,
                    sequence_length=self.args.sequence_length,
                    cycles_per_file=getattr(self.args, 'cycles_per_file', 50),
                    source_cycles_per_file=getattr(self.args, 'source_cycles_per_file', None),
                    target_cycles_per_file=getattr(self.args, 'target_cycles_per_file', None),
                    sample_random_state=getattr(self.args, 'sample_random_state', 42),
                )
        
                self.datasets['source_train'] = source_train
                self.datasets['source_val'] = source_val
                self.datasets['target_train'] = target_train
                self.datasets['target_val'] = target_val
                self.label_names = label_names
                self.df = df
                self.num_classes = len(label_names)
                self.dataset_cycle_stats = cycle_stats
                self.dataloaders = {
                    'source_train': source_train,
                    'source_val': source_val,
                    'target_train': target_train,
                    'target_val': target_val
                }
            else:
                if isinstance(args.transfer_task[0], str):
                    args.transfer_task = eval("".join(args.transfer_task))
                src_tr, src_val, tgt_tr, tgt_val, self.num_classes = Dataset(
                    args.data_dir, args.transfer_task, args.inconsistent, args.normlizetype
                ).data_split(transfer_learning=True)
                self.datasets = {
                    'source_train': src_tr,
                    'source_val': src_val,
                    'target_train': tgt_tr,
                    'target_val': tgt_val
                }
                self.dataloaders = {x: torch.utils.data.DataLoader(
                                        self.datasets[x],
                                        batch_size=args.batch_size,
                                        shuffle=(True if x.split('_')[1] == 'train' else False),
                                        num_workers=args.num_workers,
                                        pin_memory=(True if self.device == 'cuda' else False),
                                        drop_last=(True if args.last_batch and x.split('_')[1] == 'train' else False),
                                        generator=g)
                                    for x in ['source_train', 'source_val', 'target_train', 'target_val']}
        
        else:
            self.datasets = {
                'source_train': self.source_train_loader.dataset,
                'source_val': self.source_val_loader.dataset,
            }
            if self.target_train_loader is not None:
                self.datasets['target_train'] = self.target_train_loader.dataset
            if self.target_val_loader is not None:
                self.datasets['target_val'] = self.target_val_loader.dataset
            self.dataloaders = {
                'source_train': self.source_train_loader,
                'source_val': self.source_val_loader,
                'target_train': self.target_train_loader,
                'target_val': self.target_val_loader,
            }
            self.num_classes = args.num_classes
        
        # ---------- Helpers ----------
        def _has_batches(dl):
            if dl is None:
                return False
            try:
                return len(dl) > 0
            except TypeError:
                return False
        
        def _dataset_len(dl):
            ds = getattr(dl, 'dataset', None)
            try:
                return len(ds) if ds is not None else 0
            except Exception:
                return 0
        
        def _rebuild_nonempty_train_loader(name):
            """If a train loader has data but 0 batches (drop_last ate it), rebuild with drop_last=False."""
            dl = self.dataloaders.get(name)
            if dl is None:
                return
            if _has_batches(dl):
                return
            ds_len = _dataset_len(dl)
            if ds_len > 0:
                bs = min(args.batch_size, ds_len)
                self.dataloaders[name] = torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=bs,
                    shuffle=True,  # keep train shuffle behavior
                    num_workers=args.num_workers,
                    pin_memory=(True if self.device == 'cuda' else False),
                    drop_last=False
                )
                print(f"🩹 Rebuilt '{name}' with drop_last=False and batch_size={bs} (dataset_len={ds_len}).")
        
        def _first_nonempty_loader(dls, order=('source_train','target_train','source_val','target_val')):
            for name in order:
                dl = dls.get(name)
                if dl is None:
                    continue
                if _has_batches(dl):
                    return dl
                # If no batches but has data, rebuild a temporary iterator by turning off drop_last
                ds_len = _dataset_len(dl)
                if ds_len > 0:
                    bs = min(args.batch_size, ds_len)
                    return torch.utils.data.DataLoader(
                        dl.dataset, batch_size=bs, shuffle=False, num_workers=0, drop_last=False
                    )
            return None
        # -----------------------------
        
        # Fix 0-batch train loaders (tiny sets + drop_last)
        _rebuild_nonempty_train_loader('source_train')
        _rebuild_nonempty_train_loader('target_train')
        
        # Attach balanced samplers when large class skew is detected
        self._rebalance_training_loaders()
        
        
        # Target sample count (safe)
        self.target_sample_count = 0
        if self.dataloaders.get('target_train') is not None:
            self.target_sample_count = len(self.dataloaders['target_train'].dataset)
            # Class-balanced sampling for very small targets (Battery-inconsistent case)
            if self.args.data_name == 'Battery_inconsistent' and self.target_sample_count > 0:
                tb = self.args.batch_size
                nw = self.args.num_workers
                self.dataloaders['target_train'] = _wrap_with_class_balanced_sampler(
                    self.dataloaders['target_train'], batch_size=tb, num_workers=nw, device=self.device
                )
                print("⚖️  Enabled class-balanced sampling for target_train.")

            if self.target_sample_count < 100:
                args.lr = min(args.lr, 1e-4)
                logging.info(f"Reducing learning rate to {args.lr} for {self.target_sample_count} target samples")
                
        # --- Target class prior for balanced regularization ---
        self.target_class_prior = None
        target_labels = None
        tgt_loader = self.dataloaders.get('target_train')
        if tgt_loader is not None:
            dataset = tgt_loader.dataset
            for attr in ('targets', 'labels', 'y', 'ys'):
                if hasattr(dataset, attr):
                    data_attr = getattr(dataset, attr)
                    if torch.is_tensor(data_attr):
                        target_labels = data_attr.detach().cpu().numpy()
                    else:
                        target_labels = np.asarray(data_attr if isinstance(data_attr, (list, np.ndarray)) else list(data_attr))
                    break
            if target_labels is None:
                try:
                    extracted = []
                    for i in range(len(dataset)):
                        lbl = dataset[i][1]
                        if torch.is_tensor(lbl):
                            extracted.append(int(lbl.item()))
                        else:
                            extracted.append(int(lbl))
                    target_labels = np.array(extracted)
                except Exception:
                    target_labels = None

        if target_labels is not None and target_labels.size > 0:
            known_target = target_labels[target_labels < self.num_classes]
            if known_target.size > 0:
                counts = np.bincount(known_target.astype(int), minlength=self.num_classes).astype(np.float32)
                if counts.sum() > 0:
                    prior = counts / counts.sum()
                    prior = np.clip(prior, 1e-6, None)
                    prior = prior / prior.sum()
                    self.target_class_prior = torch.tensor(prior, dtype=torch.float)
                    print(f"🎯 Target class prior (known classes): {prior.tolist()}")

        default_balance_lambda = 0.05 if getattr(self.args, 'data_name', None) == 'Battery_inconsistent' else 0.0
        self.target_balance_lambda = float(getattr(self.args, 'target_balance_lambda', default_balance_lambda))
        if self.target_balance_lambda > 0 and self.target_class_prior is None and self.num_classes > 0:
            uniform_prior = np.full(self.num_classes, 1.0 / float(self.num_classes), dtype=np.float32)
            self.target_class_prior = torch.tensor(uniform_prior, dtype=torch.float)
            print("🎯 Target prior defaulted to uniform distribution.")
        if self.target_balance_lambda > 0:
            print(f"🧮 Target balance regularization λ={self.target_balance_lambda:.4f}")

        tgt_dl = self.dataloaders.get('target_train')
        
        # Determine transfer mode only if target_train actually has data
        self.transfer_mode = (self.target_sample_count > 0 and getattr(self.args, 'pretrained_model_path', None))

        
        # Pick a NON-EMPTY sample loader for inferring input shape
        sample_loader = _first_nonempty_loader(self.dataloaders, ('source_train','target_train','source_val','target_val'))
        if sample_loader is None:
            raise RuntimeError(
                "All Battery dataloaders are empty. Check filters (source_cathode, target_cathode, "
                "classification_label, sequence_length) — they may produce zero samples."
            )
        
        first_batch = next(iter(sample_loader))
        input_tensor, _ = first_batch
        args.input_channels = input_tensor.shape[1]  # Automatically infer input channels from data
        args.input_size = input_tensor.shape[-1]
        print("🧪 Input shape before CNN:", input_tensor.shape)
        
        # Choose a NON-EMPTY train loader for max_iter
        if self.transfer_mode and _has_batches(tgt_dl):
            train_loader_for_iter = tgt_dl
        else:
            src_dl = self.dataloaders.get('source_train')
            train_loader_for_iter = src_dl if _has_batches(src_dl) else sample_loader
        
        self.max_iter = len(train_loader_for_iter) * args.max_epoch

        if args.model_name in ["cnn_openmax", "cnn_features_1d_sa", "cnn_features_1d","WideResNet", "WideResNet_sa", "WideResNet_mh", "WideResNet_edited"]:
            if args.model_name == "WideResNet":
                self.model = WideResNet(
                    args.layers,
                    args.widen_factor,
                    args.droprate,
                    out_features=self.num_classes,
                    in_channels=args.input_channels,            
                    bottleneck=args.bottleneck_num
                )
            elif args.model_name == "WideResNet_sa":
                self.model = WideResNet_sa(
                    args.layers,
                    args.widen_factor,
                    args.droprate,
                    num_classes=self.num_classes,
                    num_input_channels=args.input_channels       
                )
            elif args.model_name == "WideResNet_edited":
                self.model = WideResNet_edited(
                    depth=args.layers,
                    widen_factor=args.widen_factor,
                    drop_rate=args.droprate,
                    num_classes=self.num_classes,
                    input_channels=args.input_channels          
                )
            elif args.model_name == "cnn_openmax":
                self.model = cnn_openmax(args, self.num_classes)
            elif args.model_name == "cnn_features_1d":
                from models.cnn_1d import cnn_features
                self.model = cnn_features(pretrained=args.pretrained, in_channels=args.input_channels)
            elif args.model_name == "cnn_features_1d_sa":
                from models.cnn_1d_selfattention_habbas import cnn_features
                self.model = cnn_features(pretrained=args.pretrained, in_channels=args.input_channels)
        
            
            first_conv = next((m for m in self.model.modules() if isinstance(m, nn.Conv1d)), None)
            if first_conv is not None:
                kernel = first_conv.kernel_size[0] if isinstance(first_conv.kernel_size, tuple) else first_conv.kernel_size
                if args.input_size < kernel:
                    raise ValueError(
                        f"Input sequence length {args.input_size} is smaller than the first convolution kernel size {kernel}. "
                        "Increase sequence_length or use a model with a smaller kernel."
                    )

            backbone_output_dim = self.model.output_num()
            pretrained_path = getattr(args, "pretrained_model_path", None)
            if pretrained_path and os.path.isfile(pretrained_path):
                self._load_pretrained_weights(pretrained_path)
            
            # if getattr(self.args, "transfer", False) and getattr(self.args, "pretrained_model_path", None):
            #     print(f"🔁 Loading pretrained model from {self.args.pretrained_model_path}")
            #     self.model.load_state_dict(torch.load(self.args.pretrained_model_path, map_location=self.device))
        
            if args.bottleneck:
                self.bottleneck_layer = nn.Sequential(
                    nn.Linear(backbone_output_dim, args.bottleneck_num),
                    nn.ReLU(),
                    nn.Dropout(p=args.droprate)
                )
                final_feature_dim = args.bottleneck_num
            else:
                self.bottleneck_layer = nn.Identity()
                final_feature_dim = backbone_output_dim
        
        else:
            # generic model loader
            model_cls = getattr(models, args.model_name)
            try:
                self.model = model_cls(args.pretrained, in_channel=args.input_channels)
            except TypeError:
                self.model = model_cls(args.pretrained)
            first_conv = next((m for m in self.model.modules() if isinstance(m, nn.Conv1d)), None)
            if first_conv is not None:
                kernel = first_conv.kernel_size[0] if isinstance(first_conv.kernel_size, tuple) else first_conv.kernel_size
                if args.input_size < kernel:
                    raise ValueError(
                        f"Input sequence length {args.input_size} is smaller than the first convolution kernel size {kernel}. "
                        "Increase sequence_length or use a model with a smaller kernel."
                    )
                    
            backbone_output_dim = self.model.output_num()
            pretrained_path = getattr(args, "pretrained_model_path", None)
            if pretrained_path and os.path.isfile(pretrained_path):
                self._load_pretrained_weights(pretrained_path)
                
            if args.bottleneck:
                self.bottleneck_layer = nn.Sequential(
                    nn.Linear(backbone_output_dim, args.bottleneck_num),
                    nn.ReLU(),
                    nn.Dropout(p=args.droprate)
                )
                final_feature_dim = args.bottleneck_num
            else:
                self.bottleneck_layer = nn.Identity()
                final_feature_dim = backbone_output_dim

                
                

        # self.bottleneck_layer = nn.Sequential(nn.Linear(output_features, args.bottleneck_num),
        #                                       nn.ReLU(inplace=True), nn.Dropout())
        if args.inconsistent == 'OSBP':
            if args.bottleneck:
                self.classifier_layer = getattr(models, 'classifier_OSBP')(in_feature=args.bottleneck_num,
                                                                           output_num=self.num_classes + 1,
                                                                           max_iter=self.max_iter,
                                                                           trade_off_adversarial=args.trade_off_adversarial,
                                                                           lam_adversarial=args.lam_adversarial
                                                                           )
            else:
                self.classifier_layer = getattr(models, 'classifier_OSBP')(in_feature=self.model.output_num(),
                                                                           output_num=self.num_classes + 1,
                                                                           max_iter=self.max_iter,
                                                                           trade_off_adversarial=args.trade_off_adversarial,
                                                                           lam_adversarial=args.lam_adversarial
                                                                           )
        else:
            if args.bottleneck:

                self.classifier_layer = nn.Linear(args.bottleneck_num, self.num_classes)

                if not self.transfer_mode:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=args.bottleneck_num,
                                                                            hidden_size=args.hidden_size,
                                                                            max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial)
                    self.AdversarialNet_auxiliary = getattr(models, 'AdversarialNet_auxiliary')(in_feature=args.bottleneck_num,
                                                                                                hidden_size=args.hidden_size)
                else:
                    self.AdversarialNet = None
                    self.AdversarialNet_auxiliary = None
            else:
                self.classifier_layer = nn.Linear(self.model.output_num(), self.num_classes)
                if not self.transfer_mode:
                    self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=self.model.output_num(),
                                                                            hidden_size=args.hidden_size,
                                                                            max_iter=self.max_iter,
                                                                            trade_off_adversarial=args.trade_off_adversarial,
                                                                            lam_adversarial=args.lam_adversarial)
                    self.AdversarialNet_auxiliary = getattr(models, 'AdversarialNet_auxiliary')(
                        in_feature=self.model.output_num(),
                        hidden_size=args.hidden_size)
                else:
                    self.AdversarialNet = None
                    self.AdversarialNet_auxiliary = None
        if args.bottleneck:
            self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)
        else:
            self.model_all = nn.Sequential(self.model, self.classifier_layer)
            
            
        # --- Optional: re-initialize the final classifier head for target task ---
        reinit_head = getattr(self.args, "reinit_head", True)
        if self.transfer_mode and reinit_head:
            last_fc = _find_last_linear(self.model)
            if last_fc is not None and getattr(last_fc, 'out_features', None) == self.num_classes:
                _kaiming_reset_linear(last_fc)
                print("\ud83d\udd01 Reinitialized final classifier layer for target task.")
            else:
                print("\u2139\ufe0f Skipped head reinit (no Linear head found with matching out_features).")

        # --- Capture L2-SP reference (pretrained weights copy) ---
        self._l2sp_ref = None
        lambda_l2sp = float(getattr(self.args, "lambda_l2sp", 1e-3))
        if self.transfer_mode and lambda_l2sp > 0:
            self._l2sp_ref = {n: p.detach().clone().to(self.device)
                              for n, p in self.model.named_parameters() if p.requires_grad}
            self.lambda_l2sp = lambda_l2sp
            print(f"\ud83e\uddf2 L2-SP active with \u03bb={self.lambda_l2sp}")
        else:
            self.lambda_l2sp = 0.0
        
        # Freeze early backbone layers if target data are scarce
        if self.target_sample_count and self.target_sample_count < 100:
            logging.info(f"Freezing early layers of backbone for {self.target_sample_count} target samples")
            children = list(self.model.children())
            for child in children[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.bottleneck:
                self.bottleneck_layer = torch.nn.DataParallel(self.bottleneck_layer)
            if args.inconsistent == 'UAN' and self.AdversarialNet is not None:
                self.AdversarialNet = torch.nn.DataParallel(self.AdversarialNet)
                self.AdversarialNet_auxiliary = torch.nn.DataParallel(self.AdversarialNet_auxiliary)
            self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)
            
        # --------------------------------------------------------------
        # Add SNGP (or deterministic) classification head so that its
        # parameters can be optimized together with the backbone.
        # --------------------------------------------------------------
        self.backbone = self.model
        if args.method == 'sngp':
            self.sngp_model = sngp(
                backbone=self.model,
                bottleneck_num=final_feature_dim,
                num_classes=self.num_classes,
                num_inducing=args.gp_hidden_dim,
                n_power_iterations=args.n_power_iterations,
                spec_norm_bound=args.spectral_norm_bound,
                device=self.device,
                normalize_input=False,
            )
        else:
            self.sngp_model = deterministic(
                self.backbone,
                bottleneck_num=final_feature_dim,
                num_classes=self.num_classes,
            )
        # move gp head to the correct device
        self.sngp_model.to(self.device)

        # Define the learning parameters
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        if args.inconsistent == "OSBP":
            parameter_list = []
            if model_params:
                parameter_list.append({"params": model_params, "lr": args.lr})
                
            if args.bottleneck:
                parameter_list.append({"params": self.bottleneck_layer.parameters(), "lr": args.lr})
            parameter_list.append({"params": self.classifier_layer.parameters(), "lr": args.lr})
        else:
            parameter_list = []
            if model_params:
                parameter_list.append({"params": model_params, "lr": args.lr})
            if args.bottleneck:
                parameter_list.append({"params": self.bottleneck_layer.parameters(), "lr": args.lr})
            parameter_list.append({"params": self.classifier_layer.parameters(), "lr": args.lr})
            if self.AdversarialNet_auxiliary is not None:
                parameter_list.append({"params": self.AdversarialNet_auxiliary.parameters(), "lr": args.lr})
            if self.AdversarialNet is not None:
                parameter_list.append({"params": self.AdversarialNet.parameters(), "lr": args.lr})
                
        # Ensure SNGP (or deterministic head) parameters are optimized.
        sngp_params = [p for name, p in self.sngp_model.named_parameters()
                       if 'backbone' not in name]
        if sngp_params:
            parameter_list.append({"params": sngp_params, "lr": args.lr})
            
        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(parameter_list, lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(parameter_list, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")
            
        # --- Discriminative LR for transfer: tiny LR for backbone, larger for head ---
        if self.transfer_mode:
            head = _find_last_linear(self.model)
            head_params = list(head.parameters()) if head is not None else []
            head_param_ids = {id(p) for p in head_params}
            base_params = [p for p in self.model.parameters() if id(p) not in head_param_ids]

            lr_head = float(self.args.lr)
            lr_base = float(self.args.lr) * float(getattr(self.args, "backbone_lr_mult", 0.1))
            wd = float(self.args.weight_decay)

            opt_name = str(getattr(self.args, "opt", "adam")).lower()
            if opt_name == "sgd":
                self.optimizer = torch.optim.SGD(
                    [
                        {"params": base_params, "lr": lr_base, "weight_decay": wd},
                        {"params": head_params, "lr": lr_head, "weight_decay": wd},
                    ],
                    momentum=self.args.momentum,
                    nesterov=getattr(self.args, "nesterov", True),
                )
            else:
                # default: AdamW/Adam path
                if opt_name == "adamw":
                    Opt = torch.optim.AdamW
                else:
                    Opt = torch.optim.Adam
                self.optimizer = Opt(
                    [
                        {"params": base_params, "lr": lr_base, "weight_decay": wd},
                        {"params": head_params, "lr": lr_head, "weight_decay": wd},
                    ]
                )
            print(f"🔧 Discriminative LR set → backbone: {lr_base:g}, head: {lr_head:g}")


        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")


        self.start_epoch = 0
        
        if self.transfer_mode and hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.state.clear()
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.args.lr
            print("🧽 Cleared optimizer state and reset LR for transfer.")


        # Invert the model and define the loss
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if self.AdversarialNet is not None:
            self.AdversarialNet.to(self.device)
            self.AdversarialNet_auxiliary.to(self.device)
        self.classifier_layer.to(self.device)
        self.sngp_model.to(self.device)
        
        if self.transfer_mode and self.dataloaders.get('target_train') is not None:
            with torch.no_grad():
                for p in self.model.parameters():
                    p.requires_grad = False
                calibrate_bn(self.model, self.dataloaders['target_train'], self.device, max_batches=50)
                for p in self.model.parameters():
                    p.requires_grad = True

        if args.inconsistent == "OSBP":
            self.inconsistent_loss = nn.BCELoss()

        # ---------------------------------------------
        # Determine class weights for imbalanced data.
        # Earlier revisions assumed the presence of a local
        # ``df`` variable containing the original dataframe,
        # which is not available when dataloaders are passed
        # directly (e.g. for the CWRU dataset).  We now fall
        # back to dataset or dataloader labels if ``df`` is
        # missing so both Battery and CWRU paths work.
        # ---------------------------------------------
        classification_label = getattr(args, "classification_label", None)
        all_labels = None

        # 1) Preferred source: dataframe stored during battery loading
        if hasattr(self, "df") and classification_label and classification_label in self.df.columns:
            all_labels = self.df[classification_label].values

        # 2) Fallback: labels attribute on the source dataset
        if all_labels is None:
            src_dataset = self.datasets.get('source_train')
            if hasattr(src_dataset, 'labels') and src_dataset.labels is not None:
                all_labels = np.array(src_dataset.labels)

        # 3) Final fallback: iterate over the dataloader to collect labels
        if all_labels is None and self.dataloaders.get('source_train') is not None:
            collected = []
            for _, lbl in self.dataloaders['source_train']:
                collected.extend(lbl.numpy().tolist() if isinstance(lbl, torch.Tensor) else lbl)
            if collected:
                all_labels = np.array(collected)
                
        # Blend in target-side supervision when available so the class weights
        # reflect the distribution the transfer step actually trains on.
        if target_labels is not None:
            target_labels_arr = np.asarray(target_labels)
            tgt_known = target_labels_arr[target_labels_arr < self.num_classes]
            if tgt_known.size > 0:
                if all_labels is None:
                    all_labels = tgt_known.copy()
                else:
                    all_labels = np.concatenate([all_labels, tgt_known])


        if all_labels is not None and len(all_labels) > 0:
            # -------------------------------------------------------------
            # Some datasets (e.g. CWRU) include explicit *outlier* classes
            # whose label index is greater than or equal to ``self.num_classes``.
            # The model, however, only outputs ``self.num_classes`` logits, so
            # class weights must be computed only over the known-class labels
            # to avoid a size mismatch in ``nn.CrossEntropyLoss``.  Battery
            # datasets contain no such labels, making the mask a harmless no-op.
            # -------------------------------------------------------------
            # Determine the number of output classes directly from the
            # classification head.  Relying on ``args.num_classes`` can be
            # misleading for setups such as OSBP where the classifier outputs
            # an additional ``unknown`` class.  Inspect the modules (while
            # unwrapping any DataParallel containers) and fall back to
            # ``self.num_classes`` only if no explicit ``out_features`` attribute
            # is found.

            def _unwrap(module):
                return module.module if hasattr(module, 'module') else module

            num_output_classes = None
            for head in (_unwrap(getattr(self, 'classifier_layer', None)),
                         _unwrap(getattr(self, 'sngp_model', None))):
                if head is None:
                    continue
                if hasattr(head, 'fc') and hasattr(head.fc, 'out_features'):
                    num_output_classes = head.fc.out_features
                    break
                if hasattr(head, 'out_features'):
                    num_output_classes = head.out_features
                    break

            if num_output_classes is None:
                # As a final fallback, infer the output dimension directly
                # from the model's forward pass.  This covers architectures
                # where the backbone itself produces logits (e.g., when the
                # classifier head is fused into the model or an extra unknown
                # class is appended).  Running a tiny batch through the model
                # allows us to determine the true number of logits so that the
                # class-weight vector can be sized correctly for all classes.
                try:
                    sample = next(iter(self.dataloaders['source_train']))[0][:1].to(self.device)
                    prev_mode = self.model.training
                    self.model.eval()
                    with torch.no_grad():
                        out = self.model(sample)
                        if isinstance(out, tuple):
                            out = out[0]
                        num_output_classes = out.shape[1]
                    if prev_mode:
                        self.model.train()
                except Exception:
                    num_output_classes = getattr(self.args, 'num_classes', self.num_classes)

            # ``self.num_classes`` tracks the number of *known* classes. Any
            # label >= this is treated as an outlier when computing the
            # supervised loss.
            known_mask = all_labels < self.num_classes

            
            if np.any(known_mask):
                present_classes = np.unique(all_labels[known_mask])
                balanced_weights = compute_class_weight(
                    'balanced',
                    classes=present_classes,
                    y=all_labels[known_mask]
                )

                full_weights = np.ones(num_output_classes, dtype=np.float32)
                for cls, w in zip(present_classes, balanced_weights):
                    full_weights[int(cls)] = w

                weights_tensor = torch.tensor(full_weights, dtype=torch.float, device=self.device)

                # ``CrossEntropyLoss`` requires that the provided weight vector
                # matches the number of model outputs.  If anything goes wrong
                # in the above bookkeeping (e.g. missing class indices), fall
                # back to an unweighted loss rather than raising an exception.
                if weights_tensor.numel() == num_output_classes:
                    self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
                else:
                    self.criterion = nn.CrossEntropyLoss()
            else:
                # All labels correspond to outlier classes; fall back to
                # an unweighted loss to keep training functional.
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        

    def train(self):
        best_eval_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_hscore = 0.0
        best_common_acc = 0.0
        best_balanced_acc = 0.0
        patience = getattr(self.args, 'early_stop_patience', None)
        epochs_no_improve = 0
        
        self.warmup_epochs = int(getattr(self.args, "warmup_epochs", 3))
        self._head_module = _find_last_linear(self.model)
        self._train_start_time = time.time()
        self._best_metric_time = None
        self._best_epoch = 0
        
        self.lambda_src_current = self.lambda_src_init
        self._lambda_plateau_epochs = 0
    
        for epoch in range(self.args.max_epoch):
            source_iter = None
            if self.transfer_mode and self.dataloaders.get('source_train') is not None:
                source_iter = cycle(self.dataloaders['source_train'])
            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} ----- Epoch {epoch + 1}/{self.args.max_epoch} -----")
            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if self.transfer_mode and self._head_module is not None:
                if epoch < self.warmup_epochs:
                    for p in self.model.parameters():
                        p.requires_grad = False
                    for p in self._head_module.parameters():
                        p.requires_grad = True
                    if epoch == 0:
                        print(f"🧊 Linear-probe warmup for {self.warmup_epochs} epochs (head-only).")
                elif epoch == self.warmup_epochs:
                    for p in self.model.parameters():
                        p.requires_grad = True
                    print("🔥 Unfroze backbone after warmup.")
            
            if self.transfer_mode:
                phases = ['target_train']
                if self.dataloaders.get('target_val') is not None:
                    phases.append('target_val')
            else:
                phases = ['source_train']
                if self.dataloaders.get('source_val') is not None:
                    phases.append('source_val')
                if self.dataloaders.get('target_val') is not None:
                    phases.append('target_val')
                    
            val_improved = False

            for phase in phases:
                if self.dataloaders.get(phase) is None:
                    continue
                
                if (phase == 'target_val' and self.transfer_mode and
                        self.dataloaders.get('target_train') is not None):
                    calibrate_bn(
                        self.model,
                        self.dataloaders['target_train'],
                        self.device,
                        max_batches=int(getattr(self.args, 'bn_calibration_batches', 32))
                    )
                
                    
                is_training_phase = phase.endswith('train')
                self.model.train() if is_training_phase else self.model.eval()
                if hasattr(self, 'sngp_model'):
                    self.sngp_model.train() if is_training_phase else self.sngp_model.eval()
    
                running_loss = 0.0
                running_corrects = 0
                running_total = 0
                preds_all, labels_all = [], []

                for step, (inputs, labels) in enumerate(self.dataloaders[phase]):
                        
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
    
                    if inputs.dim() == 2:
                        inputs = inputs.unsqueeze(1)
                    if inputs.shape[1] != self.args.input_channels and inputs.shape[-1] == self.args.input_channels:
                        inputs = inputs.permute(0, 2, 1)
    
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase.endswith('train')):
                        out = self.model(inputs)
                        if isinstance(out, tuple):
                            model_logits = out[0]
                            features = out[1] if len(out) > 1 else None
                        else:
                            model_logits, features = out, None
    
                        if features is not None and self.args.bottleneck:
                            features = self.bottleneck_layer(features)

                        logits = (
                            self.sngp_model.forward_classifier(
                                features,
                                update_cov=is_training_phase,
                                apply_mean_field=not is_training_phase,
                            )
                            if features is not None
                            else model_logits
                        )

                        known_mask_batch = labels < self.num_classes

                        # --- Defensive fix for class-weight size mismatches (unchanged logic) ---
                        try:
                            ce_weight = getattr(self.criterion, 'weight', None)
                            if ce_weight is not None and ce_weight.numel() != logits.shape[1]:
                                with torch.no_grad():
                                    import numpy as _np
                                    from sklearn.utils.class_weight import compute_class_weight as _ccw
                                    num_logits = logits.shape[1]
                        
                                    # Recompute balanced weights using only *known-class* labels
                                    _mask = labels < self.num_classes
                                    if _mask.any():
                                        present = _np.unique(labels[_mask].detach().cpu().numpy()).astype(int)
                                        balanced = _ccw('balanced', classes=present,
                                                        y=labels[_mask].detach().cpu().numpy())
                        
                                        # Expand to full length (unknown logit stays at 1.0)
                                        full = _np.ones(num_logits, dtype=_np.float32)
                                        for cls, w in zip(present, balanced):
                                            if int(cls) < num_logits:
                                                full[int(cls)] = float(w)
                        
                                        new_w = torch.tensor(full, device=self.device, dtype=torch.float)
                                        self.criterion = nn.CrossEntropyLoss(weight=new_w)
                                    else:
                                        # No known-class samples this batch → unweighted
                                        self.criterion = nn.CrossEntropyLoss()
                        except Exception:
                            # If anything goes wrong, fall back to unweighted to keep training running.
                            self.criterion = nn.CrossEntropyLoss()
                        # -----------------------------------------------------------------------
                        
                        # Target supervised loss (known classes only)
                        loss_tgt = None
                        total_loss = None
                        if known_mask_batch.any():
                            loss_tgt = self.criterion(logits[known_mask_batch], labels[known_mask_batch])
                            total_loss = loss_tgt
                        
                        # >>> NEW: supervised SOURCE loss mixed in every target step (during transfer) <<<
                        if (self.transfer_mode and phase == 'target_train'
                                and self.dataloaders.get('source_train') is not None
                                and self.lambda_src_current > 0):
                            # Use a persistent iterator so we don't always read the first batch
                            if not hasattr(self, '_source_iter') or self._source_iter is None:
                                self._source_iter = iter(self.dataloaders['source_train'])
                            try:
                                src_inputs, src_labels = next(self._source_iter)
                            except StopIteration:
                                self._source_iter = iter(self.dataloaders['source_train'])
                                src_inputs, src_labels = next(self._source_iter)
                        
                            src_inputs = src_inputs.to(self.device, non_blocking=True)
                            src_labels = src_labels.to(self.device, non_blocking=True)
                        
                            # shape fixes consistent with your target path
                            if src_inputs.dim() == 2:
                                src_inputs = src_inputs.unsqueeze(1)
                            if src_inputs.shape[1] != self.args.input_channels and src_inputs.shape[-1] == self.args.input_channels:
                                src_inputs = src_inputs.permute(0, 2, 1)
                        
                            src_out = self.model(src_inputs)
                            src_logits = src_out[0] if isinstance(src_out, tuple) else src_out
                        
                            # ---- Guard against out-of-bounds source targets ----
                            src_num_logits = src_logits.shape[1]
                            valid_src_mask = (src_labels >= 0) & (src_labels < src_num_logits)
                            if valid_src_mask.any():
                                src_logits_valid = src_logits[valid_src_mask]
                                src_labels_valid = src_labels[valid_src_mask]
                        
                                # --- Ensure class-weight length matches SRC logits ---
                                crit_src = self.criterion
                                try:
                                    ce_weight = getattr(crit_src, 'weight', None)
                                    if ce_weight is not None and ce_weight.numel() != src_num_logits:
                                        with torch.no_grad():
                                            import numpy as _np
                                            from sklearn.utils.class_weight import compute_class_weight as _ccw
                        
                                            present_src = _np.unique(src_labels_valid.detach().cpu().numpy()).astype(int)
                                            balanced_src = _ccw('balanced', classes=present_src,
                                                                y=src_labels_valid.detach().cpu().numpy())
                        
                                            full_src = _np.ones(src_num_logits, dtype=_np.float32)
                                            for cls, w in zip(present_src, balanced_src):
                                                if int(cls) < src_num_logits:
                                                    full_src[int(cls)] = float(w)
                        
                                            new_w_src = torch.tensor(full_src, device=self.device, dtype=torch.float)
                                            crit_src = nn.CrossEntropyLoss(weight=new_w_src)
                                except Exception:
                                    # If anything goes wrong, fall back to unweighted to keep training running.
                                    crit_src = nn.CrossEntropyLoss()
                                # ---------------------------------------------------
                        
                                loss_src = crit_src(src_logits_valid, src_labels_valid)
                                lam_src = self.lambda_src_current
                                total_loss = loss_src * lam_src if total_loss is None else total_loss + lam_src * loss_src
                            # If no valid src labels for this step, just skip source loss for this batch.
                        # <<< END NEW >>>
                        
                        # Encourage balanced target predictions by matching the batch-average
                        # probability distribution to the empirical target prior (or uniform fallback).
                        balance_lambda = getattr(self, 'target_balance_lambda', 0.0)
                        if (self.transfer_mode and phase == 'target_train' and balance_lambda > 0
                                and known_mask_batch.any()):
                            probs = torch.softmax(logits[known_mask_batch], dim=1)
                            mean_probs = probs.mean(dim=0)
                            prior = getattr(self, 'target_class_prior', None)
                            if prior is not None:
                                prior = prior.to(self.device)
                                if prior.device == self.device:
                                    self.target_class_prior = prior.detach()
                                if prior.numel() != mean_probs.numel():
                                    prior = torch.ones_like(mean_probs) / float(mean_probs.numel())
                            else:
                                prior = torch.ones_like(mean_probs) / float(mean_probs.numel())

                            mean_probs = mean_probs.clamp_min(1e-6)
                            prior = prior.clamp_min(1e-6)
                            prior = prior / prior.sum()
                            mean_probs = mean_probs / mean_probs.sum()
                            balance_loss = torch.sum(mean_probs * (mean_probs.log() - prior.log()))
                            total_loss = total_loss + balance_lambda * balance_loss if total_loss is not None else balance_lambda * balance_loss
                        
                        # L2-SP: keep weights near pretrained solution (helps on tiny/different targets)
                        if self.transfer_mode and getattr(self, "lambda_l2sp", 0.0) > 0 and getattr(self, "_l2sp_ref", None) is not None:
                            reg = torch.tensor(0.0, device=self.device)
                            for (name, p) in self.model.named_parameters():
                                if p.requires_grad and name in self._l2sp_ref:
                                    reg = reg + (p - self._l2sp_ref[name]).pow(2).sum()
                            total_loss = total_loss + self.lambda_l2sp * reg if total_loss is not None else self.lambda_l2sp * reg
                        
                        # Backprop/step (relies on zero_grad where you already do it in your loop)
                        if phase.endswith('train') and total_loss is not None:
                            total_loss.backward()
                            self.optimizer.step()
                        
                        # Accounting based only on target-known supervision for logging
                        if loss_tgt is not None:
                            running_loss += loss_tgt.item() * known_mask_batch.sum().item()
                        else:
                            # No valid known-class samples in this batch; keep loss at 0 for logging
                            loss_tgt = torch.tensor(0.0, device=self.device)


                       
                    _, preds = torch.max(logits, 1)
                    
                    
                    running_corrects += torch.sum(preds == labels.data)
                    running_total += labels.size(0)
    
                    preds_all.extend(preds.detach().cpu().numpy())
                    labels_all.extend(labels.detach().cpu().numpy())
                    
    
                epoch_loss = running_loss / running_total if running_total > 0 else 0.0
                epoch_acc = running_corrects.double() / running_total if running_total > 0 else torch.tensor(0.0)
                epoch_acc_value = float(epoch_acc)
                balanced_acc_value = float('nan')

                if phase == 'target_val':
                    preds_np = np.array(preds_all)
                    labels_np = np.array(labels_all)
                    known_eval_mask = labels_np < self.num_classes
                    if known_eval_mask.any():
                        unique_known = np.unique(labels_np[known_eval_mask])
                        if unique_known.size == 1:
                            balanced_acc_value = float(np.mean(preds_np[known_eval_mask] == labels_np[known_eval_mask]))
                        else:
                            try:
                                balanced_acc_value = float(
                                    balanced_accuracy_score(labels_np[known_eval_mask], preds_np[known_eval_mask])
                                )
                            except Exception:
                                balanced_acc_value = float('nan')

                log_msg = [
                    f"{datetime.now().strftime('%m-%d %H:%M:%S')} Epoch: {epoch}",
                    f"{phase}-Loss: {epoch_loss:.4f}",
                    f"{phase}-Acc: {epoch_acc_value:.4f}"
                ]
                
                if phase == 'target_train' and self.transfer_mode:
                    log_msg.append(f"lambda_src: {self.lambda_src_current:.4f}")
                if phase == 'target_val':
                    if not math.isnan(balanced_acc_value):
                        log_msg.append(f"{phase}-BalAcc: {balanced_acc_value:.4f}")
                    else:
                        log_msg.append(f"{phase}-BalAcc: nan")
                print(' '.join(log_msg))

                metric_for_selection = epoch_acc_value
                if phase == 'target_val' and not math.isnan(balanced_acc_value):
                    metric_for_selection = balanced_acc_value

                if phase in ['source_val', 'target_val'] and metric_for_selection > best_eval_acc:
                    best_eval_acc = metric_for_selection
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    print("✓ Best model updated based on validation metric.")
                    val_improved = True
                    self._best_epoch = epoch
                    self._best_metric_time = time.time() - self._train_start_time
            
            
    
                if phase == 'target_val':
                    preds_np = np.array(preds_all)
                    labels_np = np.array(labels_all)
                    known_mask = labels_np < self.num_classes
                    out_mask = labels_np >= self.num_classes
                
                    # Common = exact match on known classes
                    common_acc = (
                        float(np.mean(preds_np[known_mask] == labels_np[known_mask]))
                        if known_mask.any() else 0.0
                    )
                    # Outlier = predict any index >= num_known for outlier labels
                    outlier_acc = (
                        float(np.mean(preds_np[out_mask] >= self.num_classes))
                        if out_mask.any() else 0.0
                    )
                
                    if not out_mask.any() and known_mask.any():
                        hscore = common_acc
                    elif not known_mask.any() and out_mask.any():
                        hscore = outlier_acc
                    else:
                        denom = common_acc + outlier_acc
                        hscore = (2 * common_acc * outlier_acc / denom) if denom > 0 else 0.0
                
                    improved_common = common_acc > best_common_acc
                    improved_h = hscore > best_hscore
                
                    if improved_common:
                        best_common_acc = common_acc
                    if improved_h:
                        best_hscore = hscore
                
                    metrics_log = (
                        f"{datetime.now().strftime('%m-%d %H:%M:%S')} Epoch: {epoch} {phase}-common: {common_acc:.4f} "
                        f"outlier: {outlier_acc:.4f} hscore: {hscore:.4f} "
                    )
                    if not math.isnan(balanced_acc_value):
                        metrics_log += f"bal_acc: {balanced_acc_value:.4f}"
                    else:
                        metrics_log += "bal_acc: nan"
                    print(metrics_log)
                
                    # Also update BEST MODEL on improved hscore/common so we save the right model for transfer
                    improved_balanced = (not math.isnan(balanced_acc_value)) and (balanced_acc_value > best_balanced_acc)
                    if improved_balanced:
                        best_balanced_acc = balanced_acc_value
                    if improved_h or improved_common or improved_balanced:
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        val_improved = True
                        print("✓ Best target model updated based on target metrics (hscore/common/balanced).")
                        self._best_epoch = epoch
                        self._best_metric_time = time.time() - self._train_start_time
                        
                        
            if self.transfer_mode and self.dataloaders.get('target_train') is not None:
                if val_improved:
                    self._lambda_plateau_epochs = 0
                else:
                    if epoch >= self.lambda_src_warmup:
                        self._lambda_plateau_epochs += 1
                        if (self.lambda_src_decay_patience > 0
                                and self._lambda_plateau_epochs >= self.lambda_src_decay_patience
                                and self.lambda_src_current > self.lambda_src_min):
                            new_lambda = max(
                                self.lambda_src_min,
                                self.lambda_src_current * self.lambda_src_decay_factor
                            )
                            if new_lambda < self.lambda_src_current - 1e-8:
                                print(
                                    f"🔄 Decayed lambda_src mixing weight to {new_lambda:.4f} "
                                    f"after {self._lambda_plateau_epochs} plateau epochs."
                                )
                            elif new_lambda <= self.lambda_src_min + 1e-8:
                                print(
                                    f"🔄 lambda_src reached floor {new_lambda:.4f}; disabling source mixing."
                                )
                            else:
                                print(
                                    f"🔄 Adjusted lambda_src mixing weight to {new_lambda:.4f} "
                                    f"after {self._lambda_plateau_epochs} plateau epochs."
                                )
                            self.lambda_src_current = new_lambda
                            self._lambda_plateau_epochs = 0
                if val_improved:
                    self._lambda_plateau_epochs = 0
    
            self.lr_scheduler.step()
            if patience is not None:
                if val_improved:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"⏹ Early stopping triggered after {patience} epochs without improvement.")
                        break
    
        print("Training complete.")
        self.model.load_state_dict(best_model_wts)
        self.best_source_val_acc = float(best_eval_acc)
        if self.dataloaders.get('target_val') is not None:
            # Always prefer target-side metric when available (transfer or baseline)
            self.best_target_balanced_acc = float(best_balanced_acc)
            if self.best_target_balanced_acc > 0:
                self.best_val_acc_class = self.best_target_balanced_acc
            else:
                self.best_val_acc_class = best_common_acc if best_common_acc > 0 else best_hscore
        else:
            self.best_val_acc_class = self.best_source_val_acc
            
            self.best_target_balanced_acc = 0.0

        if self.transfer_mode and self.dataloaders.get('target_train') is not None:
            calibrate_bn(
                self.model,
                self.dataloaders['target_train'],
                self.device,
                max_batches=int(getattr(self.args, 'bn_calibration_batches', 64))
            )
            
        if self.transfer_mode and self.dataloaders.get('target_train') is not None:
            calibrate_bn(
                self.model,
                self.dataloaders['target_train'],
                self.device,
                max_batches=int(getattr(self.args, 'bn_calibration_batches', 64))
            )
        
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, "best_model.pth")
        )
        print(f"🔖  Saved best source model to {self.save_dir}/best_model.pth")
        if self.dataloaders.get('target_val') is not None:
            print(f"🏁 Final best target validation balanced accuracy: {self.best_val_acc_class:.4f}")
            print(f"📊 Final best target validation common accuracy: {best_common_acc:.4f}")
        else:
            print(f"🏁 Final best validation accuracy: {self.best_val_acc_class:.4f}")
        
        
        total_time = time.time() - self._train_start_time
        metrics = {
            "wall_time_sec": float(total_time),
            "time_to_best_sec": float(self._best_metric_time if self._best_metric_time is not None else total_time),
            "best_epoch": int(self._best_epoch),
            "transfer_mode": bool(self.transfer_mode),
            "num_target_samples": int(self.target_sample_count),
            "best_target_balanced_accuracy": float(getattr(self, 'best_target_balanced_acc', 0.0)),
            "best_target_common_accuracy": float(best_common_acc),
        }
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, "train_timing.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"⏱  Timing saved → {os.path.join(self.save_dir, 'train_timing.json')}")
        except Exception as e:
            print("⚠️ Could not write timing:", e)

        
        # Build a wrapper so evaluation always routes through the same
        # classification head used during training (SNGP or deterministic).
        bottleneck_for_eval = getattr(self, 'bottleneck_layer', None)
        if bottleneck_for_eval is None:
            bottleneck_for_eval = nn.Identity()

        head_for_eval = getattr(self, 'sngp_model', None)
        classifier_for_eval = getattr(self, 'classifier_layer', None)

        class EvalHeadWrapper(nn.Module):
            def __init__(self, backbone, bottleneck, head, classifier):
                super().__init__()
                self.backbone = backbone
                self.bottleneck = bottleneck if bottleneck is not None else nn.Identity()
                self.head = head
                self.classifier = classifier

            def forward(self, x):
                out = self.backbone(x)
                if isinstance(out, tuple):
                    model_logits = out[0]
                    feats = out[1] if len(out) > 1 else None
                else:
                    model_logits, feats = out, None

                if feats is not None:
                    feats = self.bottleneck(feats)

                    if self.head is not None and hasattr(self.head, 'forward_classifier'):
                        return self.head.forward_classifier(
                            feats,
                            update_cov=False,
                            apply_mean_field=True,
                        )

                    if self.classifier is not None:
                        return self.classifier(feats)

            # If no features are available (some backbones return logits only),
                # fall back to the backbone logits.
                return model_logits

        eval_model = EvalHeadWrapper(self.model, bottleneck_for_eval, head_for_eval, classifier_for_eval)
            
        # === Confusion matrices for this run ===
        # Battery datasets pin labels to [0,1,2]; others (e.g., CWRU) use full range
        labels_override = (
            list(range(5))
            if self.args.data_name == "Battery_inconsistent"
            else list(range(self.num_classes))
        )
        model_for_eval = eval_model  # use the wrapper when SNGP, else the model

        try:
            if self.dataloaders.get('source_val') is not None:
                _save_confusion_outputs(model_for_eval,
                                        self.dataloaders['source_val'],
                                        self.device,
                                        self.num_classes,
                                        self.save_dir,
                                        split_name="source_val",
                                        labels_override=labels_override)
            if self.dataloaders.get('target_val') is not None:
                _save_confusion_outputs(model_for_eval,
                                        self.dataloaders['target_val'],
                                        self.device,
                                        self.num_classes,
                                        self.save_dir,
                                        split_name="target_val",
                                        labels_override=labels_override)
        except Exception as e:
            print(f"⚠️ Failed to save confusion matrices: {e}")
            
        if getattr(self.args, 'method', '') == 'sngp':
            try:
                self._export_sngp_uncertainty(self.dataloaders.get('target_val'), 'target_val')
            except Exception as e:
                print(f"⚠️ Failed to export SNGP uncertainty: {e}")

        # now return the model instance *and* the target‐val accuracy (so optuna can optimize it)
        return eval_model, self.best_val_acc_class

