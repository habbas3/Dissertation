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
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
import random
import torch
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
        
                source_train, source_val, target_train, target_val, label_names, df = load_battery_dataset(
                    csv_path=self.args.csv,
                    source_cathodes=self.args.source_cathode,
                    target_cathodes=self.args.target_cathode,
                    classification_label=self.args.classification_label,
                    batch_size=self.args.batch_size,
                    sequence_length=self.args.sequence_length,
                )
        
                self.datasets['source_train'] = source_train
                self.datasets['source_val'] = source_val
                self.datasets['target_train'] = target_train
                self.datasets['target_val'] = target_val
                self.label_names = label_names
                self.df = df
                self.num_classes = len(label_names)
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
        
        # Target sample count (safe)
        self.target_sample_count = 0
        tgt_dl = self.dataloaders.get('target_train')
        if tgt_dl is not None:
            ds_len_tgt = _dataset_len(tgt_dl)
            self.target_sample_count = ds_len_tgt
            if 0 < ds_len_tgt < 100:
                args.lr = min(args.lr, 1e-4)
                logging.info(f"Reducing learning rate to {args.lr} for {ds_len_tgt} target samples")
        
        # Determine transfer mode only if target_train actually has data
        self.transfer_mode = (_dataset_len(tgt_dl) > 0 and getattr(self.args, 'pretrained_model_path', None))
        
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


        # Invert the model and define the loss
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if self.AdversarialNet is not None:
            self.AdversarialNet.to(self.device)
            self.AdversarialNet_auxiliary.to(self.device)
        self.classifier_layer.to(self.device)
        self.sngp_model.to(self.device)

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
        patience = getattr(self.args, 'early_stop_patience', None)
        epochs_no_improve = 0
        
        self.warmup_epochs = int(getattr(self.args, "warmup_epochs", 3))
        self._head_module = _find_last_linear(self.model)
    
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
                
                self.model.train() if phase.endswith('train') else self.model.eval()
    
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

                        logits = self.sngp_model.forward_classifier(features) if features is not None else model_logits

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
                        if self.transfer_mode and phase == 'target_train' and self.dataloaders.get('source_train') is not None:
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
                                lam_src = getattr(self.args, 'lambda_src', 1.0)
                                total_loss = loss_src * lam_src if total_loss is None else total_loss + lam_src * loss_src
                            # If no valid src labels for this step, just skip source loss for this batch.
                        # <<< END NEW >>>
                        
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
                epoch_acc = running_corrects.double() / running_total if running_total > 0 else 0.0
                print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} Epoch: {epoch} {phase}-Loss: {epoch_loss:.4f} {phase}-Acc: {epoch_acc:.4f}")
    
                if phase in ['source_val', 'target_val'] and epoch_acc > best_eval_acc:
                    best_eval_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    print("✓ Best model updated based on validation accuracy.")
                    val_improved = True
    
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
                
                    print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} Epoch: {epoch} {phase}-common: {common_acc:.4f} outlier: {outlier_acc:.4f} hscore: {hscore:.4f}")
                
                    # Also update BEST MODEL on improved hscore/common so we save the right model for transfer
                    if improved_h or improved_common:
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        val_improved = True
                        print("✓ Best target model updated based on hscore/common.")
    
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
            self.best_val_acc_class = best_common_acc if best_common_acc > 0 else best_hscore
        else:
            self.best_val_acc_class = self.best_source_val_acc
        
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, "best_model.pth")
        )
        print(f"🔖  Saved best source model to {self.save_dir}/best_model.pth")
        print(f"🏁 Final best target validation accuracy: {self.best_val_acc_class:.4f}")
        
        # Build a wrapper so evaluation can call model(x) directly when SNGP
        # Build a wrapper so evaluation can call model(x) directly when SNGP
        if self.args.method == 'sngp':
            class SNGPWrapper(nn.Module):
                def __init__(self, backbone, bottleneck, head):
                    super().__init__()
                    self.backbone = backbone
                    self.bottleneck = bottleneck
                    self.head = head

                def forward(self, x):
                    """Run backbone and route extracted features through the GP head.

                    Some backbones return a tuple ``(logits, features, *extra)`` while
                    others may return only logits.  Handle both cases so evaluation
                    doesn't crash if features are missing (e.g. due to a different
                    model implementation).  If features are unavailable, fall back to
                    the backbone's logits.
                    """
                    out = self.backbone(x)
                    if isinstance(out, tuple):
                        model_logits = out[0]
                        feats = out[1] if len(out) > 1 else None
                    else:
                        model_logits, feats = out, None

                    if feats is not None:
                        if self.bottleneck is not None and not isinstance(self.bottleneck, nn.Identity):
                            feats = self.bottleneck(feats)
                        return self.head.forward_classifier(feats)

                    # Fallback: backbone provided no features, so return its logits
                    return model_logits

            eval_model = SNGPWrapper(self.model, self.bottleneck_layer if self.args.bottleneck else nn.Identity(), self.sngp_model)
        else:
            eval_model = self.model

        # now return the model instance *and* the target‐val accuracy (so optuna can optimize it)
        return eval_model, self.best_val_acc_class

