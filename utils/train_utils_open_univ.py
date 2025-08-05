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


# def adjust_bottleneck_layer(bottleneck_layer, features, bottleneck_num):
#     # Extract the size of the features produced by the CNN
#     feature_size = features.size(1)
    
#     # Dynamically create a new bottleneck layer with the correct input size
#     adjusted_bottleneck_layer = nn.Sequential(
#         nn.Linear(feature_size, bottleneck_num),
#         nn.ReLU(inplace=True),
#         nn.Dropout()
#     )
    
#     return adjusted_bottleneck_layer


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


        # Load the datasets
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

        else:
            if isinstance(args.transfer_task[0], str):
                args.transfer_task = eval("".join(args.transfer_task))
            self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets['target_val'], self.num_classes = \
                Dataset(args.data_dir, args.transfer_task, args.inconsistent, args.normlizetype).data_split(transfer_learning=True)

        if args.data_name == 'Battery_inconsistent':
            # Already returned as DataLoaders from load_battery_dataset
            self.dataloaders = {
                'source_train': self.datasets['source_train'],
                'source_val': self.datasets['source_val'],
                'target_train': self.datasets['target_train'],
                'target_val': self.datasets['target_val']
            }
        else:
            self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                               shuffle=(True if x.split('_')[1] == 'train' else False),
                                                               num_workers=args.num_workers,
                                                               pin_memory=(True if self.device == 'cuda' else False),
                                                               drop_last=(True if args.last_batch and x.split('_')[1] == 'train' else False),
                                                               generator=g)
                                for x in ['source_train', 'source_val', 'target_train', 'target_val']}

        first_batch = next(iter(self.dataloaders['source_train']))
        input_tensor, _ = first_batch
        args.input_channels = input_tensor.shape[1]  # Automatically infer input channels from data
        args.input_size = input_tensor.shape[-1]
        print("🧪 Input shape before CNN:", input_tensor.shape)
        


        
        # Define the model
        self.max_iter = len(self.dataloaders['source_train']) * args.max_epoch
        if args.model_name in ["cnn_openmax", "cnn_features_1d_sa", "cnn_features_1d","WideResNet", "WideResNet_sa", "WideResNet_mh", "WideResNet_edited"]:
            if args.model_name == "WideResNet":
                self.model = WideResNet(args.layers, args.widen_factor, args.droprate, self.num_classes)
            elif args.model_name == "WideResNet_sa":
                self.model = WideResNet_sa(args.layers, args.widen_factor, args.droprate, self.num_classes)
            elif args.model_name == "WideResNet_edited":
                self.model = WideResNet_edited(args.layers, args.widen_factor, args.droprate, self.num_classes)
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
                print(f"🔁 Loading pretrained model from: {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location=self.device)
                incompatible = self.model.load_state_dict(state_dict, strict=False)
                if incompatible.missing_keys:
                    print(f"⚠️ Missing keys when loading pretrained model: {incompatible.missing_keys}")
                if incompatible.unexpected_keys:
                    print(f"⚠️ Unexpected keys when loading pretrained model: {incompatible.unexpected_keys}")
                # self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
            
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
                print(f"🔁 Loading pretrained model from: {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location=self.device)
                incompatible = self.model.load_state_dict(state_dict, strict=False)
                if incompatible.missing_keys:
                    print(f"⚠️ Missing keys when loading pretrained model: {incompatible.missing_keys}")
                if incompatible.unexpected_keys:
                    print(f"⚠️ Unexpected keys when loading pretrained model: {incompatible.unexpected_keys}")
                # self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
                
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
                # tmp  = nlp_layers.RandomFeatureGaussianProcess(units=self.num_classes,
                #                                normalize_input=False,
                #                                scale_random_features=True,
                #                                gp_cov_momentum=-1)
                
                # self.classifier_layer = tmp
                
                # self.classifier_layer = gp_layer(args.bottleneck_num, self.num_classes)
                self.classifier_layer = nn.Linear(args.bottleneck_num, self.num_classes)

                self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=args.bottleneck_num,
                                                                        hidden_size=args.hidden_size,
                                                                        max_iter=self.max_iter,
                                                                        trade_off_adversarial=args.trade_off_adversarial,
                                                                        lam_adversarial=args.lam_adversarial
                                                                        )
                self.AdversarialNet_auxiliary = getattr(models, 'AdversarialNet_auxiliary')(in_feature=args.bottleneck_num,
                                                                                            hidden_size=args.hidden_size)
            else:
                self.classifier_layer = nn.Linear(self.model.output_num(), self.num_classes)
                self.AdversarialNet = getattr(models, 'AdversarialNet')(in_feature=self.model.output_num(),
                                                                        hidden_size=args.hidden_size,
                                                                        max_iter=self.max_iter,
                                                                        trade_off_adversarial=args.trade_off_adversarial,
                                                                        lam_adversarial=args.lam_adversarial
                                                                        )
                self.AdversarialNet_auxiliary = getattr(models, 'AdversarialNet_auxiliary')(
                    in_feature=self.model.output_num(),
                    hidden_size=args.hidden_size)
        if args.bottleneck:
            self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)
        else:
            self.model_all = nn.Sequential(self.model, self.classifier_layer)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.bottleneck:
                self.bottleneck_layer = torch.nn.DataParallel(self.bottleneck_layer)
            if args.inconsistent == 'UAN':
                self.AdversarialNet = torch.nn.DataParallel(self.AdversarialNet)
                self.AdversarialNet_auxiliary = torch.nn.DataParallel(self.AdversarialNet)
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
        if args.inconsistent == "OSBP":
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]
        else:
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet_auxiliary.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet_auxiliary.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
                
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
        if args.inconsistent == 'UAN':
            self.AdversarialNet.to(self.device)
            self.AdversarialNet_auxiliary.to(self.device)
        self.classifier_layer.to(self.device)
        self.sngp_model.to(self.device)

        if args.inconsistent == "OSBP":
            self.inconsistent_loss = nn.BCELoss()

        # self.criterion = nn.CrossEntropyLoss()
        classification_label = args.classification_label
        all_labels = df[classification_label].values
        class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        # self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        self.criterion = nn.CrossEntropyLoss()
        

    def train(self):
        best_source_val_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_hscore = 0.0
        best_common_acc = 0.0
    
        for epoch in range(self.args.max_epoch):
            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} ----- Epoch {epoch + 1}/{self.args.max_epoch} -----")
            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if self.target_train_loader is not None:
                target_iter = cycle(self.dataloaders['target_train'])
            else:
                target_iter = None
    
            for phase in ['source_train', 'source_val', 'target_val']:
                if self.dataloaders.get(phase) is None:
                    continue
                
                self.model.train() if phase == 'source_train' else self.model.eval()
    
                running_loss = 0.0
                running_corrects = 0
                running_total = 0
                preds_all, labels_all, logits_all, inputs_all = [], [], [], []
                
                
    
                for step, batch in enumerate(self.dataloaders[phase]):
                    if phase == 'source_train' and target_iter is not None:
                        source_inputs, labels = batch
                        target_inputs, _ = next(target_iter)
                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                    else:
                        inputs, labels = batch
                        
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
    
                    if inputs.dim() == 2:
                        inputs = inputs.unsqueeze(1)
                    if inputs.shape[1] != self.args.input_channels and \
                       inputs.shape[-1] == self.args.input_channels:
                        inputs = inputs.permute(0, 2, 1)
    
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'source_train'):
                        backbone_out = self.model(inputs)
                        if isinstance(backbone_out, tuple):
                            model_logits, features = backbone_out[0], backbone_out[1]
                            domain_out = backbone_out[2] if len(backbone_out) > 2 else None
                        else:
                            model_logits = backbone_out
                            features = None
                            domain_out = None
    
                        if features is not None and self.args.bottleneck:
                            features = self.bottleneck_layer(features)

                        if features is not None:
                            logits = self.sngp_model.forward_classifier(features)
                        else:
                            logits = model_logits
    
                        if step % 50 == 0 and phase == 'source_train':
                            with torch.no_grad():
                                probs = F.softmax(logits, dim=1)
                                max_probs, preds = torch.max(probs, 1)
                                pred_counts = torch.bincount(preds, minlength=self.num_classes)
                                print(f"🧪 [Epoch {epoch:02d} | Step {step:04d}] Pred dist: {pred_counts.tolist()}")
                                print(f"🧪 [Epoch {epoch:02d} | Step {step:04d}] Avg Max Softmax Prob: {max_probs.mean().item():.4f}")
    
                        if phase != 'source_train':
                            loss = self.criterion(logits, labels)
                            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
                        else:
                            batch_size = inputs.size(0)
                            source_size = labels.size(0)
                            target_size = batch_size - source_size
    
                            logits = torch.where(torch.isnan(logits), torch.full_like(logits, 0.22815), logits)
                            logits_source = logits[:source_size]
                            classifier_loss = self.criterion(logits_source, labels)
    
                            if self.args.inconsistent != 'OSBP' and domain_out is not None:
                                features_source = features[:source_size]
                                features_target = features[source_size:]
                                logits_target = logits[source_size:]
                                domain_out_source = domain_out[:source_size]
                                domain_out_target = domain_out[source_size:]
    
                                domain_prob_source = torch.sigmoid(self.AdversarialNet(features_source))
                                domain_prob_target = torch.sigmoid(self.AdversarialNet(features_target))
                                domain_prob_source_aux = torch.sigmoid(self.AdversarialNet_auxiliary(features_source.detach()))
                                domain_prob_target_aux = torch.sigmoid(self.AdversarialNet_auxiliary(features_target.detach()))
    
                                source_share_weight = get_source_share_weight(
                                    domain_prob_source_aux,
                                    logits_source,
                                    domain_temperature=1.0,
                                    class_temperature=10.0
                                )
                                source_share_weight = normalize_weight(source_share_weight)
    
                                if target_size > 0:
                                    domain_temperature = getattr(self.args, 'domain_temperature', 1.0)
                                    class_temperature = getattr(self.args, 'class_temperature', 10.0)
    
                                    target_share_weight = get_target_share_weight(
                                        domain_out_target.detach(),
                                        logits_target.detach(),
                                        domain_temperature=domain_temperature,
                                        class_temperature=class_temperature
                                    )
                                    target_share_weight = normalize_weight(target_share_weight)
                                    mask = (target_share_weight > 0.1).float()
                                    target_share_weight = target_share_weight * mask
    
                                    if target_share_weight.numel() == 0 or torch.isnan(target_share_weight).all():
                                        print("⚠️ Empty/NaN target_share_weight. Skipping adversarial loss.")
                                        inconsistent_loss = classifier_loss
                                    else:
                                        target_share_weight = torch.clamp(
                                            torch.nan_to_num(target_share_weight, nan=0.5), 0.01, 0.99
                                        )
    
                                        tmp_src = nn.BCELoss(reduction='none')(domain_prob_source, torch.ones_like(domain_prob_source))
                                        tmp_tgt = nn.BCELoss(reduction='none')(domain_prob_target, torch.zeros_like(domain_prob_target))
                                        adv_loss = torch.mean(source_share_weight * tmp_src) + torch.mean(target_share_weight * tmp_tgt)
    
                                        aux_loss_src = nn.BCELoss()(domain_prob_source_aux, torch.ones_like(domain_prob_source_aux))
                                        aux_loss_tgt = nn.BCELoss()(domain_prob_target_aux, torch.zeros_like(domain_prob_target_aux))
    
                                        inconsistent_loss = adv_loss + aux_loss_src + aux_loss_tgt
                                else:
                                    inconsistent_loss = classifier_loss
    
                            else:
                                inconsistent_loss = classifier_loss  # fallback if OSBP or no domain_out
    
                            loss = classifier_loss + inconsistent_loss
    
                    if phase == 'source_train':
                        loss.backward()
                        self.optimizer.step()
    
                    running_loss += loss.item() * inputs.size(0)
                    print(f"🔍 logits shape: {logits.shape}, labels shape: {labels.shape}")
                    _, preds = torch.max(logits, 1)
                    if phase == 'source_train' and target_iter is not None:
                        preds = preds[:source_size]
                        logits_for_log = logits[:source_size]
                        inputs_for_log = inputs[:source_size]
                    else:
                        logits_for_log = logits
                        inputs_for_log = inputs
                        
                    if preds.numel() == 0:
                        continue
                    
                    running_corrects += torch.sum(preds == labels.data)
                    running_total += labels.size(0)
    
                    preds_all.extend(preds.detach().cpu().numpy())
                    labels_all.extend(labels.detach().cpu().numpy())
                    logits_all.append(logits_for_log.detach().cpu())
                    inputs_all.append(inputs_for_log.detach().cpu())
    
                epoch_loss = running_loss / running_total if running_total > 0 else 0.0
                epoch_acc = running_corrects.double() / running_total if running_total > 0 else 0.0
                print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} Epoch: {epoch} {phase}-Loss: {epoch_loss:.4f} {phase}-Acc: {epoch_acc:.4f}")
    
                if phase == 'source_val' and epoch_acc > best_source_val_acc:
                    best_source_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    print("✓ Best model updated based on source_val accuracy.")
    
                if phase == 'target_val':
                    preds_np = np.array(preds_all)
                    labels_np = np.array(labels_all)
                    known_mask = labels_np < self.num_classes
                    out_mask = labels_np >= self.num_classes

                    common_acc = (
                        np.mean(preds_np[known_mask] == labels_np[known_mask])
                        if known_mask.any() else 0.0
                    )
                    outlier_acc = (
                        np.mean(preds_np[out_mask] == labels_np[out_mask])
                        if out_mask.any() else 0.0
                    )
                    if not out_mask.any() and known_mask.any():
                        hscore = common_acc
                    elif not known_mask.any() and out_mask.any():
                        hscore = outlier_acc
                    else:
                        denom = common_acc + outlier_acc
                        hscore = (2 * common_acc * outlier_acc / denom) if denom > 0 else 0.0
                    
                    if common_acc > best_common_acc:
                        best_common_acc = common_acc

                    print(
                        f"{datetime.now().strftime('%m-%d %H:%M:%S')} Epoch: {epoch} {phase}-hscore: {hscore:.4f}"
                    )
                    if hscore > best_hscore:
                        best_hscore = hscore
                        print("✓ Best target hscore updated.")
    
            self.lr_scheduler.step()
    
        print("Training complete.")
        self.model.load_state_dict(best_model_wts)
        self.best_source_val_acc = float(best_source_val_acc)
        if self.dataloaders.get('target_val') is None:
            self.best_val_acc_class = self.best_source_val_acc
        else:
            self.best_val_acc_class = best_common_acc if best_common_acc > 0 else best_hscore
        
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

