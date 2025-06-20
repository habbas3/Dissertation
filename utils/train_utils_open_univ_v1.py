#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
from torch import optim
import my_datasets.global_habbas3
import models
from models.cnn_1d_original import CNN
# from models.cnn_1d_habbas import CNN
# from models.cnn_1d_habbas_hyperparstudy import CNN as cnn_features_1d_hyperparstudy
from models.wideresnet_habbas import WideResNet
from models.wideresnet_self_attention_habbas import WideResNet_sa
from models.wideresnet_multihead_attention import WideResNet_mh
from models.wideresnet_edited_habbas import WideResNet_edited
from models.cnn_1d_selfattention_habbas import cnn_features as cnn_features_1d_sa
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
    def __init__(self, args, save_dir):
        self.args = args
        self.method = args.method
        self.save_dir = save_dir
        
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
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        if args.data_name == 'Battery_inconsistent':
            self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets['target_val'], self.num_classes = \
                Dataset(args.data_dir, args.source_cathode, args.target_cathode, args.inconsistent, args.normlizetype).data_split()
        else:
            if isinstance(args.transfer_task[0], str):
                args.transfer_task = eval("".join(args.transfer_task))
            self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets['target_val'], self.num_classes = \
                Dataset(args.data_dir, args.transfer_task, args.inconsistent, args.normlizetype).data_split(transfer_learning=True)

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(True if args.last_batch and x.split('_')[1] == 'train' else False),
                                                           generator = g)
                            for x in ['source_train', 'source_val', 'target_train', 'target_val']}
        first_batch = next(iter(self.dataloaders['source_train']))
        input_size = first_batch[0].shape  # Assuming first_batch[0] contains the input data
        args.input_size = input_size[2]
        # print("Input size:", input_size[2])
        
        # Define the model
        self.max_iter = len(self.dataloaders['source_train']) * args.max_epoch
        if args.model_name in ["cnn_openmax","WideResNet", "WideResNet_sa", "WideResNet_mh", "WideResNet_edited"]:
            if args.model_name == "WideResNet":
                self.model = WideResNet(args.layers, args.widen_factor, args.droprate, self.num_classes)
            elif args.model_name == "WideResNet_sa":
                # Assuming WideResNetSA is your WideResNet model with self-attention
                self.model = WideResNet_sa(args.layers, args.widen_factor, args.droprate, self.num_classes)
            # elif args.model_name == "WideResNet_mh":
            #     self.model = WideResNet_mh(args, self.num_classes)
            elif args.model_name == "WideResNet_edited":
                self.model = WideResNet_edited(args.layers, args.widen_factor, args.droprate, self.num_classes)
            elif args.model_name == "cnn_openmax":
                self.model = cnn_openmax(args, self.num_classes)
            elif args.model_name == "cnn_features_1d":
                self.model = CNN(args, self.num_classes)
            elif args.model_name == "cnn_features_1d_sa":
                self.model = cnn_features_1d_sa(args, self.num_classes)
            # elif args.model_name == "cnn_features_1d_hyperparstudy":
            #     self.model = CNN(args, self.num_classes)
                
        else:
            self.model = getattr(models, args.model_name)(args.pretrained)
        output_features = self.model.output_num()

        self.bottleneck_layer = nn.Sequential(nn.Linear(output_features, args.bottleneck_num),
                                              nn.ReLU(inplace=True), nn.Dropout())
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
                self.classifier_layer = nn.Linear(self.model.output_num(), self.num_classes)
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

        if args.inconsistent == "OSBP":
            self.inconsistent_loss = nn.BCELoss()

        self.criterion = nn.CrossEntropyLoss()
        
        # Add SNGP HABBAS3
        self.backbone = self.model
        if args.method == 'sngp':
            self.sngp_model = sngp(self.backbone,
                                    bottleneck_num=args.bottleneck_num,
                                    num_classes=self.num_classes,
                                    num_inducing=args.gp_hidden_dim,
                                    n_power_iterations=args.n_power_iterations,
                                    spec_norm_bound=args.spectral_norm_bound,
                                    device="cuda" if self.device == 'gpu' else 'cpu')
            # self.sngp_model = sngp(self.backbone,
            #                         hidden_size=args.hidden_size,
            #                         num_classes=self.num_classes,
            #                         num_inducing=args.gp_hidden_dim,
            #                         n_power_iterations=args.n_power_iterations,
            #                         spec_norm_bound=args.spectral_norm_bound,
            #                         device="cuda" if self.device == 'gpu' else 'cpu')
        else:
            self.sngp_model = deterministic(self.backbone,
                                            bottleneck_num=args.bottleneck_num,
                                            num_classes=self.num_classes)
            
            # self.sngp_model = deterministic(self.backbone,
            #                                 hidden_size=args.hidden_size,
            #                                 num_classes=self.num_classes)

    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_hscore = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()
        best_source_val_acc = 0.0
        best_source_train_acc = 0.0
        validation_accuracy = 0
        acc = 0
        

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            iter_target = iter(self.dataloaders['target_train'])
            len_target_loader = len(self.dataloaders['target_train'])
            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                if phase == 'target_val':
                    counters = [AccuracyCounter() for x in range(self.num_classes + 1)]

                # Set model to train mode or test mode
                if phase == 'source_train':
                    # Add SNGP HABBAS3
                    # self.model.train()
                    self.sngp_model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    if args.inconsistent=="UAN":
                        self.AdversarialNet.train()
                        self.AdversarialNet_auxiliary.train()
                    #Add SNGP HABBAS3
                    # self.classifier_layer.train()
                    self.sngp_model.train()
                else:
                    # Add SNGP HABBAS3
                    # self.model.eval()
                    self.sngp_model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    if args.inconsistent=="UAN":
                        self.AdversarialNet.eval()
                        self.AdversarialNet_auxiliary.eval()
                    #Add SNGP HABBAS3
                    # self.classifier_layer.eval()
                    self.sngp_model.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if phase != 'source_train':
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        source_inputs = inputs
                        target_inputs, _ = next(iter_target)
                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    if (step + 1) % len_target_loader == 0:
                        iter_target = iter(self.dataloaders['target_train'])
                        
                    

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # forward
                        features = self.model(inputs)
                        # print("Features shape before bottleneck:", features.shape)

                        if args.bottleneck:
                            # adjusted_bottleneck_layer = adjust_bottleneck_layer(self.bottleneck_layer, features, args.bottleneck_num)
                            # features = adjusted_bottleneck_layer(features)
                            features = self.bottleneck_layer(features)
                        #Add SNGP HABBAS3
                        # outputs = self.classifier_layer(features)
                        self.backbone = self.model
                        if torch.isnan(self.sngp_model(features)).any() or torch.isnan(self.sngp_model(features)).any():
                            print("NaN values detected in outputs")
                        outputs = self.sngp_model(features)
                        if torch.isnan(outputs).any() or torch.isnan(outputs).any():
                            print("NaN values detected in outputs")
                        if phase != 'source_train':
                            logits = outputs
                            if not (phase == 'target_val' and args.inconsistent == "UAN"):
                                loss = self.criterion(logits, labels)
                                if torch.isnan(loss).any():
                                    print("NaN values detected in loss, replacing with zero")
                                    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
                            # Add SNGP HABBAS3
                            # acc = accuracy_score(to_numpy(labels), to_numpy(torch.argmax(logits, dim=-1))) * 100
                        else:
                            logits = outputs.narrow(0, 0, labels.size(0))
                            # outputs.narrow()
                            logits = torch.where(torch.isnan(logits), torch.full_like(logits, 2.2815e-01), logits)
                            classifier_loss = self.criterion(logits, labels)
                            # Add SNGP HABBAS3
                            # acc = accuracy_score(to_numpy(labels), to_numpy(torch.argmax(logits, dim=-1))) * 100

                            if args.inconsistent == 'OSBP':
                                output_t = self.classifier_layer(
                                    features.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)), adaption=True)

                                output_t_prob_unk = F.softmax(output_t, dim=1)[:, -1]
                                # print(output_t_prob_unk)
                                inconsistent_loss = self.inconsistent_loss(output_t_prob_unk,
                                                                                                   torch.tensor([
                                                                                                                    args.th] * args.batch_size).to(
                                                                                                       self.device))  #
                            else:
                                domain_prob_source = self.AdversarialNet.forward(
                                    features.narrow(0, 0, labels.size(0)))
                                domain_prob_target = self.AdversarialNet.forward(
                                    features.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)))

                                domain_prob_source_auxiliary = self.AdversarialNet_auxiliary.forward(
                                    features.narrow(0, 0, labels.size(0)).detach())
                                domain_prob_target_auxiliary = self.AdversarialNet_auxiliary.forward(
                                    features.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)).detach())

                                source_share_weight = get_source_share_weight(
                                    domain_prob_source_auxiliary, outputs.narrow(0, 0, labels.size(0)), domain_temperature=1.0,
                                    class_temperature=10.0)
                                source_share_weight = normalize_weight(source_share_weight)
                                target_share_weight = get_target_share_weight(
                                    domain_prob_target_auxiliary,
                                    outputs.narrow(0, labels.size(0), inputs.size(0) - labels.size(0)),
                                    domain_temperature=1.0,
                                    class_temperature=1.0)

                                target_share_weight = normalize_weight(target_share_weight)
                                target_share_weight = torch.where(torch.isnan(target_share_weight), torch.full_like(target_share_weight, 0.5), target_share_weight)
                                adv_loss = torch.zeros(1, 1).to(self.device)
                                adv_loss_auxiliary = torch.zeros(1, 1).to(self.device)
                                torch.autograd.set_detect_anomaly(True)
                                    
                                domain_prob_source = torch.sigmoid(self.AdversarialNet.forward(features.narrow(0, 0, labels.size(0))))
                                # print("Domain prob source after sigmoid:", domain_prob_source)
                                domain_prob_source = torch.where(torch.isnan(domain_prob_source), torch.full_like(domain_prob_source, 0.5), domain_prob_source)
                                domain_prob_source = torch.where(torch.isinf(domain_prob_source), torch.full_like(domain_prob_source, 1), domain_prob_source)

                               
                                # Ensure no NaN or Inf values
                                assert not torch.isnan(domain_prob_source).any() and not torch.isinf(domain_prob_source).any()

                                tmp = nn.BCELoss(reduction='none')(domain_prob_source, torch.ones_like(domain_prob_source))
                                adv_loss = torch.mean(source_share_weight * tmp, dim=0, keepdim=True)


                                
                                domain_prob_target = torch.sigmoid(domain_prob_target)
                                domain_prob_target = torch.where(torch.isnan(domain_prob_target), torch.full_like(domain_prob_target, 0.5), domain_prob_target)
                                domain_prob_target = torch.where(torch.isinf(domain_prob_target), torch.full_like(domain_prob_target, 1), domain_prob_target)

                                tmp = nn.BCELoss(reduction='none')(
                                    domain_prob_target,
                                    torch.zeros_like(domain_prob_target))
                                adv_loss = torch.mean(target_share_weight * tmp, dim=0, keepdim=True)
                                
                                
                                domain_prob_source_auxiliary = torch.sigmoid(domain_prob_source_auxiliary)
                                domain_prob_source_auxiliary = torch.where(torch.isnan(domain_prob_source_auxiliary), torch.full_like(domain_prob_source_auxiliary, 0.5), domain_prob_source_auxiliary)
                                domain_prob_source_auxiliary = torch.where(torch.isinf(domain_prob_source_auxiliary), torch.full_like(domain_prob_source_auxiliary, 1), domain_prob_source_auxiliary)
                                
                                
                                
                                adv_loss_auxiliary += nn.BCELoss()(domain_prob_source_auxiliary,
                                                                   torch.ones_like(
                                                                       domain_prob_source_auxiliary))
                                
                                domain_prob_target_auxiliary = torch.sigmoid(domain_prob_target_auxiliary)
                                domain_prob_target_auxiliary = torch.where(torch.isnan(domain_prob_target_auxiliary), torch.full_like(domain_prob_target_auxiliary, 0.5), domain_prob_target_auxiliary)
                                domain_prob_target_auxiliary = torch.where(torch.isinf(domain_prob_target_auxiliary), torch.full_like(domain_prob_target_auxiliary, 1), domain_prob_target_auxiliary)
                                adv_loss_auxiliary += nn.BCELoss()(domain_prob_target_auxiliary,
                                                                   torch.zeros_like(
                                                                       domain_prob_target_auxiliary))
                                inconsistent_loss = adv_loss + adv_loss_auxiliary
                            loss = classifier_loss + inconsistent_loss
                            if torch.isnan(loss).any():
                                print("NaN values detected in loss, replacing with zero")
                                loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

                        if phase == 'target_val' and args.inconsistent == "OSBP":
                            loss_temp = loss.item() * labels.size(0)
                            epoch_loss += loss_temp
                            epoch_length += labels.size(0)
                            for (each_predict, each_label) in zip(logits, labels.cpu()):
                                    counters[each_label].Ntotal += 1.0

                                    each_pred_id = np.argmax(each_predict.cpu())
                                    if each_pred_id == each_label:
                                        counters[each_label].Ncorrect += 1.0
                        elif phase == 'target_val' and args.inconsistent == "UAN":
                            for (each_predict, each_label,each_target_share_weight) in zip(logits, labels.cpu(), target_share_weight):
                                    if each_label < self.num_classes:
                                        counters[each_label].Ntotal += 1.0
                                        each_pred_id = np.argmax(each_predict.cpu())
                                        #print(each_target_share_weight)
                                        if not (each_target_share_weight[0] < args.th) and each_pred_id == each_label:
                                            counters[each_label].Ncorrect += 1.0
                                    else:
                                        counters[-1].Ntotal += 1.0
                                        if each_target_share_weight[0] < args.th:
                                            counters[-1].Ncorrect += 1.0
                            acc = accuracy_score(to_numpy(labels), to_numpy(torch.argmax(logits, dim=-1))) * 100
                            
                        else:
                            pred = logits.argmax(dim=1)
                            correct = torch.eq(pred, labels).float().sum().item()
                            loss_temp = loss.item() * labels.size(0)
                            epoch_loss += loss_temp
                            epoch_acc += correct
                            epoch_length += labels.size(0)

                        # Calculate the training information
                        if phase == 'source_train':
                            if not torch.isnan(loss):
                                self.optimizer.zero_grad()
                                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                def check_grad_hook(grad):
                                    if torch.isnan(grad).any():
                                        print("NaN gradient detected!")
                                
                                for param in self.model.parameters():
                                    param.register_hook(check_grad_hook)
                                    
                                if torch.isnan(loss).any():
                                    print("NaN value detected in loss!")
                                
                                # Check if domain_prob_source has NaN values
                                if torch.isnan(domain_prob_source).any():
                                    print("NaN value detected in domain_prob_source:", domain_prob_source)
                                
                                # Check if domain_prob_target has NaN values
                                if torch.isnan(domain_prob_target).any():
                                    print("NaN value detected in domain_prob_target:", domain_prob_target)

                                if torch.isnan(loss).any():
                                    print("NaN detected in loss before backward, pausing for debug.")
                                    # You can use a breakpoint here if you're using a debugger, or use input() to manually pause
                                    input("Press Enter to continue after inspection...")
                                
                                # Inspect model parameters for NaNs before backward pass
                                for name, param in self.model.named_parameters():
                                    if torch.isnan(param).any():
                                        print(f"NaN detected in model parameters: {name} before backward, pausing for debug.")
                                        # Pause or breakpoint for inspection
                                        input(f"Inspect parameter {name}. Press Enter to continue...")

                                # Execute backward pass
                                loss.backward()
                                
                                
                                # Inspect gradients for NaNs after backward pass
                                for name, param in self.model.named_parameters():
                                    if param.grad is not None and torch.isnan(param.grad).any():
                                        print(f"NaN detected in gradient: {name} after backward, pausing for debug.")
                                        # Pause or breakpoint for inspection
                                        input(f"Inspect gradient of {name}. Press Enter to continue...")
                                # with torch.autograd.detect_anomaly():
                                #     loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                self.optimizer.step()
                            else:
                                print("Skipping update due to NaN loss")
                            # backward
                            # self.optimizer.zero_grad()
                            # loss.backward()
                            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            # self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += labels.size(0)
                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(labels), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1
                            if epoch == args.max_epoch - 1 and args.model_name == "WideResNet_edited":
                                if not self.model.openmax.are_weibull_models_initialized():
                                    activation_vectors, labels_vector = self.collect_activation_vectors()
                                    self.model.openmax.fit_weibull(activation_vectors, labels_vector)
                            
                                # Save your model here if needed
                                # It's important to save the state of the OpenMax layer along with the model
                                model_state_dict = self.model.state_dict()
                                openmax_state_dict = self.model.openmax.state_dict()
                                torch.save({
                                    'model_state_dict': model_state_dict,
                                    'openmax_state_dict': openmax_state_dict
                                }, 'model_with_openmax.pth')
                    
                    if phase == 'source_val':
                        # Calculate accuracy for validation phase
                        correct = torch.eq(torch.argmax(logits, dim=1), labels).float().sum().item()
                        epoch_acc += correct
                        epoch_length += labels.size(0)
                if phase == 'source_val':
                    # Calculate validation accuracy for the current epoch
                    validation_accuracy = epoch_acc / epoch_length
                    logging.info(f'Validation Accuracy: {validation_accuracy:.4f}')

                
                if phase == 'target_val':
                    correct = [x.Ncorrect for x in counters]
                    amount = [x.Ntotal for x in counters]
                    common_acc = np.sum(correct[0:-1]) / np.sum(amount[0:-1])
                    outlier_acc = correct[-1] / amount[-1]
                    acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
                    acc_class = torch.ones(1, 1) * np.mean(acc_tests)
                    acc_class = acc_class[0][0]
                    acc_all = np.sum(correct[0:]) / np.sum(amount[0:])
                    hscore = 2 * common_acc * outlier_acc / (common_acc + outlier_acc)
                    if args.inconsistent == "OSBP":
                        epoch_loss = epoch_loss / epoch_length
                        logging.info(
                            'Epoch: {} {}-Loss: {:.4f} {}-common_acc: {:.4f} outlier_acc: {:.4f} acc_class: {:.4f} acc_all: {:.4f} hscore: {:.4f}, Cost {:.1f} sec'.format(
                                epoch, phase, epoch_loss, phase, common_acc, outlier_acc, acc_class, acc_all, hscore, time.time() - epoch_start
                            ))
                    else:
                        logging.info(
                            'Epoch: {} {}-common_acc: {:.4f} outlier_acc: {:.4f} acc_class: {:.4f} acc_all: {:.4f} hscore: {:.4f}, Cost {:.1f} sec'.format(
                                epoch, phase, common_acc, outlier_acc, acc_class, acc_all, hscore, time.time() - epoch_start
                            ))
                    # save the checkpoint for other learning
                    model_state_dic = self.model_all.state_dict()
                    # save the best model according to the val accuracy

                    if hscore > best_hscore:
                        best_hscore = hscore
                        global_habbas3.target_val_labels = labels
                        global_habbas3.target_val_predictions = logits
                        global_habbas3.target_val_acc = (best_hscore)
                        global_habbas3.target_outlier_acc = outlier_acc
                        global_habbas3.target_common_acc =  common_acc
                        global_habbas3.target_share_weight = target_share_weight
                        global_habbas3.correct_target = correct
                        global_habbas3.amount_target = amount
                        global_habbas3.probs_list = torch.softmax(logits, dim=-1)
                        global_habbas3.accuracy_score = acc
                        global_habbas3.best_hscore = hscore
                        print("best_hscore so far", hscore)
                        print("sngp accuracy score so far", acc)
                        
                        logging.info(
                    "save best model_hscore epoch {}, common_acc: {:.4f} outlier_acc: {:.4f} acc_class: {:.4f} acc_all: {:.4f} sngp_accuracyScore: {:.4f} best_hscore: {:.4f},".format(
                                epoch, common_acc, outlier_acc, acc_class, acc_all, acc, best_hscore))
                        torch.save(model_state_dic, os.path.join(self.save_dir,
                                                                 '{}-{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}.pth'.format(
                                                                     epoch, common_acc, outlier_acc,acc_class, acc_all, acc, best_hscore)))
                    if epoch > args.max_epoch - 2:
                        logging.info(
                    "save last model epoch {}, common_acc: {:.4f} outlier_acc: {:.4f} acc_class: {:.4f} acc_all: {:.4f} sngp_accuracyScore: {:.4f} hscore: {:.4f}".format(
                                epoch, common_acc, outlier_acc, acc_class, acc_all, acc, hscore))
                        torch.save(model_state_dic, os.path.join(self.save_dir,
                                                                 '{}-{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}.pth'.format(
                                                                     epoch, common_acc, outlier_acc,acc_class, acc_all,acc,hscore)))
                    # Print the train and val information via each epoch
                else:
                    epoch_loss = epoch_loss / epoch_length
                    epoch_acc = epoch_acc / epoch_length

                    logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                        epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                    ))
                    
                if phase == 'source_train':
                    if epoch_acc > best_source_train_acc:
                        best_source_train_acc = epoch_acc
                        global_habbas3.source_train_labels = (labels)
                        global_habbas3.source_train_predictions = (pred)
                        global_habbas3.source_train_acc = (best_source_train_acc)
                        
                        
                if phase == 'source_val':
                    if (epoch_acc > best_source_val_acc):
                        best_source_val_acc = epoch_acc
                        global_habbas3.source_val_labels = labels
                        global_habbas3.source_val_predictions = pred
                        global_habbas3.source_val_acc = best_source_val_acc

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
                        
                if phase == 'source_val':
                    if (epoch_acc > best_acc or epoch > args.max_epoch-2) and (epoch > args.middle_epoch-1):
                        best_acc = epoch_acc
                        global_habbas3.source_val_labels = labels
                        global_habbas3.source_val_predictions = pred
                        global_habbas3.source_val_acc = best_acc
                        

        return best_hscore #validation_accuracy











