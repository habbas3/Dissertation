#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings
import numpy as np
from models.Self_Attention import SelfAttention as attention_sa
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # compress to (B, 64, 1)
        
        self.flatten = nn.Flatten()  # shape becomes (B, 64)
        self.fc1 = nn.Linear(64, 256)  # explicitly define fc1
        self.relu = nn.ReLU()
        
        num_classes = 3
        self.classifier = nn.Linear(256, num_classes)  # 3 classes in your case
        
        self.domain_classifier = nn.Linear(256, 2)  # use same hidden dim

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.classifier(x)
        if hasattr(self, "domain_classifier"):
            domain_out = self.domain_classifier(x)
        else:
            domain_out = None

        return logits, x, domain_out  # optionally return features for domain classifier

    def forward_domain(self, features):
        return self.domain_classifier(features)




class cnn_features(nn.Module):
    def __init__(self, pretrained=False):
        super(cnn_features, self).__init__()
        self.model_cnn = CNN(pretrained, in_channel=1)
        self.__in_features = 256

    def forward(self, x):
        return self.model_cnn(x)

    def output_num(self):
        return self.__in_features
