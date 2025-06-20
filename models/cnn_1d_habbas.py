#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
from torch import mm
import warnings

import numpy as np
from models.Self_Attention import SelfAttention as attention_sa


# ----------------------------inputsize >=28-------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channels=6, out_features=10, conv_layers=None, fc_layers=None):
        super(CNN, self).__init__()
        if pretrained:
            warnings.warn("Pretrained model is not available")

        # Default convolutional layers configuration if None is specified
        if conv_layers is None:
            conv_layers = [
                {'in_channels': in_channels, 'out_channels': 16, 'kernel_size': 15, 'stride': 1, 'padding': 0},
                {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
                {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0},
                {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0}
            ]

        # Default fully connected layers configuration if None is specified
        if fc_layers is None:
            fc_layers = [
                {'in_features': 128 * 4, 'out_features': 256}
            ]

        self.conv_layers = nn.ModuleList()
        for layer_config in conv_layers:
            layer = nn.Sequential(
                nn.Conv1d(**layer_config),
                nn.BatchNorm1d(layer_config['out_channels']),
                nn.ReLU(inplace=True)
            )
            self.conv_layers.append(layer)

        # Add MaxPool layer after the second Conv layer
        self.conv_layers.insert(2, nn.MaxPool1d(kernel_size=2, stride=2))

        # Add AdaptiveMaxPool layer after the last Conv layer
        self.conv_layers.append(nn.AdaptiveMaxPool1d(4))

        self.fc_layers = nn.ModuleList()
        for i, layer_config in enumerate(fc_layers):
            if i == 0:
                layer = nn.Sequential(
                    nn.utils.parametrizations.spectral_norm(nn.Linear(**layer_config), n_power_iterations=1, eps=1e-12),
                    nn.ReLU(inplace=True),
                    nn.Dropout()
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(**layer_config),
                    nn.ReLU(inplace=True),
                    nn.Dropout()
                )
            self.fc_layers.append(layer)

        # Output layer
        self.output_layer = nn.Linear(fc_layers[-1]['out_features'], out_features)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.view(x.size(0), -1)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        x = self.output_layer(x)
        return x


# convnet without the last layer
class cnn_features(nn.Module):
    def __init__(self, pretrained=False, in_channels=6):
        super(cnn_features, self).__init__()
        self.model_cnn = CNN(pretrained=pretrained, in_channels=in_channels)
        self.__in_features = 256

    def forward(self, x):
        x = self.model_cnn(x)
        return x

    def output_num(self):
        return self.__in_features