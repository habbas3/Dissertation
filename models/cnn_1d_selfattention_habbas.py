#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
from torch import mm
import warnings

import numpy as np
from models.Self_Attention import SelfAttention as attention_sa
from models.Self_Attention import OutlierAttention as attention_outliers


# ----------------------------inputsize >=28-------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channels=6, out_features=10, conv_layers=None):
        super(CNN, self).__init__()
        if pretrained:
            warnings.warn("Pretrained model is not available")

        # Default convolutional layers configuration if None is specified
        if conv_layers is None:
            conv_layers = [
                {'in_channels': in_channels, 'out_channels': 16, 'kernel_size': 15},
                {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3},
                {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3},
                {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3}
            ]

        self.conv_layers = nn.ModuleList()
        for layer_config in conv_layers:
            layer = nn.Sequential(
                nn.Conv1d(**layer_config, stride=1, padding=0),
                nn.BatchNorm1d(layer_config['out_channels']),
                nn.ReLU(inplace=True)
            )
            self.conv_layers.append(layer)

        # Add MaxPool layer after the second Conv layer
        self.conv_layers.insert(2, nn.MaxPool1d(kernel_size=2, stride=2))

        # Add AdaptiveMaxPool layer after the last Conv layer
        self.conv_layers.append(nn.AdaptiveMaxPool1d(4))

        # Attention and Outlier Attention layers after the last Conv layer
        self.attention = attention_sa(conv_layers[-1]['out_channels'])
        self.outlier_attention = attention_outliers(conv_layers[-1]['out_channels'])

        self.fc = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(128 * 4, 256), n_power_iterations=1, eps=1e-12),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.output_layer = nn.Linear(256, out_features)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.attention(x)
        x = self.outlier_attention(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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