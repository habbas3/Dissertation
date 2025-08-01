#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 19:52:21 2023

@author: habbas3
"""

import torch
import math
import copy
from torch import nn
from typing import Optional
from torch import Tensor
from models.spectral_normalization import spectral_norm


def mean_field_logits(logits, covariance_matrix_diag, ridge_penalty=1e-3):
    """Applies mean field approximation to logits given uncertainty estimates."""
    variance = covariance_matrix_diag
    scaled_logits = logits / torch.sqrt(1 + variance + ridge_penalty)
    return scaled_logits


def BertLinear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def RandomFeatureLinear(i_dim, o_dim, bias=True, require_grad=False):
    m = nn.Linear(i_dim, o_dim, bias)
    # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/bert_sngp.py
    nn.init.normal_(m.weight, mean=0.0, std=0.05)
    # freeze weight
    m.weight.requires_grad = require_grad
    if bias:
        nn.init.uniform_(m.bias, a=0.0, b=2. * math.pi)
        # freeze bias
        m.bias.requires_grad = require_grad
    return m



class SNGP(nn.Module):
    def __init__(self, backbone, bottleneck_num, num_inducing, num_classes, 
                 gp_kernel_scale=1.0, gp_output_bias=0., layer_norm_eps=1e-12,
                 n_power_iterations=1, spec_norm_bound=0.95, scale_random_features=True,
                 normalize_input=True, gp_cov_momentum=0.999, gp_cov_ridge_penalty=1e-3,
                 epochs=100, device='cpu'):
        super(SNGP, self).__init__()
        self.__in_features = 256
        self.backbone = backbone
        self.final_epochs = epochs - 1
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_momentum = gp_cov_momentum

        self.pooled_output_dim = bottleneck_num
        self.last_pooled_layer = spectral_norm(BertLinear(bottleneck_num, self.pooled_output_dim),
                                               n_power_iterations=n_power_iterations, norm_bound=spec_norm_bound)

        self.gp_input_scale = 1. / math.sqrt(gp_kernel_scale)
        self.gp_feature_scale = math.sqrt(2. / float(num_inducing))
        self.gp_output_bias = gp_output_bias
        self.scale_random_features = scale_random_features
        self.normalize_input = normalize_input

        self._gp_input_normalize_layer = torch.nn.LayerNorm(bottleneck_num, eps=layer_norm_eps)
        self._gp_output_layer = nn.Linear(num_inducing, num_classes, bias=True)
        # bert gp_output_bias_trainable is false
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L69
        self._gp_output_bias = torch.tensor([self.gp_output_bias] * num_classes).to(device)
        self._random_feature = RandomFeatureLinear(self.pooled_output_dim, num_inducing)

        # Laplace Random Feature Covariance
        # Posterior precision matrix for the GP's random feature coefficients.
        self.initial_precision_matrix = (self.gp_cov_ridge_penalty * torch.eye(num_inducing).to(device))
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)
        

# class SNGP(nn.Module):
#     def __init__(self, backbone,
#                  hidden_size=256,
#                  gp_kernel_scale=1.0,
#                  num_inducing=1024,
#                  gp_output_bias=0.,
#                  layer_norm_eps=1e-12,
#                  n_power_iterations=1,
#                  spec_norm_bound=0.95,
#                  scale_random_features=True,
#                  normalize_input=True,
#                  gp_cov_momentum=0.999,
#                  gp_cov_ridge_penalty=1e-3,
#                  epochs=100,
#                  num_classes=10,
#                  device='cpu'):
#         super(SNGP, self).__init__()
#         self.__in_features = 256
#         self.backbone = backbone
#         self.final_epochs = epochs - 1
#         self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
#         self.gp_cov_momentum = gp_cov_momentum

#         self.pooled_output_dim = hidden_size
#         self.last_pooled_layer = spectral_norm(BertLinear(hidden_size, self.pooled_output_dim),
#                                                n_power_iterations=n_power_iterations, norm_bound=spec_norm_bound)

#         self.gp_input_scale = 1. / math.sqrt(gp_kernel_scale)
#         self.gp_feature_scale = math.sqrt(2. / float(num_inducing))
#         self.gp_output_bias = gp_output_bias
#         self.scale_random_features = scale_random_features
#         self.normalize_input = normalize_input

#         self._gp_input_normalize_layer = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         self._gp_output_layer = nn.Linear(num_inducing, num_classes, bias=False)
#         # bert gp_output_bias_trainable is false
#         # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L69
#         self._gp_output_bias = torch.tensor([self.gp_output_bias] * num_classes).to(device)
#         self._random_feature = RandomFeatureLinear(self.pooled_output_dim, num_inducing)

#         # Laplace Random Feature Covariance
#         # Posterior precision matrix for the GP's random feature coefficients.
#         self.initial_precision_matrix = (self.gp_cov_ridge_penalty * torch.eye(num_inducing).to(device))
#         self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)

    def extract_bert_features(self, latent_feature):
        # https://github.com/google/uncertainty-baselines/blob/b3686f75a10b1990c09b8eb589657090b8837d2c/uncertainty_baselines/models/bert_sngp.py#L336
        # Extract BERT encoder output (i.e., the CLS token).
        first_token_tensors = latent_feature[:, 0, :]
        cls_output = self.last_pooled_layer(first_token_tensors)
        return cls_output
    
    def forward_classifier(self, features):
        """
        Bypass backbone entirely and return ONLY the GP head logits tensor.
        """
        if features is None:
            raise ValueError("Features input to forward_classifier is None. Check your model forward pass.")

        out = self.gp_layer(features)
        # gp_layer often returns (gp_features, gp_logits)
        if isinstance(out, tuple) and len(out) == 2:
            gp_feat, gp_logits = out
            return gp_logits
        # otherwise it's already just the logits
        return out



    def gp_layer(self, gp_inputs, update_cov=True):
        # Supports lengthscale for custom random feature layer by directly
        # rescaling the input.
        if self.normalize_input:
            gp_inputs = self._gp_input_normalize_layer(gp_inputs)

        gp_feature = self._random_feature(gp_inputs)
        # cosine
        gp_feature = torch.cos(gp_feature)

        if self.scale_random_features:
            gp_feature = gp_feature * self.gp_input_scale

        # Computes posterior center (i.e., MAP estimate) and variance.
        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias

        if update_cov:
            # update precision matrix
            self.update_cov(gp_feature)
        return gp_feature, gp_output

    def reset_cov(self):
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)

    def update_cov(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L346
        batch_size = gp_feature.size()[0]
        precision_matrix_minibatch = torch.matmul(gp_feature.t(), gp_feature)
        # Updates the population-wise precision matrix.
        if self.gp_cov_momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision
            # matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                    self.gp_cov_momentum * self.precision_matrix +
                    (1. - self.gp_cov_momentum) * precision_matrix_minibatch)
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch
        #self.precision_matrix.weight = precision_matrix_new
        self.precision_matrix = torch.nn.Parameter(precision_matrix_new, requires_grad=False)

    def compute_predictive_covariance(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L403
        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)

        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = torch.matmul(feature_cov_matrix, gp_feature.t()) * self.gp_cov_ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)
        return gp_cov_matrix

    def forward(self, input_ids):
        # Ensure input shape is [B, C, T]
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)  # [B, T] → [B, 1, T]
        elif input_ids.dim() == 3 and input_ids.shape[1] != 1:
            input_ids = input_ids.transpose(1, 2)  # [B, T, C] → [B, C, T]
    
        latent_feature = self.backbone(input_ids)  # [B, C, T]
        latent_feature = self.bottleneck(latent_feature)  # <-- No crash now
        logits = self.classifier(latent_feature)
    
        if self.random_feature is not None:
            logits = mean_field_logits(
                logits,
                self.random_feature(latent_feature),
                self.ridge_penalty
            )
        return logits




    
    def output_num(self):
        return self.__in_features

class Deterministic(nn.Module):
    def __init__(self, backbone, bottleneck_num, num_classes, layer_norm_eps=1e-12, normalize_input=True):
        super(Deterministic, self).__init__()
        self.backbone = backbone
        self.normalize_input = normalize_input

        # If input normalization is required, add a LayerNorm
        if self.normalize_input:
            self.input_normalize_layer = torch.nn.LayerNorm(bottleneck_num, eps=layer_norm_eps)

        self.fc = nn.Linear(bottleneck_num, num_classes)  # Use nn.Linear instead of BertLinear unless you have a specific reason to use BertLinear

    def forward(self, input_ids):
        input_ids = input_ids.unsqueeze(1)  # Add channel dimension (B, 1, L)
        features = self.backbone(input_ids)  # Get features from CNN or WRN
    
        if isinstance(features, tuple):
            features = features[0]  # Unwrap if needed
    
        features = features.view(features.size(0), -1)  # Flatten for linear layer
    
        if self.normalize_input:
            features = self.input_normalize_layer(features)
    
        logits = self.fc(features)
        return logits
    
    def forward_classifier(self, features):
        if self.normalize_input:
            features = self.input_normalize_layer(features)
        return self.fc(features)
    
    
    

    
# class Deterministic(nn.Module):
#     def __init__(self, backbone,
#                  hidden_size=768,
#                  num_classes=3):
#         super(Deterministic, self).__init__()
#         self.backbone = backbone
#         self.fc = BertLinear(hidden_size, num_classes)

#     def forward(self, input_ids, token_type_ids: Optional[Tensor] = None,
#                 attention_mask: Optional[Tensor] = None, return_gp_cov: bool = False,
#                 update_cov: bool = False):
#         latent_feature, _ = self.backbone(input_ids, token_type_ids, attention_mask)
#         cls_output = self.fc(latent_feature[:, 0, :])
#         if return_gp_cov:
#             return cls_output, None
#         return cls_output


if __name__ == "__main__":
    import numpy as np
    from bert import BertModel, Config

    lm_config = Config()
    bert = BertModel(lm_config)
    bert.load_pretrain_huggingface(torch.load("../ckpt/bert-base-uncased-pytorch_model.bin"))
    sngp_model = Deterministic(bert)

    input_ids = torch.tensor(np.ones([5, 36])).to(torch.long)
    output, cov = sngp_model(input_ids, input_ids, input_ids, return_gp_cov=True)
    print(output, cov)
    print("Load SNGP Model")