#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:37:14 2024

@author: habbas
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import MultiHeadAttention as attention
from models.Self_Attention import SelfAttention as attention_sa
from models.Self_Attention import OutlierAttention as attention_outliers
from scipy.stats import weibull_min
from sklearn.metrics.pairwise import euclidean_distances
from torch.autograd import Variable
import numpy as np
import warnings

# CNN with self-attention and OpenMax layer
class CNN_OpenMax(nn.Module):
    def __init__(self, args, num_classes):
        super(CNN_OpenMax, self).__init__()
        self.cnn = CNN(pretrained=False, in_channel=args.input_channels)
        # self.attn = attention_sa(128)  # Self-attention layer after the CNN
        # self.attn_outliers = attention_outliers(128) 
        self.fc_input_size = 256
        self.fc = nn.utils.parametrizations.spectral_norm(nn.Linear(self.fc_input_size, num_classes))
        self.openmax = OpenMaxLayer(num_features=self.fc.in_features, num_classes=num_classes)

    def forward(self, x, *args, **kwargs):
        # Pass the input through the CNN
        x = self.cnn(x)
    
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        features = x  # Save features for OpenMax layer
    
        # Pass through the fully connected layer
        logits = self.fc(x)
    
        if not self.training and self.openmax is not None and self.openmax.are_weibull_models_initialized():
            # Only apply OpenMax post-processing during inference
            return self.openmax(logits, features)
        
        return logits  # Always return logits for training

    
    def _calculate_fc_input_size(self, input_size, nChannels, n):
        # Mock input for size calculation
        mock_input = torch.zeros(1, 1, input_size)
        out = self.conv1(mock_input)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.avg_pool1d(out, out.size(2))
        return out.view(-1).size(0)

    def output_num(self):
        return self.fc.out_features
    
class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=7, out_channel=10):
        super(CNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")
            
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=3, padding=7),  # Ensure same output size
            nn.BatchNorm1d(16, eps=1e-05),  # Added small epsilon to BatchNorm
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # Ensure same output size
            nn.BatchNorm1d(32, eps=1e-05),  # Added small epsilon to BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # Ensure same output size
            nn.BatchNorm1d(64, eps=1e-05),  # Added small epsilon to BatchNorm
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # Ensure same output size
            nn.BatchNorm1d(128, eps=1e-05),  # Added small epsilon to BatchNorm
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))  # Ensuring output size is [batch_size, 128, 4]

        self.attn = attention_sa(128)  # Assuming attention_sa is defined elsewhere
        self.attn_outliers = attention_outliers(128)  # Assuming attention_outliers is defined elsewhere

        self.layer5 = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(128 * 4, 256), n_power_iterations=1, eps=1e-9),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)) 

            
        # #Add attention layer

        
    def forward(self, x):
        x = self.layer1(x)
        if torch.isnan(x).any():
            raise ValueError("NaN values detected after layer1")
        x = self.layer2(x)
        if torch.isnan(x).any():
            raise ValueError("NaN values detected after layer2")    
        x = self.layer3(x)
        if torch.isnan(x).any():
            raise ValueError("NaN values detected after layer3")
        x = self.layer4(x)
        if torch.isnan(x).any():
            raise ValueError("NaN values detected after layer4")
    
        # Apply self-attention after convolutional layers
        x = self.attn(x)
        x = self.attn_outliers(x)
        if torch.isnan(x).any():
            raise ValueError("NaN values detected after attention layer")
    
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        if torch.isnan(x).any():
            raise ValueError("NaN values detected after layer5")
        return x
        
class OpenMaxLayer(nn.Module):
    def __init__(self, num_features, num_classes, tail_size=70, alpha=10):
        super(OpenMaxLayer, self).__init__()
        self.num_classes = num_classes
        # self.num_features = num_features
        self.tail_size = tail_size
        self.alpha = alpha
        self.mean_vecs = torch.zeros(num_classes, num_features)
        self.weibull_models = {}
        
    def are_weibull_models_initialized(self):
        return all(i in self.weibull_models for i in range(self.num_classes))
    
    def fit_weibull(self, feature_vectors, labels):
        """
        Fit the Weibull models for each class and feature based on the training set features.
        """
        # Convert feature_vectors to PyTorch tensor if it's a NumPy array
        def fit_weibull(self, feature_vectors, labels):
            if isinstance(feature_vectors, np.ndarray):
                feature_vectors = torch.tensor(feature_vectors, device=self.mean_vecs.device)
            
            for i in range(self.num_classes):
                class_feature_vectors = feature_vectors[labels == i]
                mean_vector = torch.mean(class_feature_vectors, dim=0)
                self.mean_vecs[i] = mean_vector
        
                distances = torch.norm(class_feature_vectors - self.mean_vecs[i], dim=1)
                distances, _ = torch.sort(distances)
                tails = distances[-self.tail_size:]
        
                try:
                    self.weibull_models[i] = weibull_min.fit(tails.cpu().numpy())
                except Exception as e:
                    print(f"Error fitting Weibull model for class {i}: {e}")
  
    
    def compute_openmax(self, logits, features):
        """
        Compute the OpenMax probabilities given the logits and the feature vectors.
        """
        # Compute the euclidean distance from the mean activation vector
        features_processed = features.view(features.size(0), -1)  # Flatten features if necessary
        mean_vecs_processed = self.mean_vecs.view(1, self.num_classes, -1)  # Reshape for broadcasting
        dists = torch.norm(features_processed[:, None, :] - mean_vecs_processed, dim=2).flatten()

        
        # Compute the OpenMax score
        scores = torch.zeros((features.shape[0], self.num_classes))
        for i in range(self.num_classes):
            c = dists[i].detach().numpy()
            w_params = self.weibull_models[i]
            w_score = weibull_min.cdf(c, *w_params)
            scores[:, i] = logits[:, i] * (1 - torch.tensor(w_score) ** self.alpha)
        
        # Convert scores to probabilities
        exp_scores = torch.exp(scores)
        probs = exp_scores / torch.sum(exp_scores, axis=1, keepdims=True)
        return probs
    
    def forward(self, logits, features):
        """
        Forward pass for the OpenMax layer.
        """
        openmax_probs = self.compute_openmax(logits, features)
        if torch.isnan(openmax_probs).any():
            raise ValueError("NaN values detected after openmax layer")
        return openmax_probs