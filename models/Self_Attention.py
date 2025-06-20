#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:03:45 2023

@author: habbas3
"""

import torch
import torch.nn as nn


class OutlierAttention(nn.Module):
    def __init__(self, in_dim, eps=1e-8):
        super(OutlierAttention, self).__init__()
        self.outlier_score = nn.Conv1d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.alpha = nn.Parameter(torch.ones(1))  # Learnable parameter
        self.eps = eps

    def forward(self, x):
        scores = torch.sigmoid(self.outlier_score(x))
        scores = scores / (scores.sum(dim=2, keepdim=True) + self.eps)  # Avoid division by zero
        enhanced_x = self.alpha * scores * x + (1 - self.alpha) * x  # Blend input and enhanced features
        
    
    # Check for NaN and replace with zeros
        if torch.isnan(enhanced_x).any():
            enhanced_x = torch.where(torch.isnan(enhanced_x), torch.zeros_like(enhanced_x), enhanced_x)
            
        return enhanced_x
        


class SelfAttention(nn.Module):
    def __init__(self, in_dim, eps=1e-8):
        super(SelfAttention, self).__init__()
        
        self.query = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        self.eps = eps
        
    def forward(self, x):
        # [Batch, Channel, Feature]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Attention score
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention.clamp(min=-1e8, max=1e8)  # Clamp to avoid extreme values
        attention = torch.nn.functional.softmax(attention, dim=2) + self.eps  # Add small epsilon
        
        out = torch.matmul(attention, v)
        out = self.gamma * out + x
        
        if torch.isnan(out).any():
            out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
        
        return out