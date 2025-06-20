#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:36:47 2023

@author: habbas3
"""
#Based on the Attention Layer shared by Latif Cander
import torch
import torch.nn as nn

class AttentionCNN(nn.Module):
    def __init__(self):
        super(AttentionCNN,self).__init__()
        self.L = 57
        self.D = 128
        self.K = 1
        
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid(),
        )   
