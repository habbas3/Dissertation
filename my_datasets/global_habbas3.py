#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:29:37 2023

@author: habbas3
"""

# settings.py

def init():
    global source_val_labels
    global source_val_predictions
    global source_val_acc
    global source_train_labels
    global source_train_predictions
    global source_train_acc
    global target_val_labels
    global target_val_predictions
    global target_val_acc
    global target_outlier_acc
    global target_common_acc
    global target_share_weight
    global amount_target
    global correct_target
    global probs_list
    global accuracy_score
    global best_hscore
    
    source_val_labels = []
    source_val_predictions = []
    source_val_acc = []
    source_train_labels = []
    source_train_predictions = []
    source_train_acc = []
    target_val_labels = []
    target_val_predictions = []
    target_val_acc = []
    target_outlier_acc = []
    target_common_acc = []
    target_share_weight = []
    amount_target = []
    correct_target = []
    probs_list = []
    accuracy_score = []
    best_hscore = []