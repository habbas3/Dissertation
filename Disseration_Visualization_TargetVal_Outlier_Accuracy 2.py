#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:09:08 2024

@author: habbas
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tasks_1 = ["[0 to 1]", "[0 to 2]", "[0 to 3]", "[1 to 0]", "[1 to 2]"]
target_outlier_accuracy_1 = {
    'CNN': [41.67, 47.83, 56.25, 51.06, 42.55],
    'CNN - Self Attention': [54.17, 56.52, 43.75, 36.17, 44.68],
    'CNN-Openmax-SA': [54.17, 52.17, 64.58, 48.94, 46.81],
    'WideResnet': [45.83, 52.18, 54.17, 63.83, 44.68],
    'WideResnet - Self Attention': [62.50, 47.83, 56.25, 66.00, 53.19],
    'WideResnet - Openmax Layer - Self Attention': [45.83, 47.83, 64.59, 57.45, 59.57]
}

tasks_2 = ["[1 to 3]", "[2 to 0]", "[2 to 1]", "[2 to 3]"]
target_outlier_accuracy_2 = {
    'CNN': [39.58, 37.82, 37.35, 33.33],
    'CNN - Self Attention': [50.00, 40.34, 51.20, 39.58],
    'CNN-Openmax-SA': [54.17, 51.26, 57.23, 41.67],
    'WideResnet': [58.83, 65.55, 63.85, 35.42],
    'WideResnet - Self Attention': [56.25, 52.94, 47.00, 33.33],
    'WideResnet - Openmax Layer - Self Attention': [56.25, 56.30, 53.01, 62.50]
}

df_outlier_1 = pd.DataFrame(target_outlier_accuracy_1, index=tasks_1)
df_outlier_2 = pd.DataFrame(target_outlier_accuracy_2, index=tasks_2)


# Define the color palette to match the previous plot for 'Accuracy Score'
palette = sns.color_palette("husl", 6)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Plotting the results for Target Outlier Accuracy
plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
for i, (model, color) in enumerate(zip(df_outlier_1.columns, palette)):
    sns.lineplot(x=df_outlier_1.index, y=df_outlier_1[model], label=model, color=color, marker='o')
plt.title('Target Outlier Accuracy Model Comparison', fontsize=14)
plt.xlabel('Transfer Task', fontsize=12)
plt.ylabel('Outlier Accuracy (%)', fontsize=12)
plt.xticks(rotation=45)

# Plotting the results for Target Validation Accuracy
plt.subplot(1, 2, 2)
for i, (model, color) in enumerate(zip(df_outlier_2.columns, palette)):
    sns.lineplot(x=df_outlier_2.index, y=df_outlier_2[model], label=model, color=color, marker='o')
plt.title('Target Outlier Accuracy Model Comparison', fontsize=14)
plt.xlabel('Transfer Task', fontsize=12)
plt.ylabel('Outlier Accuracy (%)', fontsize=12)
plt.xticks(rotation=45)


###########
target_validation_accuracy_1 = {
    'CNN': [55.27, 54.32, 56.60, 57.77, 51.95],
    'CNN - Self Attention': [62.90, 58.45, 53.48, 45.48, 52.82],
    'CNN-Openmax-SA': [60.80, 57.51, 55.14, 60.30, 58.25],
    'WideResnet': [59.09, 60.24, 55.53, 48.62, 57.12],
    'WideResnet - Self Attention': [58.41, 58.90, 55.54, 58.70, 57.47],
    'WideResnet - Openmax Layer - Self Attention': [53.57, 53.18, 55.14, 52.40, 55.57]
}

target_validation_accuracy_2 = {
    'CNN': [53.67, 49.74, 48.06, 48.89],
    'CNN - Self Attention': [52.00, 52.13, 56.62, 50.79],
    'CNN-Openmax-SA': [63.62, 56.59, 58.08, 50.65],
    'WideResnet': [53.85, 59.66, 61.30, 50.76],
    'WideResnet - Self Attention': [55.19, 58.03, 53.50, 47.62],
    'WideResnet - Openmax Layer - Self Attention': [59.21, 60.00, 54.86, 64.52]
}

# Create dataframes
df_validation_1 = pd.DataFrame(target_validation_accuracy_1, index=tasks_1)
df_validation_2 = pd.DataFrame(target_validation_accuracy_2, index=tasks_2)


# Define the color palette to match the previous plot for 'Accuracy Score'
palette = sns.color_palette("husl", 6)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Plotting the results for Target Outlier Accuracy
plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
for i, (model, color) in enumerate(zip(df_validation_1.columns, palette)):
    sns.lineplot(x=df_validation_1.index, y=df_validation_1[model], label=model, color=color, marker='o')
plt.title('Target Validation Accuracy Model Comparison', fontsize=14)
plt.xlabel('Transfer Task', fontsize=12)
plt.ylabel('Outlier Accuracy (%)', fontsize=12)
plt.xticks(rotation=45)

# Plotting the results for Target Validation Accuracy
plt.subplot(1, 2, 2)
for i, (model, color) in enumerate(zip(df_validation_2.columns, palette)):
    sns.lineplot(x=df_validation_2.index, y=df_validation_2[model], label=model, color=color, marker='o')
plt.title('Target Validation Accuracy Model Comparison', fontsize=14)
plt.xlabel('Transfer Task', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.xticks(rotation=45)