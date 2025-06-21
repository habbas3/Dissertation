#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 21:55:22 2024

@author: habbas
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tasks_1 = ["[0 to 1]", "[0 to 2]", "[0 to 3]", "[1 to 0]", "[1 to 2]"]
tasks_2 = ["[1 to 3]", "[2 to 0]", "[2 to 1]", "[2 to 3]"]
data_sngp_1 = {
    'CNN': [85.48, 79.03, 82.76, 71.82, 48.42],
    'CNN - Self Attention': [80.65, 74.19, 75.86, 76.36, 50.53],
    'CNN-Openmax-SA': [83.87, 82.26, 68.97, 83.63, 50.53],
    'WideResnet': [69.35, 79.03, 69.00, 79.09, 50.53],
    'WideResnet - Self Attention': [82.26, 87.10, 69.00, 82.72, 50.53],
    'WideResnet - Openmax Layer - Self Attention': [61.29, 61.29, 69.00, 63.64, 40.00]
}
y_min = 30;
y_max = 100

data_sngp_2 = {
    'CNN': [50.00, 47.67, 40.00, 50.00],
    'CNN - Self Attention': [48.96, 50.00, 40.00, 46.88],
    'CNN-Openmax-SA': [50.00, 50.00, 40.00, 37.50],
    'WideResnet': [47.92, 48.84, 40.00, 50.00],
    'WideResnet - Self Attention': [46.88, 50.00, 40.00, 50.00],
    'WideResnet - Openmax Layer - Self Attention': [43.75, 50.00, 40.00, 50.00]
    }

data_sngp_1 = pd.DataFrame(data_sngp_1, index=tasks_1)
data_sngp_2 = pd.DataFrame(data_sngp_2, index=tasks_2)



# Define the color palette for consistency with the previous plots
palette = sns.color_palette("husl", 6)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Plotting the results for Target Outlier Accuracy
plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
for i, (model, color) in enumerate(zip(data_sngp_1.columns, palette)):
    sns.lineplot(x=data_sngp_1.index, y=data_sngp_1[model], label=model, color=color, marker='o')
plt.title('Accuracy Score (SNGP) Model Comparison', fontsize=14)
plt.xlabel('Transfer Task', fontsize=12)
plt.ylabel('Accuracy Score (SNGP) (%)', fontsize=12)

plt.set_ylim([y_min, y_max])

plt.xticks(rotation=45)

# Plotting the results for Target Validation Accuracy
plt.subplot(1, 2, 2)
for i, (model, color) in enumerate(zip(data_sngp_2.columns, palette)):
    sns.lineplot(x=data_sngp_2.index, y=data_sngp_2[model], label=model, color=color, marker='o')
plt.title('Accuracy Score (SNGP) Model Comparison', fontsize=14)
plt.xlabel('Transfer Task', fontsize=12)
plt.ylabel('Accuracy Score (SNGP) (%)', fontsize=12)
plt.set_ylim([y_min, y_max])
plt.xticks(rotation=45)