#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:48:24 2024

@author: habbas
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Since the image with data is not accessible directly, the following data is a dummy and should be replaced with actual values
# Replace this dummy data with the actual data values from the user's results to generate the figure.

# Creating a sample dataframe with dummy data
data = {
    'Transfer Task': [0, 1, 2, 3],
    'CNN': [85.48, 79.03, 82.76, 71.82],
    'CNN - Self Attention': [80.65, 74.19, 75.86, 76.36],
    'CNN-Openmax-SA': [83.87, 82.26, 68.97, 83.63],
    'WideResnet': [69.35, 79.03, 69.00, 79.09],
    'WideResnet - Self Attention': [82.26, 87.10, 69.00, 82.72],
    'WideResnet - Openmax Layer - Self Attention': [61.29, 61.29, 69.00, 63.64]
}

df = pd.DataFrame(data)
df['Transfer Task'] = [0, 1, 2, 3]


# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Increase the size of the figure
plt.figure(figsize=(14, 8))

# Plotting
ax = sns.lineplot(data=df.set_index('Transfer Task'), markers=True, dashes=False)


# Adding titles and labels
ax.set_title('Model Performance Comparison', fontsize=16)
ax.set_xlabel('Transfer Task', fontsize=14)
ax.set_ylabel('Accuracy Score (SNGP)', fontsize=14)
ax.legend(title='Models', loc='upper left', bbox_to_anchor=(1, 1))


# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()