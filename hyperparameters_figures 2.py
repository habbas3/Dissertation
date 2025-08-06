#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:38:23 2024

@author: habbas
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data extracted from the provided text
data = {
    "Transfer Task": ["[0 to 1]", "[0 to 2]", "[0 to 3]", "[1 to 0]", "[1 to 2]", "[1 to 3]", "[2 to 0]", "[2 to 1]", "[2 to 3]", "[3 to 0]", "[3 to 1]", "[3 to 2]"],
    "CNN - Self Attention": [
        {"Initial Learning Rate": 8.55E-05, "Batch Size": 32, "Hidden Size": 128, "Dropout Rate": 0.16, "Bottleneck Size": 128},
        {"Initial Learning Rate": 4.30E-03, "Batch Size": 128, "Hidden Size": 512, "Dropout Rate": 0.29, "Bottleneck Size": 128},
        {"Initial Learning Rate": 2.10E-03, "Batch Size": 64, "Hidden Size": 512, "Dropout Rate": 0.319, "Bottleneck Size": 128},
        {"Initial Learning Rate": 2.05E-05, "Batch Size": 32, "Hidden Size": 512, "Dropout Rate": 0.40, "Bottleneck Size": 128},
        {"Initial Learning Rate": 6.04E-05, "Batch Size": 32, "Hidden Size": 1024, "Dropout Rate": 0.48, "Bottleneck Size": 128},
        {"Initial Learning Rate": 6.30E-04, "Batch Size": 64, "Hidden Size": 512, "Dropout Rate": 0.25, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.05E-03, "Batch Size": 32, "Hidden Size": 1024, "Dropout Rate": 0.198, "Bottleneck Size": 128},
        {"Initial Learning Rate": 3.47E-05, "Batch Size": 64, "Hidden Size": 1024, "Dropout Rate": 0.462, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.99E-03, "Batch Size": 32, "Hidden Size": 256, "Dropout Rate": 0.224, "Bottleneck Size": 128},
        {"Initial Learning Rate": 9.96E-03, "Batch Size": 32, "Hidden Size": 1024, "Dropout Rate": 0.324, "Bottleneck Size": 128},
        {"Initial Learning Rate": 4.80E-03, "Batch Size": 256, "Hidden Size": 256, "Dropout Rate": 0.361, "Bottleneck Size": 128},
        {"Initial Learning Rate": 2.03E-04, "Batch Size": 256, "Hidden Size": 1024, "Dropout Rate": 0.131, "Bottleneck Size": 128}
    ],
    "CNN-Openmax-SA": [
        {"Initial Learning Rate": 7.60E-04, "Batch Size": 128, "Hidden Size": 256, "Dropout Rate": 0.36, "Bottleneck Size": 128},
        {"Initial Learning Rate": 7.70E-03, "Batch Size": 64, "Hidden Size": 256, "Dropout Rate": 0.49, "Bottleneck Size": 128},
        {"Initial Learning Rate": 7.90E-04, "Batch Size": 128, "Hidden Size": 512, "Dropout Rate": 0.334, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.30E-03, "Batch Size": 32, "Hidden Size": 1024, "Dropout Rate": 0.44, "Bottleneck Size": 128},
        {"Initial Learning Rate": 6.50E-04, "Batch Size": 64, "Hidden Size": 256, "Dropout Rate": 0.50, "Bottleneck Size": 128},
        {"Initial Learning Rate": 6.13E-04, "Batch Size": 64, "Hidden Size": 512, "Dropout Rate": 0.21, "Bottleneck Size": 128},
        {"Initial Learning Rate": 4.40E-04, "Batch Size": 32, "Hidden Size": 256, "Dropout Rate": 0.4895, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.67E-02, "Batch Size": 256, "Hidden Size": 512, "Dropout Rate": 0.291, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.39E-04, "Batch Size": 64, "Hidden Size": 512, "Dropout Rate": 0.474, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.14E-03, "Batch Size": 64, "Hidden Size": 1024, "Dropout Rate": 0.48, "Bottleneck Size": 128},
        {"Initial Learning Rate": 3.50E-02, "Batch Size": 256, "Hidden Size": 128, "Dropout Rate": 0.261, "Bottleneck Size": 128},
        {"Initial Learning Rate": 7.40E-03, "Batch Size": 256, "Hidden Size": 512, "Dropout Rate": 0.372, "Bottleneck Size": 128}
    ],
    "WideResnet": [
        {"Initial Learning Rate": 3.78E-03, "Batch Size": 256, "Hidden Size": 256, "Dropout Rate": 0.39, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.36E-02, "Batch Size": 128, "Hidden Size": 256, "Dropout Rate": 0.42, "Bottleneck Size": 128},
        {"Initial Learning Rate": 2.10E-03, "Batch Size": 64, "Hidden Size": 1024, "Dropout Rate": 0.111, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.10E-03, "Batch Size": 32, "Hidden Size": 1024, "Dropout Rate": 0.14, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.00E-02, "Batch Size": 64, "Hidden Size": 512, "Dropout Rate": 0.24, "Bottleneck Size": 128},
        {"Initial Learning Rate": 5.65E-03, "Batch Size": 64, "Hidden Size": 128, "Dropout Rate": 0.36, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.60E-03, "Batch Size": 64, "Hidden Size": 128, "Dropout Rate": 0.165, "Bottleneck Size": 128},
        {"Initial Learning Rate": 5.14E-03, "Batch Size": 256, "Hidden Size": 128, "Dropout Rate": 0.369, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.82E-03, "Batch Size": 64, "Hidden Size": 256, "Dropout Rate": 0.224, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.70E-02, "Batch Size": 32, "Hidden Size": 1024, "Dropout Rate": 0.299, "Bottleneck Size": 128},
        {"Initial Learning Rate": 7.01E-03, "Batch Size": 256, "Hidden Size": 1024, "Dropout Rate": 0.325, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.48E-02, "Batch Size": 256, "Hidden Size": 1024, "Dropout Rate": 0.399, "Bottleneck Size": 128}
    ],
    "WideResnet - Self Attention": [
        {"Initial Learning Rate": 8.59E-03, "Batch Size": 64, "Hidden Size": 1024, "Dropout Rate": 0.38, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.81E-02, "Batch Size": 32, "Hidden Size": 512, "Dropout Rate": 0.27, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.75E-02, "Batch Size": 32, "Hidden Size": 128, "Dropout Rate": 0.423, "Bottleneck Size": 128},
        {"Initial Learning Rate": 3.90E-03, "Batch Size": 256, "Hidden Size": 256, "Dropout Rate": 0.34, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.50E-02, "Batch Size": 64, "Hidden Size": 1024, "Dropout Rate": 0.2576, "Bottleneck Size": 128},
        {"Initial Learning Rate": 9.75E-03, "Batch Size": 32, "Hidden Size": 128, "Dropout Rate": 0.50, "Bottleneck Size": 128},
        {"Initial Learning Rate": 3.18E-02, "Batch Size": 64, "Hidden Size": 128, "Dropout Rate": 0.2576, "Bottleneck Size": 128},
        {"Initial Learning Rate": 3.02E-02, "Batch Size": 64, "Hidden Size": 512, "Dropout Rate": 0.2958, "Bottleneck Size": 128},
        {"Initial Learning Rate": 9.62E-03, "Batch Size": 32, "Hidden Size": 256, "Dropout Rate": 0.222, "Bottleneck Size": 128},
        {"Initial Learning Rate": 3.30E-03, "Batch Size": 64, "Hidden Size": 128, "Dropout Rate": 0.125, "Bottleneck Size": 128},
        {"Initial Learning Rate": 8.85E-04, "Batch Size": 256, "Hidden Size": 1024, "Dropout Rate": 0.473, "Bottleneck Size": 128},
        {"Initial Learning Rate": 6.80E-03, "Batch Size": 256, "Hidden Size": 512, "Dropout Rate": 0.254, "Bottleneck Size": 128}
        ],
    
    "WideResnet - Openmax SA": [
        {"Initial Learning Rate": 2.66E-04, "Batch Size": 64, "Hidden Size": 1024, "Dropout Rate": 0.49, "Bottleneck Size": 128},
        {"Initial Learning Rate": 3.50E-04, "Batch Size": 32, "Hidden Size": 1024, "Dropout Rate": 0.25, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.40E-04, "Batch Size": 64, "Hidden Size": 128, "Dropout Rate": 0.154, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.80E-03, "Batch Size": 32, "Hidden Size": 256, "Dropout Rate": 0.46, "Bottleneck Size": 128},
        {"Initial Learning Rate": 2.80E-03, "Batch Size": 128, "Hidden Size": 256, "Dropout Rate": 0.46, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.75E-04, "Batch Size": 64, "Hidden Size": 1024, "Dropout Rate": 0.47, "Bottleneck Size": 128},
        {"Initial Learning Rate": 9.36E-05, "Batch Size": 64, "Hidden Size": 1024, "Dropout Rate": 0.368, "Bottleneck Size": 128},
        {"Initial Learning Rate": 7.50E-03, "Batch Size": 256, "Hidden Size": 512, "Dropout Rate": 0.495, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.69E-02, "Batch Size": 64, "Hidden Size": 256, "Dropout Rate": 0.4132, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.90E-02, "Batch Size": 128, "Hidden Size": 128, "Dropout Rate": 0.281, "Bottleneck Size": 128},
        {"Initial Learning Rate": 1.45E-03, "Batch Size": 128, "Hidden Size": 256, "Dropout Rate": 0.146, "Bottleneck Size": 128},
        {"Initial Learning Rate": 3.00E-03, "Batch Size": 32, "Hidden Size": 256, "Dropout Rate": 0.377, "Bottleneck Size": 128}
    ]
    }

# Create DataFrame for each model's hyperparameters
df = pd.DataFrame(data["Transfer Task"], columns=["Transfer Task"])

# CNN - Self Attention
df_cnn_sa = pd.DataFrame(data["CNN - Self Attention"])
df_cnn_sa.columns = [f"CNN-SA - {col}" for col in df_cnn_sa.columns]

# CNN-Openmax-SA
df_cnn_openmax_sa = pd.DataFrame(data["CNN-Openmax-SA"])
df_cnn_openmax_sa.columns = [f"CNN-Openmax-SA - {col}" for col in df_cnn_openmax_sa.columns]

# WideResnet
df_wideresnet = pd.DataFrame(data["WideResnet"])
df_wideresnet.columns = [f"WideResnet - {col}" for col in df_wideresnet.columns]

# WideResnet - Self Attention
df_wideresnet_sa = pd.DataFrame(data["WideResnet - Self Attention"])
df_wideresnet_sa.columns = [f"WideResnet-SA - {col}" for col in df_wideresnet_sa.columns]

# WideResnet - Openmax SA
df_wideresnet_openmax_sa = pd.DataFrame(data["WideResnet - Openmax SA"])
df_wideresnet_openmax_sa.columns = [f"WideResnet-Openmax-SA - {col}" for col in df_wideresnet_openmax_sa.columns]

# Concatenate all dataframes horizontally
df_final = pd.concat([df, df_cnn_sa, df_cnn_openmax_sa, df_wideresnet, df_wideresnet_sa, df_wideresnet_openmax_sa], axis=1)

# Display table in a clear format for publication
pd.set_option('display.max_columns', None)  # Show all columns
print(df_final.to_markdown(index=False))  # Convert the DataFrame to a markdown table format

df_final.to_csv('hyperparamters_table.csv', index=False)
print(df_final.to_string(index=False))

performance_data = df_final.filter(regex='Accuracy Score|Transfer Task')

# Set 'Transfer Task' as the index
performance_data.set_index('Transfer Task', inplace=True)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(performance_data, annot=True, cmap='viridis', fmt='.2f')
plt.title('Heatmap of Model Performance Across Transfer Tasks')
plt.show()
