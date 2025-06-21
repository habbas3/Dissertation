import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the tasks and their results
tasks1 = ["[0 to 1]", "[0 to 2]", "[0 to 3]", "[1 to 0]", "[1 to 2]", "[1 to 3]"]
tasks2 = ["[2 to 0]", "[2 to 1]", "[2 to 3]", "[3 to 0]", "[3 to 1]", "[3 to 2]"]

target_common_accuracy = {
    'CNN': [64.46, 73.05, 58.23, 61.78, 64.58, 37.50, 53.68, 44.21, 75.56, 68.18, 80.00, 79.83],
    'CNN - Self Attention': [61.45, 54.14, 60.34, 71.20, 56.25, 68.75, 60.00, 53.68, 53.66, 64.10, 75.56, 68.09],
    'CNN-Openmax-SA': [67.47, 54.29, 48.95, 77.45, 58.33, 52.08, 58.95, 57.89, 42.22, 59.10, 77.78, 61.70],
    'WideResnet': [65.66, 63.69, 59.49, 60.73, 68.75, 52.08, 72.63, 46.32, 64.44, 76.92, 82.22, 91.49],
    'WideResnet - Self Attention': [64.03, 61.06, 41.78, 49.84, 75.68, 51.16, 50.53, 60.00, 60.98, 63.64, 51.11, 70.21],
    'WideResnet - Openmax Layer - Self Attention': [63.25, 54.49, 49.37, 60.00, 47.92, 69.56, 32.63, 51.58, 60.00, 83.33, 54.62, 68.07]
}

target_outlier_accuracy = {
    'CNN': [50.00, 47.83, 54.17, 51.06, 46.81, 47.92, 54.62, 66.87, 34.04, 39.13, 40.91, 34.78],
    'CNN - Self Attention': [66.67, 65.22, 45.83, 50.00, 57.44, 52.08, 53.78, 62.05, 67.44, 59.52, 55.56, 88.89],
    'CNN-Openmax-SA': [50.00, 59.09, 64.58, 51.06, 61.70, 50.00, 57.14, 58.43, 72.34, 60.87, 66.67, 77.78],
    'WideResnet': [54.17, 47.83, 50.00, 48.94, 55.32, 68.75, 43.70, 53.01, 48.94, 42.86, 66.67, 55.56],
    'WideResnet - Self Attention': [56.52, 68.42, 54.17, 70.21, 40.91, 50.00, 54.62, 49.40, 53.49, 58.70, 77.78, 88.89],
    'WideResnet - Openmax Layer - Self Attention': [66.67, 56.52, 68.42, 61.11, 78.72, 46.81, 8.82, 54.82, 61.70, 46.81, 65.22, 52.17]
}


accuracy_score = {
    'CNN': [75.81, 72.58, 68.97, 84.78, 48.39, 18.75, 36.36, 40.00, 37.50, 41.94, 85.71, 85.71],
    'CNN - Self Attention': [83.33, 87.10, 75.86, 92.86, 51.62, 37.50, 36.36, 40.00, 40.63, 41.94, 78.87, 83.80],
    'CNN-Openmax-SA': [72.58, 79.03, 79.31, 92.82, 51.62, 28.13, 31.82, 40.00, 37.50, 41.94, 83.81, 83.80],
    'WideResnet': [81.58, 79.00, 68.97, 92.84, 51.62, 37.50, 36.36, 40.00, 40.63, 41.94, 83.81, 83.80],
    'WideResnet - Self Attention': [85.48, 80.00, 55.17, 76.05, 51.62, 34.38, 36.36, 40.00, 40.63, 41.94, 66.90, 83.80],
    'WideResnet - Openmax Layer - Self Attention': [85.48, 83.33, 68.97, 92.85, 50.53, 37.50, 22.73, 40.00, 40.63, 50.53, 85.71, 85.71]
}

h_score = {
    'CNN': [56.31, 55.49, 53.86, 55.91, 54.27, 42.07, 54.15, 53.23, 46.94, 49.72, 54.14, 48.45],
    'CNN - Self Attention': [63.95, 59.16, 52.09, 58.75, 56.84, 59.27, 56.72, 57.56, 59.77, 61.73, 64.03, 77.11],
    'CNN-Openmax-SA': [57.45, 56.59, 55.69, 61.56, 54.84, 51.02, 58.03, 58.16, 53.32, 59.97, 71.80, 68.81],
    'WideResnet': [59.36, 54.63, 54.34, 54.20, 61.31, 59.27, 54.57, 49.44, 55.63, 55.04, 73.63, 69.13],
    'WideResnet - Self Attention': [60.04, 64.53, 47.17, 51.64, 53.10, 50.57, 52.49, 54.19, 56.99, 61.07, 61.69, 78.45],
    'WideResnet - Openmax Layer - Self Attention': [64.91, 55.49, 54.34, 60.55, 59.57, 55.96, 41.98, 53.15, 60.84, 59.95, 59.45, 59.07]
}

sngp_accuracy_score = {
    'CNN': [70.00, 69.84, 70.26, 75.63, 56.44, 42.51, 56.84, 47.74, 49.10, 54.21, 68.84, 66.94],
    'CNN - Self Attention': [75.82, 77.92, 71.76, 75.22, 58.37, 54.49, 57.69, 49.19, 57.35, 54.54, 69.86, 75.86],
    'CNN-Openmax-SA': [72.30, 73.79, 75.38, 76.47, 59.42, 48.62, 55.97, 49.39, 58.05, 58.68, 74.09, 73.09],
    'WideResnet': [76.10, 70.09, 65.08, 74.02, 59.86, 54.49, 51.98, 46.48, 59.86, 55.54, 74.70, 73.20],
    'WideResnet - Self Attention': [77.68, 76.32, 52.11, 72.21, 57.13, 50.55, 56.28, 48.06, 60.32, 62.11, 65.09, 76.31],
    'WideResnet - Openmax Layer - Self Attention': [79.30, 75.44, 62.25, 76.14, 58.92, 53.39, 45.45, 47.72, 61.60, 64.60, 70.61, 70.48]
}

# Function to create and display the plots
# Function to create and display the plots for the first half of the tasks
# Define markers for each model
markers = ['o', 's', 'D', 'X', '^', 'P']

def plot_results_first_half(data, title, ylabel):
    plt.figure(figsize=(14, 8))
    palette = sns.color_palette("husl", 6)
    sns.set_style("whitegrid")
    for model, color, marker in zip(data.keys(), palette, markers):
        sns.lineplot(x=tasks1, y=data[model][:6], label=model, marker=marker, color=color, markersize=10)
    plt.title(title, fontsize=16)
    plt.xlabel('Transfer Task', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.ylim(15, 95)
    plt.xticks(rotation=45)
    plt.legend(title='Models')
    plt.tight_layout()
    plt.show()

# Function to create and display the plots for the second half of the tasks
def plot_results_second_half(data, title, ylabel):
    plt.figure(figsize=(14, 8))
    palette = sns.color_palette("husl", 6)
    sns.set_style("whitegrid")
    for model, color, marker in zip(data.keys(), palette, markers):
        sns.lineplot(x=tasks2, y=data[model][6:], label=model, marker=marker, color=color, markersize=10)
    plt.title(title, fontsize=16)
    plt.xlabel('Transfer Task', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.ylim(15, 95)
    plt.xticks(rotation=45)
    plt.legend(title='Models')
    plt.tight_layout()
    plt.show()




# Plotting the results for the first half of the tasks
plot_results_first_half(target_common_accuracy, 'Target Common Accuracy (First Half)', 'Common Accuracy (%)')
plot_results_first_half(target_outlier_accuracy, 'Target Outlier Accuracy (First Half)', 'Outlier Accuracy (%)')
plot_results_first_half(accuracy_score, 'Accuracy Score (First Half)', 'Accuracy (%)')
plot_results_first_half(h_score, 'H-score (First Half)', 'H-score (%)')
plot_results_first_half(sngp_accuracy_score, 'SNGP Accuracy Score (First Half)', 'SNGP Accuracy (%)')

# Plotting the results for the second half of the tasks
plot_results_second_half(target_common_accuracy, 'Target Common Accuracy (Second Half)', 'Common Accuracy (%)')
plot_results_second_half(target_outlier_accuracy, 'Target Outlier Accuracy (Second Half)', 'Outlier Accuracy (%)')
plot_results_second_half(accuracy_score, 'Accuracy Score (Second Half)', 'Accuracy (%)')
plot_results_second_half(h_score, 'H-score (Second Half)', 'H-score (%)')
plot_results_second_half(sngp_accuracy_score, 'SNGP Accuracy Score (Second Half)', 'SNGP Accuracy (%)')

#Radar Chart of AUPRC
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Data
categories = ['0 to 1', '0 to 2', '0 to 3', '1 to 0', '1 to 2', '1 to 3', '2 to 0', '2 to 1', '2 to 3', '3 to 0', '3 to 1', '3 to 2']
num_categories = len(categories)

# AUPRC values for each model
auprc_values = {
    'CNN': [0.779, 0.815, 0.880, 0.862, 0.667, 0.667, 0.80, 0.50, 0.63, 0.71, 0.67,0.67],
    'CNN - Self Attention': [0.802, 0.875, 0.873, 0.741, 0.667, 0.667, 0.80, 0.50, 0.72, 0.60, 0.67, 0.67],
    'CNN-Openmax-SA': [0.869, 0.858, 0.911, 0.750, 0.667, 0.667, 0.78, 0.50, 0.83, 0.74, 0.67, 0.67],
    'WideResnet': [0.873, 0.766, 0.720, 0.750, 0.667, 0.667, 0.65, 0.50, 0.83, 0.70, 0.67, 0.67],
    'WideResnet - Self Attention': [0.875, 0.844, 0.540, 0.889, 0.667, 0.667, 0.80, 0.50, 0.83, 0.83, 0.67, 0.67],
    'WideResnet - Openmax Layer - Self Attention': [0.875, 0.875, 0.634, 0.750, 0.667, 0.667, 0.72, 0.50, 0.83, 0.83, 0.67, 0.67],
}

# Radar chart setup
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Create the radar chart for each model
for model, values in auprc_values.items():
    # Repeat the first value to close the circle
    values += values[:1]
    angles = [n / float(num_categories) * 2 * pi for n in range(num_categories)]
    angles += angles[:1]

    ax.plot(angles, values, label=model)
    ax.fill(angles, values, alpha=0.1)

# Add labels
plt.xticks(angles[:-1], categories, weight = 'bold', size=14)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=14, weight = 'bold')
plt.ylim(0, 1)

# Add title and legend
plt.title('AUPRC Results for Different Transfer Tasks and Models', size=20, color='black', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.show()

#Radar chart of SNGP Accuracy Score
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Data
categories = ['0 to 1', '0 to 2', '0 to 3', '1 to 0', '1 to 2', '1 to 3', '2 to 0', '2 to 1', '2 to 3', '3 to 0', '3 to 1', '3 to 2']
num_categories = len(categories)

sngp_accuracy_score = {
    'CNN': [70.00, 69.84, 70.26, 75.63, 56.44, 42.51, 56.84, 47.74, 49.10, 54.21, 68.84, 66.94],
    'CNN - Self Attention': [75.82, 77.92, 71.76, 75.22, 58.37, 54.49, 57.69, 49.19, 57.35, 54.54, 69.86, 75.86],
    'CNN-Openmax-SA': [72.30, 73.79, 75.38, 76.47, 59.42, 48.62, 55.97, 49.39, 58.05, 58.68, 74.09, 73.09],
    'WideResnet': [76.10, 70.09, 65.08, 74.02, 59.86, 54.49, 51.98, 46.48, 59.86, 55.54, 74.70, 73.20],
    'WideResnet - Self Attention': [77.68, 76.32, 52.11, 72.21, 57.13, 50.55, 56.28, 48.06, 60.32, 62.11, 65.09, 76.31],
    'WideResnet - Openmax Layer - Self Attention': [79.30, 75.44, 62.25, 76.14, 58.92, 53.39, 45.45, 47.72, 61.60, 64.60, 70.61, 70.48]
}

# Radar chart setup

# Create the radar chart for each model
def plot_radar_chart(data, categories, title):
    num_categories = len(categories)
    angles = [n / float(num_categories) * 2 * pi for n in range(num_categories)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Create the radar chart for each model
    for model, values in data.items():
        # Create a temporary list for plotting
        values += values[:1]
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.1)

    # Add labels
    plt.xticks(angles[:-1], categories, weight='bold', size=14)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=14, weight='bold')
    plt.ylim(0, 100)

    # Add title and legend
    plt.title(title, size=20, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.show()



data = {model: values[:] for model, values in sngp_accuracy_score.items()}

plot_radar_chart(data, categories, 'SNGP Accuracy Score Results for All Transfer Tasks')

