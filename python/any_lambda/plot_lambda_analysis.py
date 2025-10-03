#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot lambda sensitivity analysis for SVM On Tree
Shows how different lambda values affect performance vs LinearSVC
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

# Dataset sizes
N_values = np.array([100, 200, 400, 1000, 2000, 5000])

# Lambda values tested
lambda_values = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]

# Training time data (Full FIT including pairs scan for λ≠1, DP+DFS only for λ=1)
training_data = {
    1.0: [0.000060, 0.000107, 0.000207, 0.000788, 0.001738, 0.005647],  # DP+DFS only (from previous)
    2.0: [0.000668, 0.002386, 0.008975, 0.053672, 0.221372, 1.436363],  # Full FIT (DP+DFS+pairs)
    5.0: [0.000720, 0.002490, 0.008864, 0.054253, 0.222587, 1.442955],  # Full FIT (DP+DFS+pairs)
    10.0: [0.000663, 0.002381, 0.009069, 0.053681, 0.218954, 1.443378], # Full FIT (DP+DFS+pairs)
    20.0: [0.000663, 0.002370, 0.008868, 0.053559, 0.220453, 1.549262], # Full FIT (DP+DFS+pairs)
    30.0: [0.000665, 0.002422, 0.008802, 0.053449, 0.218054, 1.443699]  # Full FIT (DP+DFS+pairs)
}

# Prediction time data for different lambdas
prediction_data = {
    1.0: [0.000016, 0.000018, 0.000021, 0.000026, 0.000038, 0.000074],
    2.0: [0.000018, 0.000021, 0.000025, 0.000029, 0.000040, 0.000089],
    5.0: [0.000020, 0.000022, 0.000025, 0.000030, 0.000040, 0.000088],
    10.0: [0.000018, 0.000021, 0.000025, 0.000030, 0.000039, 0.000088],
    20.0: [0.000018, 0.000022, 0.000026, 0.000031, 0.000041, 0.000089],
    30.0: [0.000019, 0.000023, 0.000025, 0.000030, 0.000041, 0.000070]
}

# Accuracy data for different lambdas  
accuracy_data = {
    1.0: [90.50, 89.00, 89.25, 88.85, 88.85, 88.69],
    2.0: [90.50, 89.00, 89.38, 88.60, 88.82, 88.60],
    5.0: [90.50, 89.00, 89.25, 89.00, 88.82, 88.66],
    10.0: [90.50, 89.00, 89.25, 89.00, 88.82, 88.66],
    20.0: [89.50, 87.50, 89.25, 89.00, 88.82, 88.74],
    30.0: [89.50, 89.00, 87.62, 89.00, 88.82, 88.74]
}

# LinearSVC baseline (updated with new benchmark results)
training_linear = [0.001257, 0.001382, 0.001513, 0.001879, 0.002653, 0.004992]
prediction_linear = [0.000239, 0.000242, 0.000254, 0.000277, 0.000327, 0.000484]
accuracy_linear = [89.75, 89.25, 89.12, 88.55, 88.89, 88.70]

# Set larger font sizes for better visibility
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18
})

# Create figure with 3 subplots in 1 row
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Colors for different lambdas
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
lambda_labels = [f'λ={lam}' for lam in lambda_values]

# Plot 1: Training Time Comparison
ax = axes[0]
ax.set_title('(a) Training Time vs λ', fontweight='bold')
ax.set_xlabel('Dataset Size (N)', fontweight='bold')
ax.set_ylabel('Training Time (s)', fontweight='bold')

# Plot LinearSVC baseline
ax.semilogx(N_values, training_linear, 'k--', linewidth=2, 
           label='LinearSVC', alpha=0.8, marker='s', markersize=6)

# Plot different lambdas
for i, lam in enumerate(lambda_values):
    ax.semilogx(N_values, training_data[lam], 'o-', color=colors[i], 
               linewidth=2.5, markersize=6, label=lambda_labels[i], alpha=0.8)

ax.set_xticks(N_values)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
ax.legend(loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)

# Plot 2: Prediction Time Comparison  
ax = axes[1]
ax.set_title('(b) Prediction Time vs λ', fontweight='bold')
ax.set_xlabel('Dataset Size (N)', fontweight='bold')
ax.set_ylabel('Prediction Time (s)', fontweight='bold')

# Plot LinearSVC baseline
ax.semilogx(N_values, prediction_linear, 'k--', linewidth=2,
           label='LinearSVC', alpha=0.8, marker='s', markersize=6)

# Plot different lambdas
for i, lam in enumerate(lambda_values):
    ax.semilogx(N_values, prediction_data[lam], 'o-', color=colors[i],
               linewidth=2.5, markersize=6, label=lambda_labels[i], alpha=0.8)

ax.set_xticks(N_values)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
ax.legend(loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)

# Plot 3: Accuracy Comparison
ax = axes[2]
ax.set_title('(c) Classification Accuracy vs λ', fontweight='bold')
ax.set_xlabel('Dataset Size (N)', fontweight='bold')  
ax.set_ylabel('Accuracy (%)', fontweight='bold')

# Plot LinearSVC baseline
ax.semilogx(N_values, accuracy_linear, 'k--', linewidth=2,
           label='LinearSVC', alpha=0.8, marker='s', markersize=6)

# Plot different lambdas
for i, lam in enumerate(lambda_values):
    ax.semilogx(N_values, accuracy_data[lam], 'o-', color=colors[i],
               linewidth=2.5, markersize=6, label=lambda_labels[i], alpha=0.8)

ax.set_xticks(N_values)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
ax.set_ylim([85, 92])
ax.legend(loc='lower right', ncol=2)
ax.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout(pad=3.0, w_pad=2.5)

# Save plots
output_dir = "/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images"
plt.savefig(f"{output_dir}/lambda_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/lambda_sensitivity_analysis.pdf", bbox_inches='tight')

print("Lambda sensitivity analysis plots generated successfully!")
print("Files created:")
print("- lambda_sensitivity_analysis.png/pdf")