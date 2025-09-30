#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot benchmark results with updated data and larger font sizes for 4 plots in 1 row.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Updated results from optimized C++ module (rebuilt and latest benchmark run)
N_values = np.array([100, 200, 400, 1000, 2000, 5000])

# Training time (DFS+DP only)
training_tree = np.array([0.000060, 0.000107, 0.000207, 0.000788, 0.001738, 0.005647])
training_linear = np.array([0.001331, 0.001456, 0.001503, 0.002033, 0.002651, 0.004976])

# Prediction time (cached, optimized)
prediction_tree = np.array([0.000016, 0.000018, 0.000021, 0.000026, 0.000038, 0.000074])
prediction_linear = np.array([0.000241, 0.000244, 0.000259, 0.000276, 0.000323, 0.000469])

# Accuracy (%)
accuracy_tree = np.array([90.50, 89.00, 89.25, 88.85, 88.85, 88.69])
accuracy_linear = np.array([89.75, 89.25, 89.12, 88.55, 88.89, 88.70])

# Calculate speedups
speedup_training = training_linear / training_tree
speedup_prediction = prediction_linear / prediction_tree

# Set larger font sizes for better visibility in 1-row layout
plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.labelsize': 16,      # Axis labels
    'axes.titlesize': 18,      # Subplot titles
    'xtick.labelsize': 11,     # X-axis tick labels (reduced to avoid overlap)
    'ytick.labelsize': 14,     # Y-axis tick labels
    'legend.fontsize': 14,     # Legend
    'figure.titlesize': 20     # Main title
})

# Create figure with 4 subplots in 1 row - increased width for better spacing
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# Colors for consistency
color_tree = '#2E8B57'      # SeaGreen
color_linear = '#FF6347'    # Tomato
alpha = 0.7

# Plot 1: Training Time Comparison (log-log scale)
axes[0].loglog(N_values, training_tree, 'o-', color=color_tree, linewidth=3, 
               markersize=8, label='SVM On Tree', alpha=alpha)
axes[0].loglog(N_values, training_linear, 's--', color=color_linear, linewidth=3, 
               markersize=8, label='LinearSVC', alpha=alpha)
axes[0].set_xlabel('Dataset Size (N)', fontweight='bold')
axes[0].set_ylabel('Training Time (s)', fontweight='bold')
axes[0].set_title('(a) Training Time', fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)
# Set explicit tick labels for better readability
axes[0].set_xticks(N_values)
axes[0].set_xticklabels([str(n) for n in N_values])

# Plot 2: Prediction Time Comparison (log-log scale)
axes[1].loglog(N_values, prediction_tree, 'o-', color=color_tree, linewidth=3, 
               markersize=8, label='SVM On Tree', alpha=alpha)
axes[1].loglog(N_values, prediction_linear, 's--', color=color_linear, linewidth=3, 
               markersize=8, label='LinearSVC', alpha=alpha)
axes[1].set_xlabel('Dataset Size (N)', fontweight='bold')
axes[1].set_ylabel('Prediction Time (s)', fontweight='bold')
axes[1].set_title('(b) Prediction Time', fontweight='bold')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(N_values)
axes[1].set_xticklabels([str(n) for n in N_values])

# Plot 3: Accuracy (dùng log-scale cho trục x)
axes[2].semilogx(N_values, accuracy_tree, 'o-', color=color_tree, linewidth=3,
                 markersize=8, label='SVM On Tree', alpha=alpha)
axes[2].semilogx(N_values, accuracy_linear, 's--', color=color_linear, linewidth=3,
                 markersize=8, label='LinearSVC', alpha=alpha)

axes[2].set_xlabel('Dataset Size (N)', fontweight='bold')
axes[2].set_ylabel('Accuracy (%)', fontweight='bold')
axes[2].set_title('(c) Classification Accuracy', fontweight='bold')
axes[2].legend(loc='lower right')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([85, 95])

# đặt tick đúng các N và hiển thị số đầy đủ
from matplotlib import ticker as mticker
axes[2].set_xticks(N_values)
axes[2].xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x)}')
)

# Plot 4: Speedup Ratios
axes[3].semilogx(N_values, speedup_training, 'o-', color='#4169E1', linewidth=3, 
                 markersize=8, label='Training Speedup', alpha=alpha)
axes[3].semilogx(N_values, speedup_prediction, 's--', color='#DC143C', linewidth=3, 
                 markersize=8, label='Prediction Speedup', alpha=alpha)
axes[3].axhline(y=1, color='black', linestyle='-', alpha=0.5, linewidth=2)
axes[3].set_xlabel('Dataset Size (N)', fontweight='bold')
axes[3].set_ylabel('Speedup Ratio', fontweight='bold')
axes[3].set_title('(d) Performance Speedup', fontweight='bold')
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3)
axes[3].set_ylim([0.5, 30])
axes[3].set_xticks(N_values)
axes[3].set_xticklabels([str(n) for n in N_values])

# Adjust layout and spacing - more aggressive spacing
plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)

# Save plots
output_dir = "/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images"
plt.savefig(f"{output_dir}/benchmark_comparison_fixed.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/benchmark_comparison_fixed.pdf", bbox_inches='tight')
# Also save with original name for backward compatibility
plt.savefig(f"{output_dir}/benchmark_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/benchmark_comparison.pdf", bbox_inches='tight')

# Save individual plots for reference
individual_titles = ['training_time', 'prediction_time', 'accuracy', 'speedup']
for i, title in enumerate(individual_titles):
    fig_single, ax_single = plt.subplots(figsize=(8, 6))
    
    if i == 0:  # Training time
        ax_single.loglog(N_values, training_tree, 'o-', color=color_tree, linewidth=3, 
                        markersize=8, label='SVM On Tree', alpha=alpha)
        ax_single.loglog(N_values, training_linear, 's--', color=color_linear, linewidth=3, 
                        markersize=8, label='LinearSVC', alpha=alpha)
        ax_single.set_ylabel('Training Time (s)', fontweight='bold')
        ax_single.set_title('Training Time Comparison', fontweight='bold')
    elif i == 1:  # Prediction time
        ax_single.loglog(N_values, prediction_tree, 'o-', color=color_tree, linewidth=3, 
                        markersize=8, label='SVM On Tree', alpha=alpha)
        ax_single.loglog(N_values, prediction_linear, 's--', color=color_linear, linewidth=3, 
                        markersize=8, label='LinearSVC', alpha=alpha)
        ax_single.set_ylabel('Prediction Time (s)', fontweight='bold')
        ax_single.set_title('Prediction Time Comparison', fontweight='bold')
    elif i == 2:  # Accuracy
        ax_single.plot(N_values, accuracy_tree, 'o-', color=color_tree, linewidth=3, 
                      markersize=8, label='SVM On Tree', alpha=alpha)
        ax_single.plot(N_values, accuracy_linear, 's--', color=color_linear, linewidth=3, 
                      markersize=8, label='LinearSVC', alpha=alpha)
        ax_single.set_ylabel('Accuracy (%)', fontweight='bold')
        ax_single.set_title('Classification Accuracy', fontweight='bold')
        ax_single.set_ylim([85, 95])
    else:  # Speedup
        ax_single.semilogx(N_values, speedup_training, 'o-', color='#4169E1', linewidth=3, 
                          markersize=8, label='Training Speedup', alpha=alpha)
        ax_single.semilogx(N_values, speedup_prediction, 's--', color='#DC143C', linewidth=3, 
                          markersize=8, label='Prediction Speedup', alpha=alpha)
        ax_single.axhline(y=1, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax_single.set_ylabel('Speedup Ratio', fontweight='bold')
        ax_single.set_title('Performance Speedup', fontweight='bold')
        ax_single.set_ylim([0.5, 30])
    
    ax_single.set_xlabel('Dataset Size (N)', fontweight='bold')
    ax_single.legend()
    ax_single.grid(True, alpha=0.3)
    ax_single.set_xticks(N_values)
    ax_single.set_xticklabels([str(n) for n in N_values])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/{title}.pdf", bbox_inches='tight')
    plt.close()

print("All plots have been generated and saved!")
print("Files created:")
print("- benchmark_comparison.png/pdf (4 plots in 1 row)")
print("- training_time.png/pdf")  
print("- prediction_time.png/pdf")
print("- accuracy.png/pdf")
print("- speedup.png/pdf")