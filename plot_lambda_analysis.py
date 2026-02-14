# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Plot lambda sensitivity analysis for SVM On Tree
# Shows how different lambda values affect performance vs LinearSVC
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import ticker as mticker

# # Dataset sizes
# N_values = np.array([100, 200, 400, 1000, 2000, 5000])

# # Lambda values tested
# lambda_values = [1.0, 2.0, 50.0, 400.0, 1000.0, 10000.0]

# # Training time data (Full FIT including pairs scan for λ≠1, DP+DFS only for λ=1)
# training_data = {
#     1.0: [0.000098, 0.000161, 0.000298, 0.000789, 0.001866, 0.006436],  # DP+DFS only (from previous)
#     2.0: [0.000139, 0.000303, 0.000844, 0.004147, 0.015387, 0.117167],  # Full FIT (DP+DFS+pairs)
#     50.0: [0.000106, 0.000226, 0.000565, 0.002851, 0.011129, 0.073656],  # Full FIT (DP+DFS+pairs)
#     400.0: [0.000101, 0.000202, 0.000563, 0.002695, 0.009993, 0.076192],
#     1000.0: [0.000095, 0.000223, 0.000607, 0.002530, 0.009893, 0.076959],
#     10000.0: [0.000101, 0.000220, 0.000603, 0.002540, 0.009388, 0.072244],  
# }

# # Prediction time data for different lambdas
# prediction_data = {
#     1.0: [0.000026, 0.000031, 0.000042, 0.000069, 0.000114, 0.000260],
#     2.0: [0.000027, 0.000032, 0.000041, 0.000071, 0.000117, 0.000267],
#     50.0: [0.000019, 0.000022, 0.000024, 0.000050, 0.000085, 0.000158],
#     400.0: [0.000019, 0.000020, 0.000025, 0.000041, 0.000069, 0.000159],
#     1000.0: [0.000017, 0.000023, 0.000029, 0.000041, 0.000069, 0.000160],
#     10000.0: [0.000019, 0.000022, 0.000026, 0.000040, 0.000067, 0.000154],
# }

# # Accuracy data for different lambdas  
# accuracy_data = {
#     1.0: [90.50, 89.00, 89.25, 88.85, 88.85, 88.69],
#     2.0: [90.50, 89.00, 89.25, 88.85, 88.85, 88.69],
#     50.0: [90.00, 89.00, 89.62, 88.85, 88.85, 88.69],
#     400.0: [90.00, 88.75, 89.12, 88.55, 88.78, 88.69],
#     1000.0: [90.00, 88.75, 85.62, 88.55, 88.75, 88.69],
#     10000.0: [90.00, 88.75, 85.62, 83.75, 86.92, 86.77],
# }

# # LinearSVC baseline (updated with new benchmark results)
# # training_linear = [0.001257, 0.001382, 0.001513, 0.001879, 0.002653, 0.004992]
# # prediction_linear = [0.000239, 0.000242, 0.000254, 0.000277, 0.000327, 0.000484]
# # accuracy_linear = [89.75, 89.25, 89.12, 88.55, 88.89, 88.70]

# training_linear = [0.001242, 0.001302, 0.001389, 0.001879, 0.002467, 0.004707]
# prediction_linear = [0.000261, 0.000264, 0.000267, 0.000292, 0.000331, 0.000435]
# accuracy_linear = [89.75, 89.25, 89.12, 88.55, 88.89, 88.70]

# # Set larger font sizes for better visibility
# plt.rcParams.update({
#     'font.size': 12,
#     'axes.labelsize': 14,
#     'axes.titlesize': 16,
#     'xtick.labelsize': 11,
#     'ytick.labelsize': 12,
#     'legend.fontsize': 11,
#     'figure.titlesize': 18
# })

# # Create figure with 3 subplots in 1 row
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# # Colors for different lambdas
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
# lambda_labels = [f'λ={lam}' for lam in lambda_values]

# # Plot 1: Training Time Comparison
# ax = axes[0]
# ax.set_title('(a) Training Time vs λ', fontweight='bold')
# ax.set_xlabel('Dataset Size (N)', fontweight='bold')
# ax.set_ylabel('Training Time (s)', fontweight='bold')

# # Plot LinearSVC baseline
# ax.semilogx(N_values, training_linear, 'k--', linewidth=2, 
#            label='LinearSVC', alpha=0.8, marker='s', markersize=6)

# # Plot different lambdas
# for i, lam in enumerate(lambda_values):
#     ax.semilogx(N_values, training_data[lam], 'o-', color=colors[i], 
#                linewidth=2.5, markersize=6, label=lambda_labels[i], alpha=0.8)

# ax.set_xticks(N_values)
# ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
# ax.legend(loc='upper left', ncol=2)
# ax.grid(True, alpha=0.3)

# # Plot 2: Prediction Time Comparison  
# ax = axes[1]
# ax.set_title('(b) Prediction Time vs λ', fontweight='bold')
# ax.set_xlabel('Dataset Size (N)', fontweight='bold')
# ax.set_ylabel('Prediction Time (s)', fontweight='bold')

# # Plot LinearSVC baseline
# ax.semilogx(N_values, prediction_linear, 'k--', linewidth=2,
#            label='LinearSVC', alpha=0.8, marker='s', markersize=6)

# # Plot different lambdas
# for i, lam in enumerate(lambda_values):
#     ax.semilogx(N_values, prediction_data[lam], 'o-', color=colors[i],
#                linewidth=2.5, markersize=6, label=lambda_labels[i], alpha=0.8)

# ax.set_xticks(N_values)
# ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
# ax.legend(loc='upper left', ncol=2)
# ax.grid(True, alpha=0.3)

# # Plot 3: Accuracy Comparison
# ax = axes[2]
# ax.set_title('(c) Classification Accuracy vs λ', fontweight='bold')
# ax.set_xlabel('Dataset Size (N)', fontweight='bold')  
# ax.set_ylabel('Accuracy (%)', fontweight='bold')

# # Plot LinearSVC baseline
# ax.semilogx(N_values, accuracy_linear, 'k--', linewidth=2,
#            label='LinearSVC', alpha=0.8, marker='s', markersize=6)

# # Plot different lambdas
# for i, lam in enumerate(lambda_values):
#     ax.semilogx(N_values, accuracy_data[lam], 'o-', color=colors[i],
#                linewidth=2.5, markersize=6, label=lambda_labels[i], alpha=0.8)

# ax.set_xticks(N_values)
# ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
# ax.set_ylim([85, 92])
# ax.legend(loc='upper right', ncol=2)
# ax.grid(True, alpha=0.3)

# # Adjust layout and save
# plt.tight_layout(pad=3.0, w_pad=2.5)

# # Save plots
# output_dir = "/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images"
# plt.savefig(f"{output_dir}/lambda_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
# plt.savefig(f"{output_dir}/lambda_sensitivity_analysis.pdf", bbox_inches='tight')

# print("Lambda sensitivity analysis plots generated successfully!")
# print("Files created:")
# print("- lambda_sensitivity_analysis.png/pdf")

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Plot lambda sensitivity analysis for SVM On Tree
# Improved readability: log-log for time plots, distinct markers/linestyles, global legend, auto y-limits for accuracy.
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import ticker as mticker

# # -----------------------------
# # Data
# # -----------------------------
# N_values = np.array([100, 200, 400, 1000, 2000, 5000])
# lambda_values = [1.0, 2.0, 50.0, 400.0, 1000.0, 10000.0]

# training_data = {
#     1.0:    [0.000098, 0.000161, 0.000298, 0.000789, 0.001866, 0.006436],
#     2.0:    [0.000139, 0.000303, 0.000844, 0.004147, 0.015387, 0.117167],
#     50.0:   [0.000106, 0.000226, 0.000565, 0.002851, 0.011129, 0.073656],
#     400.0:  [0.000101, 0.000202, 0.000563, 0.002695, 0.009993, 0.076192],
#     1000.0: [0.000095, 0.000223, 0.000607, 0.002530, 0.009893, 0.076959],
#     10000.0:[0.000101, 0.000220, 0.000603, 0.002540, 0.009388, 0.072244],
# }

# prediction_data = {
#     1.0:    [0.000026, 0.000031, 0.000042, 0.000069, 0.000114, 0.000260],
#     2.0:    [0.000027, 0.000032, 0.000041, 0.000071, 0.000117, 0.000267],
#     50.0:   [0.000019, 0.000022, 0.000024, 0.000050, 0.000085, 0.000158],
#     400.0:  [0.000019, 0.000020, 0.000025, 0.000041, 0.000069, 0.000159],
#     1000.0: [0.000017, 0.000023, 0.000029, 0.000041, 0.000069, 0.000160],
#     10000.0:[0.000019, 0.000022, 0.000026, 0.000040, 0.000067, 0.000154],
# }

# accuracy_data = {
#     1.0:    [90.50, 89.00, 89.25, 88.85, 88.85, 88.69],
#     2.0:    [90.50, 89.00, 89.25, 88.85, 88.85, 88.69],
#     50.0:   [90.00, 89.00, 89.62, 88.85, 88.85, 88.69],
#     400.0:  [90.00, 88.75, 89.12, 88.55, 88.78, 88.69],
#     1000.0: [90.00, 88.75, 85.62, 88.55, 88.75, 88.69],
#     10000.0:[90.00, 88.75, 85.62, 83.75, 86.92, 86.77],
# }

# training_linear   = [0.001242, 0.001302, 0.001389, 0.001879, 0.002467, 0.004707]
# prediction_linear = [0.000261, 0.000264, 0.000267, 0.000292, 0.000331, 0.000435]
# accuracy_linear   = [89.75, 89.25, 89.12, 88.55, 88.89, 88.70]

# # -----------------------------
# # Style (make lines separable)
# # -----------------------------
# plt.rcParams.update({
#     'font.size': 12,
#     'axes.labelsize': 14,
#     'axes.titlesize': 16,
#     'xtick.labelsize': 11,
#     'ytick.labelsize': 12,
#     'legend.fontsize': 11,
# })

# # Distinct markers + line styles (this is what fixes "overlap" perception most)
# markers    = ['o', 's', '^', 'D', 'P', 'X']
# linestyles = ['-', '--', '-.', ':', (0, (5, 2)), (0, (3, 1, 1, 1))]

# # Keep your colors (optional). If you want, you can remove colors and rely on markers/linestyles only.
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# def format_x_as_int(ax):
#     ax.set_xticks(N_values)
#     ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))

# def add_grid(ax):
#     ax.grid(True, which='both', alpha=0.25, linewidth=0.8)

# # -----------------------------
# # Figure
# # -----------------------------
# fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), constrained_layout=True)

# # We will build one shared legend for the whole figure
# legend_handles = []
# legend_labels  = []

# # -----------------------------
# # (a) Training Time
# # -----------------------------
# ax = axes[0]
# ax.set_title('(a) Training Time vs λ', fontweight='bold')
# ax.set_xlabel('Dataset Size (N)', fontweight='bold')
# ax.set_ylabel('Training Time (s)', fontweight='bold')

# # Use log-log to separate lines across decades (this is key)
# ax.set_xscale('log', base=10)
# ax.set_yscale('log', base=10)

# h = ax.plot(
#     N_values, training_linear,
#     color='k', linestyle='-', linewidth=2.8,
#     marker='s', markersize=7, markerfacecolor='white', markeredgewidth=1.2,
#     alpha=0.95, label='LinearSVC', zorder=10
# )[0]
# legend_handles.append(h); legend_labels.append('LinearSVC')

# for i, lam in enumerate(lambda_values):
#     y = np.array(training_data[lam], dtype=float)
#     h = ax.plot(
#         N_values, y,
#         color=colors[i], linestyle='-', linewidth=2.6,
#         marker=markers[i], markersize=7,
#         markerfacecolor='white', markeredgewidth=1.2,
#         alpha=0.95, label=f'λ={lam}', zorder=5
#     )[0]
#     legend_handles.append(h); legend_labels.append(f'λ={lam}')

# format_x_as_int(ax)
# add_grid(ax)

# # -----------------------------
# # (b) Prediction Time
# # -----------------------------
# ax = axes[1]
# ax.set_title('(b) Prediction Time vs λ', fontweight='bold')
# ax.set_xlabel('Dataset Size (N)', fontweight='bold')
# ax.set_ylabel('Prediction Time (s)', fontweight='bold')

# ax.set_xscale('log', base=10)
# ax.set_yscale('log', base=10)

# ax.plot(
#     N_values, prediction_linear,
#     color='k', linestyle='-', linewidth=2.8,
#     marker='s', markersize=7, markerfacecolor='white', markeredgewidth=1.2,
#     alpha=0.95, zorder=10
# )

# for i, lam in enumerate(lambda_values):
#     y = np.array(prediction_data[lam], dtype=float)
#     ax.plot(
#         N_values, y,
#         color=colors[i], linestyle='-', linewidth=2.6,
#         marker=markers[i], markersize=7,
#         markerfacecolor='white', markeredgewidth=1.2,
#         alpha=0.95, zorder=5
#     )

# format_x_as_int(ax)
# add_grid(ax)

# # -----------------------------
# # (c) Accuracy
# # -----------------------------
# ax = axes[2]
# ax.set_title('(c) Classification Accuracy vs λ', fontweight='bold')
# ax.set_xlabel('Dataset Size (N)', fontweight='bold')
# ax.set_ylabel('Accuracy (%)', fontweight='bold')

# ax.set_xscale('log', base=10)

# ax.plot(
#     N_values, accuracy_linear,
#     color='k', linestyle='-', linewidth=2.8,
#     marker='s', markersize=7, markerfacecolor='white', markeredgewidth=1.2,
#     alpha=0.95, zorder=10
# )

# for i, lam in enumerate(lambda_values):
#     y = np.array(accuracy_data[lam], dtype=float)
#     ax.plot(
#         N_values, y,
#         color=colors[i], linestyle='-', linewidth=2.6,
#         marker=markers[i], markersize=7,
#         markerfacecolor='white', markeredgewidth=1.2,
#         alpha=0.95, zorder=5
#     )

# format_x_as_int(ax)

# # Auto y-limits so lines are never clipped (fixes your “line lọt xuống dưới”)
# all_acc = [accuracy_linear]
# for lam in lambda_values:
#     all_acc.append(accuracy_data[lam])
# all_acc = np.array(all_acc, dtype=float)
# ymin = np.nanmin(all_acc) - 0.5
# ymax = np.nanmax(all_acc) + 0.5
# ax.set_ylim([ymin, ymax])

# add_grid(ax)

# # -----------------------------
# # Shared legend (cleaner than 3 legends)
# # -----------------------------
# # Put legend on top center, not hiding curves
# fig.legend(
#     legend_handles, legend_labels,
#     loc='upper center', ncol=4, frameon=True,
#     bbox_to_anchor=(0.5, 1.03)
# )

# # -----------------------------
# # Save
# # -----------------------------
# output_dir = "/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images"
# plt.savefig(f"{output_dir}/lambda_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
# plt.savefig(f"{output_dir}/lambda_sensitivity_analysis.pdf", bbox_inches='tight')

# print("Lambda sensitivity analysis plots generated successfully!")
# print("Files created:")
# print("- lambda_sensitivity_analysis.png/pdf")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

# -----------------------------
# Data
# -----------------------------
N_values = np.array([100, 200, 400, 1000, 2000, 5000])
lambda_values = [1.0, 2.0, 50.0, 400.0, 1000.0, 10000.0]

training_data = {
    1.0:    [0.000103, 0.000170, 0.000322, 0.000874, 0.002404, 0.008984],
    2.0:    [0.000519, 0.001898, 0.008712, 0.054825, 0.200795, 1.361056],
    50.0:   [0.000818, 0.001993, 0.009296, 0.054086, 0.202749, 1.406915],
    400.0:  [0.000833, 0.001875, 0.007333, 0.049087, 0.199763, 1.315562],
    1000.0: [0.000676, 0.001980, 0.008348, 0.050284, 0.199732, 1.373516],
    10000.0:[0.000744, 0.002176, 0.008256, 0.052844, 0.205323, 1.331268],
}

prediction_data = {
    1.0:    [0.000028, 0.000034, 0.000043, 0.000069, 0.000124, 0.000276],
    2.0:    [0.000028, 0.000034, 0.000047, 0.000093, 0.000122, 0.000266],
    50.0:   [0.000039, 0.000036, 0.000057, 0.000096, 0.000122, 0.000271],
    400.0:  [0.000040, 0.000034, 0.000046, 0.000074, 0.000121, 0.000270],
    1000.0: [0.000033, 0.000035, 0.000046, 0.000078, 0.000118, 0.000288],
    10000.0:[0.000034, 0.000041, 0.000048, 0.000078, 0.000122, 0.000263],
}

accuracy_data = {
    1.0:    [90.50, 89.00, 89.62, 89.05, 88.90, 88.84],
    2.0:    [90.50, 89.00, 89.62, 89.05, 88.90, 88.84],
    50.0:   [90.50, 89.00, 89.62, 89.05, 88.90, 88.84],
    400.0:  [89.50, 87.50, 87.62, 89.00, 85.05, 88.84],
    1000.0: [89.50, 68.50, 89.62, 88.60, 87.72, 84.27],
    10000.0:[89.50, 88.75, 85.62, 82.50, 63.75, 87.79],
}

training_linear   = [0.001263, 0.001311, 0.001398, 0.002057, 0.003124, 0.006668]
prediction_linear = [0.000176, 0.000181, 0.000184, 0.000219, 0.000278, 0.000454]
accuracy_linear   = [89.75, 89.25, 89.12, 88.55, 88.89, 88.70]

# -----------------------------
# Style
# -----------------------------
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
})

# Solid lines for all
SOLID = '-'

# Distinct markers (main separation)
markers = ['o', 's', '^', 'D', 'P', 'X']
colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def format_x_as_int(ax):
    ax.set_xticks(N_values)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))

def add_grid(ax):
    ax.grid(True, which='both', alpha=0.25, linewidth=0.8)

# -----------------------------
# Figure
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))

legend_handles = []
legend_labels  = []

# (a) Training time
ax = axes[0]
ax.set_title('(a) Training Time vs λ', fontweight='bold')
ax.set_xlabel('Dataset Size (N)')
ax.set_ylabel('Training Time (s)',)
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

h = ax.plot(
    N_values, training_linear,
    color='k', linestyle=SOLID, linewidth=3.2,
    marker='s', markersize=7, markerfacecolor='white', markeredgewidth=1.2,
    alpha=0.95, label='LinearSVC', zorder=10
)[0]
legend_handles.append(h); legend_labels.append('LinearSVC')

for i, lam in enumerate(lambda_values):
    y = np.array(training_data[lam], dtype=float)
    h = ax.plot(
        N_values, y,
        color=colors[i], linestyle=SOLID, linewidth=2.8,
        marker=markers[i], markersize=7,
        markerfacecolor='white', markeredgewidth=1.2,
        alpha=0.95, label=f'λ={lam}', zorder=6
    )[0]
    legend_handles.append(h); legend_labels.append(f'λ={lam}')

format_x_as_int(ax)
add_grid(ax)

# (b) Prediction time
ax = axes[1]
ax.set_title('(b) Prediction Time vs λ', fontweight='bold')
ax.set_xlabel('Dataset Size (N)')
ax.set_ylabel('Prediction Time (s)')
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

ax.plot(
    N_values, prediction_linear,
    color='k', linestyle=SOLID, linewidth=3.2,
    marker='s', markersize=7, markerfacecolor='white', markeredgewidth=1.2,
    alpha=0.95, zorder=10
)

for i, lam in enumerate(lambda_values):
    y = np.array(prediction_data[lam], dtype=float)
    ax.plot(
        N_values, y,
        color=colors[i], linestyle=SOLID, linewidth=2.8,
        marker=markers[i], markersize=7,
        markerfacecolor='white', markeredgewidth=1.2,
        alpha=0.95, zorder=6
    )

format_x_as_int(ax)
add_grid(ax)

# (c) Accuracy
ax = axes[2]
ax.set_title('(c) Classification Accuracy vs λ', fontweight='bold')
ax.set_xlabel('Dataset Size (N)')
ax.set_ylabel('Accuracy (%)')
ax.set_xscale('log', base=10)

ax.plot(
    N_values, accuracy_linear,
    color='k', linestyle=SOLID, linewidth=3.2,
    marker='s', markersize=7, markerfacecolor='white', markeredgewidth=1.2,
    alpha=0.95, zorder=10
)

for i, lam in enumerate(lambda_values):
    y = np.array(accuracy_data[lam], dtype=float)
    ax.plot(
        N_values, y,
        color=colors[i], linestyle=SOLID, linewidth=2.8,
        marker=markers[i], markersize=7,
        markerfacecolor='white', markeredgewidth=1.2,
        alpha=0.95, zorder=6
    )

format_x_as_int(ax)
add_grid(ax)

# Auto ylim (avoid clipping)
all_acc = [accuracy_linear] + [accuracy_data[lam] for lam in lambda_values]
all_acc = np.array(all_acc, dtype=float)
ymin = np.nanmin(all_acc) - 0.5
ymax = np.nanmax(all_acc) + 0.5
ax.set_ylim([ymin, ymax])

# -----------------------------
# Legend placement (NOT covering titles)
# Reserve top space for legend by using rect=[..., ..., ..., top]
# -----------------------------
plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88], w_pad=2.5)

fig.legend(
    legend_handles, legend_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.98),   # inside reserved top band
    ncol=4,
    frameon=True
)

# -----------------------------
# Save
# -----------------------------
import os
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/lambda_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/lambda_sensitivity_analysis.pdf", bbox_inches='tight')

print(f"Saved: {output_dir}/lambda_sensitivity_analysis.png / .pdf")
