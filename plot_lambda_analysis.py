# Plot lambda sensitivity analysis for SVM On Tree.
# Three subplots: (a) training time, (b) prediction time, (c) accuracy.
# Data from benchmark runs with lambda in {1, 2, 50, 400, 1000, 10000},
# sizes in {100, 200, 400, 1000, 2000, 5000}, and 100 repeats (median).

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

# Dataset sizes (n_per_class; total training set = 2 * n_per_class)
N_values = np.array([100, 200, 400, 1000, 2000, 5000])
lambda_values = [1.0, 2.0, 50.0, 400.0, 1000.0, 10000.0]

# SVM On Tree training time (seconds, median over 100 repeats)
training_data = {
    1.0:     [0.000103, 0.000170, 0.000322, 0.000874, 0.002404, 0.008984],
    2.0:     [0.000519, 0.001898, 0.008712, 0.054825, 0.200795, 1.361056],
    50.0:    [0.000818, 0.001993, 0.009296, 0.054086, 0.202749, 1.406915],
    400.0:   [0.000833, 0.001875, 0.007333, 0.049087, 0.199763, 1.315562],
    1000.0:  [0.000676, 0.001980, 0.008348, 0.050284, 0.199732, 1.373516],
    10000.0: [0.000744, 0.002176, 0.008256, 0.052844, 0.205323, 1.331268],
}

# SVM On Tree prediction time (seconds, median over 100 repeats)
prediction_data = {
    1.0:     [0.000028, 0.000034, 0.000043, 0.000069, 0.000124, 0.000276],
    2.0:     [0.000028, 0.000034, 0.000047, 0.000093, 0.000122, 0.000266],
    50.0:    [0.000039, 0.000036, 0.000057, 0.000096, 0.000122, 0.000271],
    400.0:   [0.000040, 0.000034, 0.000046, 0.000074, 0.000121, 0.000270],
    1000.0:  [0.000033, 0.000035, 0.000046, 0.000078, 0.000118, 0.000288],
    10000.0: [0.000034, 0.000041, 0.000048, 0.000078, 0.000122, 0.000263],
}

# SVM On Tree accuracy (%, median over 100 repeats)
accuracy_data = {
    1.0:     [90.50, 89.00, 89.62, 89.05, 88.90, 88.84],
    2.0:     [90.50, 89.00, 89.62, 89.05, 88.90, 88.84],
    50.0:    [90.50, 89.00, 89.62, 89.05, 88.90, 88.84],
    400.0:   [89.50, 87.50, 87.62, 89.00, 85.05, 88.84],
    1000.0:  [89.50, 68.50, 89.62, 88.60, 87.72, 84.27],
    10000.0: [89.50, 88.75, 85.62, 82.50, 63.75, 87.79],
}

# LinearSVC baseline (median over 100 repeats)
training_linear   = [0.001263, 0.001311, 0.001398, 0.002057, 0.003124, 0.006668]
prediction_linear = [0.000176, 0.000181, 0.000184, 0.000219, 0.000278, 0.000454]
accuracy_linear   = [89.75, 89.25, 89.12, 88.55, 88.89, 88.70]

# Plot style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
})

markers = ['o', 's', '^', 'D', 'P', 'X']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def format_x_as_int(ax):
    ax.set_xticks(N_values)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))


def add_grid(ax):
    ax.grid(True, which='both', alpha=0.25, linewidth=0.8)


fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
legend_handles = []
legend_labels = []

# (a) Training time
ax = axes[0]
ax.set_title('(a) Training Time vs lambda', fontweight='bold')
ax.set_xlabel('Dataset Size (N)')
ax.set_ylabel('Training Time (s)')
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

h = ax.plot(
    N_values, training_linear,
    color='k', linestyle='-', linewidth=3.2,
    marker='s', markersize=7, markerfacecolor='white', markeredgewidth=1.2,
    alpha=0.95, label='LinearSVC', zorder=10
)[0]
legend_handles.append(h)
legend_labels.append('LinearSVC')

for i, lam in enumerate(lambda_values):
    y = np.array(training_data[lam], dtype=float)
    h = ax.plot(
        N_values, y,
        color=colors[i], linestyle='-', linewidth=2.8,
        marker=markers[i], markersize=7,
        markerfacecolor='white', markeredgewidth=1.2,
        alpha=0.95, label=f'lambda={lam}', zorder=6
    )[0]
    legend_handles.append(h)
    legend_labels.append(f'lambda={lam}')

format_x_as_int(ax)
add_grid(ax)

# (b) Prediction time
ax = axes[1]
ax.set_title('(b) Prediction Time vs lambda', fontweight='bold')
ax.set_xlabel('Dataset Size (N)')
ax.set_ylabel('Prediction Time (s)')
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

ax.plot(
    N_values, prediction_linear,
    color='k', linestyle='-', linewidth=3.2,
    marker='s', markersize=7, markerfacecolor='white', markeredgewidth=1.2,
    alpha=0.95, zorder=10
)

for i, lam in enumerate(lambda_values):
    y = np.array(prediction_data[lam], dtype=float)
    ax.plot(
        N_values, y,
        color=colors[i], linestyle='-', linewidth=2.8,
        marker=markers[i], markersize=7,
        markerfacecolor='white', markeredgewidth=1.2,
        alpha=0.95, zorder=6
    )

format_x_as_int(ax)
add_grid(ax)

# (c) Accuracy
ax = axes[2]
ax.set_title('(c) Classification Accuracy vs lambda', fontweight='bold')
ax.set_xlabel('Dataset Size (N)')
ax.set_ylabel('Accuracy (%)')
ax.set_xscale('log', base=10)

ax.plot(
    N_values, accuracy_linear,
    color='k', linestyle='-', linewidth=3.2,
    marker='s', markersize=7, markerfacecolor='white', markeredgewidth=1.2,
    alpha=0.95, zorder=10
)

for i, lam in enumerate(lambda_values):
    y = np.array(accuracy_data[lam], dtype=float)
    ax.plot(
        N_values, y,
        color=colors[i], linestyle='-', linewidth=2.8,
        marker=markers[i], markersize=7,
        markerfacecolor='white', markeredgewidth=1.2,
        alpha=0.95, zorder=6
    )

format_x_as_int(ax)
add_grid(ax)

# Auto y-limits for accuracy
all_acc = [accuracy_linear] + [accuracy_data[lam] for lam in lambda_values]
all_acc = np.array(all_acc, dtype=float)
ax.set_ylim([np.nanmin(all_acc) - 0.5, np.nanmax(all_acc) + 0.5])

# Shared legend
plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88], w_pad=2.5)
fig.legend(
    legend_handles, legend_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.98),
    ncol=4,
    frameon=True,
)

# Save
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/lambda_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_dir}/lambda_sensitivity_analysis.pdf", bbox_inches='tight')
print(f"Saved: {output_dir}/lambda_sensitivity_analysis.png / .pdf")
