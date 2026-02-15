# Plot benchmark results for SVM On Tree vs LinearSVC (lambda = 1).
# Four subplots in one row: (a) training time, (b) prediction time,
# (c) accuracy, (d) speedup ratios.
# Data from benchmark with 100 repeats (median), proper train/test split.

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Dataset sizes (n_per_class; total training set = 2 * n_per_class)
N = np.array([100, 200, 400, 1000, 2000, 5000])

# Training time (FIT, seconds, median over 100 repeats)
train_time_tree = np.array([0.000098, 0.000159, 0.000294, 0.000779, 0.001812, 0.006406])
train_time_linear = np.array([0.001226, 0.001279, 0.001381, 0.001842, 0.002419, 0.004670])

# Prediction time (seconds, median over 100 repeats)
pred_time_tree = np.array([0.000026, 0.000031, 0.000040, 0.000068, 0.000112, 0.000260])
pred_time_linear = np.array([0.000253, 0.000256, 0.000263, 0.000282, 0.000317, 0.000435])

# Accuracy (fraction)
acc_tree = np.array([0.9050, 0.8900, 0.8925, 0.8885, 0.8885, 0.8869])
acc_linear = np.array([0.8975, 0.8925, 0.8912, 0.8855, 0.8889, 0.8870])

speedup_train = train_time_linear / train_time_tree
speedup_pred = pred_time_linear / pred_time_tree

plt.style.use('default')


def set_logx_ticks(ax, N):
    ax.set_xscale('log')
    ax.set_xticks(N)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='x', style='plain')
    ax.minorticks_off()


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))
plt.subplots_adjust(wspace=0.35)

color_tree = '#2E86AB'
color_linear = '#A23B72'

title_size = 23
label_size = 22
tick_size = 18
legend_size = 16

# (a) Training time
ax1.loglog(N, train_time_tree, 'o-', color=color_tree, linewidth=2.5, markersize=8, label='SVM on Tree')
ax1.loglog(N, train_time_linear, 's-', color=color_linear, linewidth=2.5, markersize=8, label='LinearSVC')
ax1.set_title('(a) Training Time', fontsize=title_size, fontweight='bold')
ax1.set_xlabel('Number of samples (N)', fontsize=label_size)
ax1.set_ylabel('Training time (s)', fontsize=label_size)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=legend_size, loc='best')
ax1.set_xticks(N)
ax1.set_xticklabels([str(n) for n in N], fontsize=tick_size, rotation=45)
ax1.tick_params(axis='y', labelsize=tick_size)

# (b) Prediction time
ax2.loglog(N, pred_time_tree, 'o-', color=color_tree, linewidth=2.5, markersize=8, label='SVM on Tree')
ax2.loglog(N, pred_time_linear, 's-', color=color_linear, linewidth=2.5, markersize=8, label='LinearSVC')
ax2.set_title('(b) Prediction Time', fontsize=title_size, fontweight='bold')
ax2.set_xlabel('Number of samples (N)', fontsize=label_size)
ax2.set_ylabel('Prediction time (s)', fontsize=label_size)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=legend_size, loc='best')
ax2.set_xticks(N)
ax2.set_xticklabels([str(n) for n in N], fontsize=tick_size, rotation=45)
ax2.tick_params(axis='y', labelsize=tick_size)

# (c) Accuracy
ax3.plot(N, acc_tree, 'o-', color=color_tree, linewidth=2.5, markersize=8, label='SVM on Tree')
ax3.plot(N, acc_linear, 's-', color=color_linear, linewidth=2.5, markersize=8, label='LinearSVC')
ax3.set_title('(c) Classification Accuracy', fontsize=title_size, fontweight='bold')
ax3.set_xlabel('Number of samples (N)', fontsize=label_size)
ax3.set_ylabel('Accuracy', fontsize=label_size)
ax3.set_ylim(0.87, 0.92)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=legend_size, loc='best')
set_logx_ticks(ax3, N)
ax3.set_xticklabels([str(n) for n in N], fontsize=tick_size, rotation=45)
ax3.tick_params(axis='y', labelsize=tick_size)

# (d) Speedup ratios
ax4.plot(N, speedup_train, 'o-', color='#F18F01', linewidth=2.5, markersize=8, label='Training speedup')
ax4.plot(N, speedup_pred, 's-', color='#C73E1D', linewidth=2.5, markersize=8, label='Prediction speedup')
ax4.set_title('(d) Speedup Ratios', fontsize=title_size, fontweight='bold')
ax4.set_xlabel('Number of samples (N)', fontsize=label_size)
ax4.set_ylabel('Speedup (LinearSVC / Tree)', fontsize=label_size)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=legend_size, loc='best')
set_logx_ticks(ax4, N)
ax4.set_xticklabels([str(n) for n in N], fontsize=tick_size, rotation=45)
ax4.tick_params(axis='y', labelsize=tick_size)

plt.tight_layout(pad=2.0)

# Save
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{output_dir}/benchmark_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{output_dir}/benchmark_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_dir}/benchmark_comparison.png/pdf")
