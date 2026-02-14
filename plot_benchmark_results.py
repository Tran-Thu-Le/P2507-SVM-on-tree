# import matplotlib.pyplot as plt
# import numpy as np

# # Data from benchmark results (Fair Comparison with LinearSVC)
# N = np.array([100, 200, 400, 1000, 2000, 5000])

# # Training Time (s)
# train_time_tree = np.array([0.000106, 0.000174, 0.000340, 0.000769, 0.001623, 0.004299])
# train_time_linear = np.array([0.001281, 0.001371, 0.001468, 0.001642, 0.002001, 0.002991])

# # Prediction Time (s)
# pred_time_tree = np.array([0.000121, 0.000125, 0.000146, 0.000192, 0.000280, 0.000565])
# pred_time_linear = np.array([0.000232, 0.000243, 0.000246, 0.000257, 0.000283, 0.000346])

# # Accuracy (%)
# acc_tree = np.array([88.0, 90.5, 89.0, 89.1, 88.9, 88.8])
# acc_linear = np.array([85.5, 89.8, 89.3, 88.4, 88.6, 88.8])

# # Set up the plotting style - 4 plots in 1 row
# plt.style.use('default')
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

# # Colors
# color_tree = '#2E86AB'  # Blue for SVM On Tree
# color_linear = '#A23B72'  # Purple for LinearSVC

# # Consistent tick settings like accuracy plot
# def setup_consistent_ticks(ax, is_log=False):
#     if is_log:
#         ax.set_xticks(N)
#         ax.set_xticklabels([str(n) for n in N])
#     else:
#         ax.set_xticks(N)
#         ax.set_xticklabels([str(n) for n in N])
#     ax.tick_params(axis='both', which='major', labelsize=10)

# # 1. Training Time
# ax1.loglog(N, train_time_tree, 'o-', color=color_tree, linewidth=2.5, markersize=7, label='SVM On Tree')
# ax1.loglog(N, train_time_linear, 's-', color=color_linear, linewidth=2.5, markersize=7, label='LinearSVC')
# ax1.set_xlabel('Number of Samples (N)', fontsize=11)
# ax1.set_ylabel('Training Time (s)', fontsize=11)
# ax1.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
# ax1.grid(True, alpha=0.3)
# ax1.legend(fontsize=10)
# # Set explicit x-axis ticks for clarity
# ax1.set_xticks(N)
# ax1.set_xticklabels([str(n) for n in N])
# ax1.tick_params(axis='both', which='major', labelsize=10)

# # 2. Prediction Time  
# ax2.loglog(N, pred_time_tree, 'o-', color=color_tree, linewidth=2.5, markersize=7, label='SVM On Tree')
# ax2.loglog(N, pred_time_linear, 's-', color=color_linear, linewidth=2.5, markersize=7, label='LinearSVC')
# ax2.set_xlabel('Number of Samples (N)', fontsize=11)
# ax2.set_ylabel('Prediction Time (s)', fontsize=11)
# ax2.set_title('Prediction Time Comparison', fontsize=12, fontweight='bold')
# ax2.grid(True, alpha=0.3)
# ax2.legend(fontsize=10)
# # Set explicit x-axis ticks for clarity
# ax2.set_xticks(N)
# ax2.set_xticklabels([str(n) for n in N])
# ax2.tick_params(axis='both', which='major', labelsize=10)

# # 3. Accuracy (reference for consistent styling)
# ax3.plot(N, acc_tree, 'o-', color=color_tree, linewidth=2.5, markersize=7, label='SVM On Tree')
# ax3.plot(N, acc_linear, 's-', color=color_linear, linewidth=2.5, markersize=7, label='LinearSVC')
# ax3.set_xlabel('Number of Samples (N)', fontsize=11)
# ax3.set_ylabel('Accuracy (%)', fontsize=11)
# ax3.set_title('Classification Accuracy Comparison', fontsize=12, fontweight='bold')
# ax3.set_ylim(84, 92)  # Focus on the relevant range
# ax3.grid(True, alpha=0.3)
# ax3.legend(fontsize=10)
# # Consistent x-axis ticks
# ax3.set_xticks(N)
# ax3.set_xticklabels([str(n) for n in N])
# ax3.tick_params(axis='both', which='major', labelsize=10)

# # 4. Speedup Ratios
# speedup_train = train_time_linear / train_time_tree
# speedup_pred = pred_time_linear / pred_time_tree

# ax4.plot(N, speedup_train, 'o-', color='#F18F01', linewidth=2.5, markersize=7, label='Training Speedup')
# ax4.plot(N, speedup_pred, 's-', color='#C73E1D', linewidth=2.5, markersize=7, label='Prediction Speedup')
# ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
# ax4.set_xlabel('Number of Samples (N)', fontsize=11)
# ax4.set_ylabel('Speedup Ratio (LinearSVC/Tree)', fontsize=11)
# ax4.set_title('Performance Speedup Ratios', fontsize=12, fontweight='bold')
# ax4.grid(True, alpha=0.3)
# ax4.legend(fontsize=10)
# ax4.set_ylim(0.4, 15)
# # Consistent x-axis ticks
# ax4.set_xticks(N)
# ax4.set_xticklabels([str(n) for n in N])
# ax4.tick_params(axis='both', which='major', labelsize=10)

# # Adjust layout for 1 row
# plt.tight_layout(pad=2.0)

# # Save the figure
# plt.savefig('/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images/benchmark_comparison.png', 
#             dpi=300, bbox_inches='tight', facecolor='white')
# plt.savefig('/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images/benchmark_comparison.pdf', 
#             dpi=300, bbox_inches='tight', facecolor='white')

# # Create individual plots for better LaTeX integration
# fig_width, fig_height = 6, 4

# # Individual plot 1: Training Time
# plt.figure(figsize=(fig_width, fig_height))
# plt.loglog(N, train_time_tree, 'o-', color=color_tree, linewidth=2.5, markersize=7, label='SVM On Tree')
# plt.loglog(N, train_time_linear, 's-', color=color_linear, linewidth=2.5, markersize=7, label='LinearSVC')
# plt.xlabel('Number of Samples (N)', fontsize=12)
# plt.ylabel('Training Time (s)', fontsize=12)
# plt.title('Training Time Comparison', fontsize=13, fontweight='bold')
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=11)
# plt.tight_layout()
# plt.savefig('/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images/training_time.png', 
#             dpi=300, bbox_inches='tight', facecolor='white')
# plt.savefig('/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images/training_time.pdf', 
#             dpi=300, bbox_inches='tight', facecolor='white')
# plt.close()

# # Individual plot 2: Prediction Time
# plt.figure(figsize=(fig_width, fig_height))
# plt.loglog(N, pred_time_tree, 'o-', color=color_tree, linewidth=2.5, markersize=7, label='SVM On Tree')
# plt.loglog(N, pred_time_linear, 's-', color=color_linear, linewidth=2.5, markersize=7, label='LinearSVC')
# plt.xlabel('Number of Samples (N)', fontsize=12)
# plt.ylabel('Prediction Time (s)', fontsize=12)
# plt.title('Prediction Time Comparison', fontsize=13, fontweight='bold')
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=11)
# plt.tight_layout()
# plt.savefig('/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images/prediction_time.png', 
#             dpi=300, bbox_inches='tight', facecolor='white')
# plt.savefig('/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images/prediction_time.pdf', 
#             dpi=300, bbox_inches='tight', facecolor='white')
# plt.close()

# # Individual plot 3: Accuracy
# plt.figure(figsize=(fig_width, fig_height))
# plt.plot(N, acc_tree, 'o-', color=color_tree, linewidth=2.5, markersize=7, label='SVM On Tree')
# plt.plot(N, acc_linear, 's-', color=color_linear, linewidth=2.5, markersize=7, label='LinearSVC')
# plt.xlabel('Number of Samples (N)', fontsize=12)
# plt.ylabel('Accuracy (%)', fontsize=12)
# plt.title('Classification Accuracy Comparison', fontsize=13, fontweight='bold')
# plt.ylim(84, 92)
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=11)
# plt.tight_layout()
# plt.savefig('/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images/accuracy.png', 
#             dpi=300, bbox_inches='tight', facecolor='white')
# plt.savefig('/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images/accuracy.pdf', 
#             dpi=300, bbox_inches='tight', facecolor='white')
# plt.close()

# print("All plots have been generated and saved!")
# print("Files saved:")
# print("- benchmark_comparison.png/pdf (combined plot)")
# print("- training_time.png/pdf")
# print("- prediction_time.png/pdf") 
# print("- accuracy.png/pdf")

# import matplotlib.pyplot as plt
# import numpy as np

# # =========================
# # NEW Data (FAIR benchmark)
# # =========================
# N = np.array([200, 400, 800, 2000, 4000, 10000])

# train_time_tree = np.array([0.000098, 0.000237, 0.000735, 0.001592, 0.002648, 0.013825])
# train_time_linear = np.array([0.001328, 0.002342, 0.002457, 0.002402, 0.002755, 0.007296])

# pred_time_tree = np.array([0.000027, 0.000034, 0.000065, 0.000102, 0.000141, 0.000401])
# pred_time_linear = np.array([0.000267, 0.000444, 0.000619, 0.000452, 0.000409, 0.000790])

# acc_tree = np.array([0.9050, 0.8900, 0.8925, 0.8885, 0.8885, 0.8869])
# acc_linear = np.array([0.8975, 0.8925, 0.8912, 0.8855, 0.8889, 0.8870])

# speedup_train = train_time_linear / train_time_tree
# speedup_pred = pred_time_linear / pred_time_tree

# # =========================
# # Plot
# # =========================
# plt.style.use('default')
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# (ax1, ax2), (ax3, ax4) = axes

# color_tree = '#2E86AB'
# color_linear = '#A23B72'

# def set_xticks(ax):
#     ax.set_xticks(N)
#     ax.set_xticklabels([str(n) for n in N])
#     ax.tick_params(axis='both', which='major', labelsize=10)

# # (a) Training time
# ax1.loglog(N, train_time_tree, 'o-', color=color_tree, linewidth=2.5, markersize=7, label='SVM on Tree')
# ax1.loglog(N, train_time_linear, 's-', color=color_linear, linewidth=2.5, markersize=7, label='LinearSVC')
# ax1.set_xlabel('Number of training samples (N)', fontsize=11)
# ax1.set_ylabel('Training time (s)', fontsize=11)
# ax1.set_title('Training Time', fontsize=12, fontweight='bold')
# ax1.grid(True, alpha=0.3)
# ax1.legend(fontsize=10)
# set_xticks(ax1)

# # (b) Prediction time
# ax2.loglog(N, pred_time_tree, 'o-', color=color_tree, linewidth=2.5, markersize=7, label='SVM on Tree')
# ax2.loglog(N, pred_time_linear, 's-', color=color_linear, linewidth=2.5, markersize=7, label='LinearSVC')
# ax2.set_xlabel('Number of training samples (N)', fontsize=11)
# ax2.set_ylabel('Prediction time (s)', fontsize=11)
# ax2.set_title('Prediction Time (2N evaluation)', fontsize=12, fontweight='bold')
# ax2.grid(True, alpha=0.3)
# ax2.legend(fontsize=10)
# set_xticks(ax2)

# # (c) Accuracy
# ax3.plot(N, acc_tree, 'o-', color=color_tree, linewidth=2.5, markersize=7, label='SVM on Tree')
# ax3.plot(N, acc_linear, 's-', color=color_linear, linewidth=2.5, markersize=7, label='LinearSVC')
# ax3.set_xlabel('Number of training samples (N)', fontsize=11)
# ax3.set_ylabel('Accuracy', fontsize=11)
# ax3.set_title('Classification Accuracy', fontsize=12, fontweight='bold')
# ax3.set_ylim(0.87, 0.915)
# ax3.grid(True, alpha=0.3)
# ax3.legend(fontsize=10)
# set_xticks(ax3)

# # (d) Speedup
# ax4.plot(N, speedup_train, 'o-', color='#F18F01', linewidth=2.5, markersize=7, label='Training speedup')
# ax4.plot(N, speedup_pred, 's-', color='#C73E1D', linewidth=2.5, markersize=7, label='Prediction speedup')
# ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal')
# ax4.set_xlabel('Number of training samples (N)', fontsize=11)
# ax4.set_ylabel('Speedup (LinearSVC / Tree)', fontsize=11)
# ax4.set_title('Speedup Ratios', fontsize=12, fontweight='bold')
# ax4.grid(True, alpha=0.3)
# ax4.legend(fontsize=10)
# ax4.set_ylim(0.4, max(14.0, float(np.max(speedup_pred) * 1.1)))
# set_xticks(ax4)

# plt.tight_layout(pad=2.0)

# # Save 2x2 figure
# plt.savefig('/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images/benchmark_comparison_2x2.png',
#             dpi=300, bbox_inches='tight', facecolor='white')
# plt.savefig('/Users/tchuan/Code/SVM on Tree/P2507-SVM-on-tree/latex/data/images/benchmark_comparison_2x2.pdf',
#             dpi=300, bbox_inches='tight', facecolor='white')

# print("Saved combined plots: benchmark_comparison_2x2.png/.pdf")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# === NEW DATA (median over 7 runs, lambda=1.0, FAIR 2N eval) ===
# N = np.array([100, 200, 400, 1000, 2000, 5000])

# train_time_tree   = np.array([0.000098, 0.000237, 0.000735, 0.001592, 0.002648, 0.013825])
# train_time_linear = np.array([0.001328, 0.002342, 0.002457, 0.002402, 0.002755, 0.007296])

# pred_time_tree   = np.array([0.000027, 0.000034, 0.000065, 0.000102, 0.000141, 0.000401])
# pred_time_linear = np.array([0.000267, 0.000444, 0.000619, 0.000452, 0.000409, 0.000790])

# acc_tree   = np.array([0.9050, 0.8900, 0.8925, 0.8885, 0.8885, 0.8869])
# acc_linear = np.array([0.8975, 0.8925, 0.8912, 0.8855, 0.8889, 0.8870])

N = np.array([100, 200, 400, 1000, 2000, 5000])  # <-- keep this as n (per class)

# ===== repeats = 100 (medians) =====
# Training time (FIT, seconds)
train_time_tree   = np.array([0.000098, 0.000159, 0.000294, 0.000779, 0.001812, 0.006406])
train_time_linear = np.array([0.001226, 0.001279, 0.001381, 0.001842, 0.002419, 0.004670])

# Prediction time (seconds) on eval set size 2N_total = 4n
pred_time_tree   = np.array([0.000026, 0.000031, 0.000040, 0.000068, 0.000112, 0.000260])
pred_time_linear = np.array([0.000253, 0.000256, 0.000263, 0.000282, 0.000317, 0.000435])

# Accuracy
acc_tree   = np.array([0.9050, 0.8900, 0.8925, 0.8885, 0.8885, 0.8869])
acc_linear = np.array([0.8975, 0.8925, 0.8912, 0.8855, 0.8889, 0.8870])

speedup_train = train_time_linear / train_time_tree
speedup_pred  = pred_time_linear / pred_time_tree

plt.style.use('default')

def set_logx_ticks(ax, N):
    ax.set_xscale('log')
    ax.set_xticks(N)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='x', style='plain')
    ax.minorticks_off()

# ============================================================
# OLD CODE: 2x2 layout (commented out for reference)
# ============================================================
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))
# 
# # Margin
# plt.subplots_adjust(
#     wspace=0.4, # Khoảng cách chiều ngang (width space)
#     hspace=0.6  # Khoảng cách chiều dọc (height space)
# )
# 
# # Colors
# color_tree = '#2E86AB'
# color_linear = '#A23B72'
# 
# # 1) Training time (log-log)
# ax1.loglog(N, train_time_tree, 'o-', color=color_tree, linewidth=2.5, markersize=7, label='SVM on Tree')
# ax1.loglog(N, train_time_linear, 's-', color=color_linear, linewidth=2.5, markersize=7, label='LinearSVC')
# ax1.set_title('Training Time', fontsize=16, fontweight='bold')
# ax1.set_xlabel('Number of training samples (N)', fontsize=12)
# ax1.set_ylabel('Training time (s)', fontsize=12)
# ax1.grid(True, alpha=0.3)
# ax1.legend()
# ax1.set_xticks(N)
# ax1.set_xticklabels([str(n) for n in N])
# 
# # 2) Prediction time (log-log)
# ax2.loglog(N, pred_time_tree, 'o-', color=color_tree, linewidth=2.5, markersize=7, label='SVM on Tree')
# ax2.loglog(N, pred_time_linear, 's-', color=color_linear, linewidth=2.5, markersize=7, label='LinearSVC')
# ax2.set_title('Prediction Time (2N evaluation)', fontsize=16, fontweight='bold')
# ax2.set_xlabel('Number of training samples (N)', fontsize=12)
# ax2.set_ylabel('Prediction time (s)', fontsize=12)
# ax2.grid(True, alpha=0.3)
# ax2.legend()
# ax2.set_xticks(N)
# ax2.set_xticklabels([str(n) for n in N])
# 
# # 3) Accuracy (log-x to avoid overlap)
# ax3.plot(N, acc_tree, 'o-', color=color_tree, linewidth=2.5, markersize=7, label='SVM on Tree')
# ax3.plot(N, acc_linear, 's-', color=color_linear, linewidth=2.5, markersize=7, label='LinearSVC')
# ax3.set_title('Classification Accuracy', fontsize=16, fontweight='bold')
# ax3.set_xlabel('Number of training samples (N)', fontsize=12)
# ax3.set_ylabel('Accuracy', fontsize=12)
# ax3.grid(True, alpha=0.3)
# ax3.legend()
# set_logx_ticks(ax3, N)
# 
# # 4) Speedup (log-x to avoid overlap)
# ax4.plot(N, speedup_train, 'o-', linewidth=2.5, markersize=7, label='Training speedup')
# ax4.plot(N, speedup_pred,  's-', linewidth=2.5, markersize=7, label='Prediction speedup')
# ax4.set_title('Speedup Ratios', fontsize=16, fontweight='bold')
# ax4.set_xlabel('Number of training samples (N)', fontsize=12)
# ax4.set_ylabel('Speedup (LinearSVC / Tree)', fontsize=12)
# ax4.grid(True, alpha=0.3)
# ax4.legend()
# set_logx_ticks(ax4, N)
# 
# plt.tight_layout(pad=2.0)
# ============================================================

# ============================================================
# NEW CODE: 1x4 layout (4 plots in 1 row for paper)
# ============================================================
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))

# Margin for 1-row layout
plt.subplots_adjust(wspace=0.35)

# Colors
color_tree = '#2E86AB'
color_linear = '#A23B72'

# Font sizes for paper readability
title_size = 23
label_size = 22
tick_size = 18
legend_size = 16

# 1) Training time (log-log)
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

# 2) Prediction time (log-log)
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

# 3) Accuracy (log-x to avoid overlap)
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

# 4) Speedup (log-x to avoid overlap)
ax4.plot(N, speedup_train, 'o-', color='#F18F01', linewidth=2.5, markersize=8, label='Training speedup')
ax4.plot(N, speedup_pred,  's-', color='#C73E1D', linewidth=2.5, markersize=8, label='Prediction speedup')
# ax4.axhline(y=1, linestyle='--', color='gray', alpha=0.6, label='Equal (=1)')
ax4.set_title('(d) Speedup Ratios', fontsize=title_size, fontweight='bold')
ax4.set_xlabel('Number of samples (N)', fontsize=label_size)
ax4.set_ylabel('Speedup (LinearSVC / Tree)', fontsize=label_size)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=legend_size, loc='best')
set_logx_ticks(ax4, N)
ax4.set_xticklabels([str(n) for n in N], fontsize=tick_size, rotation=45)
ax4.tick_params(axis='y', labelsize=tick_size)

plt.tight_layout(pad=2.0)

import os
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{output_dir}/benchmark_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{output_dir}/benchmark_comparison.pdf',
            dpi=300, bbox_inches='tight', facecolor='white')

print(f"Saved: {output_dir}/benchmark_comparison.png/pdf (1x4 layout for paper)")
