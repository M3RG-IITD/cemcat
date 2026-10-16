from new_plot import *
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

set_font(weight='normal', size=20)

# --- Data for (a) ---
e2 = [2.04, 4.08, 6.11, 8.16, 10.2, 12.24, 14.27, 16.31,
      18.35, 20.39, 22.42, 24.46, 26.5, 28.54, 30]
l1 = [1.090848684310913, 1.082137107849121, 1.0749711990356445,
      1.0698847770690918, 1.0665922164916992, 1.064202904701233,
      1.0608892440795898, 1.0615594387054443, 1.0584527254104614,
      1.0571368932724, 1.0547302961349487, 1.0533106327056885,
      1.0546979904174805, 1.053818702697754, 1.0540558099746704]
l2 = [1.1629, 1.1455, 1.1355, 1.1287, 1.1236, 1.1192, 1.116,
      1.1137, 1.1114, 1.1092, 1.1075, 1.1066, 1.1055, 1.1051, 1.1048]

# --- Data for (b) ---
labels_b = ['APL', 'CMT', 'DSC', 'MAT', 'PRO', 'SMT', 'SPL']
cemscibert_scores = [46.27, 86.99, 46.51, 67.14, 74.96, 56.65, 27.18]
matscibert_scores = [44.26, 83.95, 44.84, 65.31, 72.76, 53.45, 34.29]

# --- Data for (c) ---
metrics = ['Precision', 'Recall', 'F1-Score']
rule_based = [86, 100, 93]
matscibert = [86, 84, 85]
cemscibert = [96, 100, 98]

# --- Combined figure: (a) + (c) on top, (b) on bottom ---
fig = plt.figure(figsize=(13.5, 11))
gs = GridSpec(2, 2, width_ratios=[5.5, 8],
             height_ratios=[5.5, 6], hspace=0.3, wspace=0.3)

# (a) Perplexity — top left
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(e2, np.exp(l1), '-o', color='hotpink', label='Val')
ax1.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax1.set_ylabel('Perplexity')
ax1.set_xlabel('Epoch')
ax1.text(0.85, 0.15, '(a)', transform=ax1.transAxes)
ax1.legend()

# (c) Model performance comparison — top right
ax3 = fig.add_subplot(gs[0, 1])
x_c = np.arange(len(metrics))
width_c = 0.25
ax3.bar(x_c - width_c, rule_based, width_c, label='Rules', fc='#648FFF', ec='k')
ax3.bar(x_c, matscibert, width_c, label='MatSciBERT', fc='#785EF0', ec='k')
ax3.bar(x_c + width_c, cemscibert, width_c, label='CemSciBERT', fc='hotpink', ec='k')
ax3.set_xticks(x_c)
ax3.set_xticklabels(metrics)
ax3.set_ylabel('Percentage (%)')
ax3.set_ylim(80, 110)
ax3.set_yticks([80, 85, 90, 95, 100])
# ax3.set_title('Model Performance Comparison')
ax3.legend(loc='upper left', ncol=2, handlelength=1, columnspacing=0.5,frameon=False)
ax3.grid(axis='y', linestyle='--', alpha=0.4)
ax3.text(0.9, 0.9, '(c)', transform=ax3.transAxes)

# (b) NER F1 scores — bottom, spanning full width
ax2 = fig.add_subplot(gs[1, :])
x_b = np.arange(len(labels_b))
width_b = 0.35
ax2.bar(x_b - width_b / 2, cemscibert_scores, width_b,
        label='CemSciBERT', fc='hotpink', ec='k')
ax2.bar(x_b + width_b / 2, matscibert_scores, width_b,
        label='MatSciBERT', fc='#785EF0', ec='k')
ax2.set_xlabel('Categories')
ax2.set_ylabel('F1 Score')
ax2.set_xticks(x_b)
ax2.set_xticklabels(labels_b)
ax2.legend(handlelength=1)
ax2.text(0.05, 0.9, '(b)', transform=ax2.transAxes)

# plt.savefig('../figures/fig_2_abc.png', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig_2_abc.pdf', dpi=1200, format='pdf',bbox_inches='tight')
plt.show()