"""NER F1 by entity type: CemSciBERT, MatSciBERT, and the three small LLMs.

Same layout as panel (b) in fig_2_abc.py. BERT scores are the published
percentages from that figure. LLM scores are phrase F1 from results/ner/report.md.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from new_plot import set_font

set_font(weight="normal", size=20)

labels = ["APL", "CMT", "DSC", "MAT", "PRO", "SMT", "SPL"]
# Order matches labels: APL, CMT, DSC, MAT, PRO, SMT, SPL.
series = [
    ("CemSciBERT", [46.27, 86.99, 46.51, 67.14, 74.96, 56.65, 27.18], "hotpink"),
    ("MatSciBERT", [44.26, 83.95, 44.84, 65.31, 72.76, 53.45, 34.29], "#785EF0"),
    ("Qwen2.5-0.5B", [0.00, 1.12, 0.00, 0.34, 0.00, 0.00, 0.00], "#648FFF"),
    ("Gemma3-270M", [0.75, 0.00, 0.60, 0.76, 2.72, 0.00, 0.00], "#FE6100"),
    ("Ministral-3B", [0.00, 0.00, 1.64, 10.44, 1.91, 0.00, 0.84], "#009E73"),
]

fig, ax = plt.subplots(figsize=(13.5, 5.5))
x = np.arange(len(labels))
width = 0.15
offsets = np.arange(len(series)) - (len(series) - 1) / 2
for offset, (name, scores, color) in zip(offsets, series):
    ax.bar(x + offset * width, scores, width, label=name, fc=color, ec="k")

ax.set_xlabel("Categories")
ax.set_ylabel("F1 Score")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 100)
ax.legend(ncol=3, handlelength=1, columnspacing=0.8, frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(out_dir, exist_ok=True)
pdf_path = os.path.join(out_dir, "fig_ner_llm.pdf")
png_path = os.path.join(out_dir, "fig_ner_llm.png")
fig.savefig(pdf_path, dpi=1200, format="pdf", bbox_inches="tight")
fig.savefig(png_path, dpi=300, format="png", bbox_inches="tight")
print(pdf_path)
print(png_path)
