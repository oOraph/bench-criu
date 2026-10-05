#!/usr/bin/env python3
"""Bar chart of the gpt-oss-120b checkpoint/restore results (p4de, A100-80GB, driver 615).
Numbers are the mean of the two runs in summary.md. Run: python3 plot_gptoss.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

variants = ["upstream criu-dev\n(--image-io-mode=direct)", "our plugin\n(parallel staging pages)", "our plugin\n+ CUDA custom storage"]
dump = [(59.378 + 58.467) / 2, (66.699 + 65.554) / 2, (11.481 + 11.497) / 2]
restore = [(41.928 + 42.314) / 2, (16.914 + 16.981) / 2, (10.556 + 10.553) / 2]

x = np.arange(len(variants)); w = 0.36
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w / 2, dump, w, label="dump (checkpoint)", color="#4C72B0")
b2 = ax.bar(x + w / 2, restore, w, label="restore (until /health answers)", color="#DD8452")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f} s", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x, variants)
ax.set_ylabel("seconds (lower is better)")
ax.set_title("vLLM + gpt-oss-120b: 76 GB of GPU memory checkpointed/restored\nA100-80GB, 8× NVMe RAID-0 (16 GB/s), driver 615.71.09, mean of 2 runs")
ax.set_ylim(0, max(dump) * 1.15)
ax.legend(loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("gptoss-120b-a100.png", dpi=130)
print("wrote gptoss-120b-a100.png")
