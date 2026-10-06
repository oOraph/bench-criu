#!/usr/bin/env python3
"""Bar chart of gpt-oss-120b checkpoint/restore on one p4de (A100-80GB, driver 615), 2026-10-06.
Means of the two runs in this directory's summary.md and ../2026-10_p4de_compression_zero_skip/summary.md.
Run: python3 plot_gptoss.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

variants = [
    "upstream criu-dev\n(--image-io-mode=direct)",
    "upstream criu-dev\n+ LZ4 256K, parallel decode",
    "our branch, custom storage off\n(parallel staging-page offload)",
    "our branch, custom storage on\n(driver >= 615)",
]
dump = [(57.124 + 58.336) / 2, (101.103 + 100.102) / 2, (51.796 + 51.480) / 2, (11.395 + 11.316) / 2]
restore = [(44.211 + 41.557) / 2, (70.482 + 69.808) / 2, (17.372 + 16.533) / 2, (10.366 + 10.346) / 2]

x = np.arange(len(variants)); w = 0.36
fig, ax = plt.subplots(figsize=(10.5, 5.2))
b1 = ax.bar(x - w / 2, dump, w, label="dump (checkpoint)", color="#4C72B0")
b2 = ax.bar(x + w / 2, restore, w, label="restore (until /health answers)", color="#DD8452")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f} s", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x, variants, fontsize=9)
ax.set_ylabel("seconds (lower is better)")
ax.set_title("vLLM + gpt-oss-120b: 76 GB of GPU memory checkpointed/restored\nA100-80GB, 8× NVMe RAID-0 (16 GB/s), driver 615.71.09, mean of 2 runs, one box")
ax.set_ylim(0, max(dump) * 1.15)
ax.legend(loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("gptoss-120b-a100.png", dpi=130)
print("wrote gptoss-120b-a100.png")
