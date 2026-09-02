#!/usr/bin/env python3
"""Release charts for the GLM-5.3-Flash EXL3 3.5bpw model card."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = "/tmp/charts"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})

OURS = "#2d8cf0"
REF = "#9aa5b1"
BAR = "#e05252"
OK = "#35b558"

# 1. KLD fidelity
fig, ax = plt.subplots(figsize=(7, 4))
labels = ["4bpw uniform\n(reference)", "3.5bpw mixed K3/K4\n(this checkpoint)"]
vals = [0.024555, 0.029654]
bars = ax.bar(labels, vals, color=[REF, OURS], width=0.5)
ax.axhline(0.06, color=BAR, linestyle="--", linewidth=2,
           label="absolute gate: mean < 0.06 nats")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.4f}",
            ha="center", fontweight="bold")
ax.set_ylabel("mean KLD vs BF16 teacher (nats)")
ax.set_ylim(0, 0.075)
ax.set_title("Five-run KLD fidelity gate - both PASS (2026-09-02)\n"
             "51,175 positions, full-vocab float64, teacher->student")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/kld-gate.png")
plt.close(fig)

# 2. Quality benchmarks
fig, ax = plt.subplots(figsize=(8, 4.2))
import numpy as np
x = np.arange(3)
w = 0.36
ours = [96.89, 77.8, 82.32]
ref = [96.51, 75.4, 72.6]
ax.bar(x - w / 2, ref, w, label="4bpw uniform (same hardware/protocol)", color=REF)
ax.bar(x + w / 2, ours, w, label="3.5bpw mixed K3/K4 (this checkpoint)", color=OURS)
for xi, (o, r) in zip(x, zip(ours, ref)):
    ax.text(xi + w / 2, o + 0.8, f"{o:.1f}", ha="center", fontweight="bold")
    ax.text(xi - w / 2, r + 0.8, f"{r:.1f}", ha="center", color="#555")
ax.set_xticks(x)
ax.set_xticklabels(["GSM8K 5-shot\n(thinking on)", "MATH-500\n(raw)", "HumanEval\n(raw)"])
ax.set_ylabel("score")
ax.set_ylim(0, 110)
ax.set_title("Quality gates at 3.5bpw - at or above 4bpw on all three")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/quality-gates.png")
plt.close(fig)

# 3. Speed vs context
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
# decode bars
labels = ["4bpw DFlash2 cell\n(192K-230K ctx)", "3.5bpw mixed\n(1M ctx, MTP3+graphs)"]
ax1.bar(labels, [91, 150], color=[REF, OURS], width=0.5)
ax1.text(0, 93, "78-91", ha="center", fontweight="bold")
ax1.text(1, 152, "126-150", ha="center", fontweight="bold")
ax1.set_ylabel("decode tok/s (engine-side)")
ax1.set_ylim(0, 175)
ax1.set_title("Decode: faster at 5x the context")
# prefill curve
depth = [500, 750, 950]
pre = [2836.7, 2840.6, 2793.2]
ax2.plot(depth, pre, "o-", color=OURS, linewidth=2.5)
for d, p in zip(depth, pre):
    ax2.annotate(f"{p:.0f}", (d, p), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontweight="bold")
ax2.set_xlabel("context depth (K tokens)")
ax2.set_ylabel("prefill tok/s")
ax2.set_ylim(2400, 3100)
ax2.set_title("Deep-context prefill: flat across depth")
fig.suptitle("3.5bpw mixed serving speeds (2x RTX PRO 6000, measured)", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/speed-depth.png", bbox_inches="tight")
plt.close(fig)

print("charts:", sorted(os.listdir(OUT)))
