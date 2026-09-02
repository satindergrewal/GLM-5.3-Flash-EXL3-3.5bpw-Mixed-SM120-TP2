#!/usr/bin/env python3
"""Corrected concurrency + agentic chart for the 3.5bpw model card.

Data source (box): eval/conc-ladder.json, confirmed pass 2026-09-02,
serve boot21 (1M ctx, MTP3, graphs, max_num_seqs=16,
max_num_batched_tokens=4096; non-stream, ignore_eos, forced 1024,
thinking off, 8K prompts). The old flat ~108-110 row was measured on
max_num_seqs=1 (serialized streams) and is shown as the invalid
reference line.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.environ.get("CHART_OUT", os.path.join(os.path.dirname(__file__), "out"))
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})

OURS = "#2d8cf0"
REF = "#9aa5b1"
BAR = "#e05252"
OK = "#35b558"

streams = [1, 2, 4, 6, 8, 12, 16]
# confirmed ladder: single warm median, then concurrency aggregates
agg = [186, 162.9, 207.4, 227.5, 246.4, 180.7, 189.0]
per_stream = [186, 88.0, 53.9, 39.5, 31.4, 15.6, 13.6]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

# Left: aggregate scaling
ax1.plot(streams, agg, "o-", color=OURS, linewidth=2.5,
         label="aggregate decode (confirmed pass)")
for s, a in zip(streams, agg):
    ax1.annotate(f"{a:.0f}", (s, a), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontweight="bold")
ax1.axhline(109, color=REF, linestyle="--", linewidth=2,
            label="old invalid row (max_num_seqs=1, serialized)")
ax1.annotate("108-110 flat = streams never shared a step",
             (10, 116), fontsize=9, color="#555", ha="center")
ax1.set_xlabel("concurrent streams")
ax1.set_ylabel("aggregate decode tok/s")
ax1.set_xticks(streams)
ax1.set_ylim(0, 280)
ax1.set_title("Concurrency scaling, 1M profile, MTP3 + graphs\n"
              "peak 246 tok/s at 8 streams")

# Right: agentic walls (as published)
labels = ["3 agent tasks\nserial", "3 agent tasks\nconcurrent"]
walls = [22.06, 3.32]
bars = ax2.bar(labels, walls, color=[REF, OK], width=0.5)
for b, v in zip(bars, walls):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f} s",
             ha="center", fontweight="bold")
ax2.annotate("6.6x", (1, 11), ha="center", fontsize=14,
             fontweight="bold", color=OK)
ax2.set_ylabel("wall time (s)")
ax2.set_ylim(0, 26)
ax2.set_title("Multi-agentic: 3 tool-using tasks (2 turns each)")

fig.suptitle("GLM-5.3-Flash 3.5bpw mixed - concurrency corrected 2026-09-02\n"
             "(max_num_seqs=16 batching profile, 2x RTX PRO 6000, measured)", y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/concurrency-agentic.png", bbox_inches="tight")
plt.close(fig)
print("written:", f"{OUT}/concurrency-agentic.png")
