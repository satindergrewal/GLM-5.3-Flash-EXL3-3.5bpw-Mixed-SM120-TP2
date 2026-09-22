import matplotlib.pyplot as plt
import numpy as np

# All numbers below are parsed from bench2_results/*_summary.log (10 runs each,
# C=30). The chart shows every run; nothing is selected out.

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

# -------------------------------------------------------------
# CHART 1: Estonia long-context tracking (700k packet, 6-hop chain)
# Stacked: every one of the 30 streams lands in exactly one bucket.
# FAIL is zero in all 10 runs, so it is not a series.
# -------------------------------------------------------------
runs = ['R%d' % i for i in range(1, 11)]
estonia_pass  = [29, 30, 29, 27, 26, 28, 24, 28, 26, 21]
estonia_decoy = [ 0,  0,  1,  2,  0,  0,  0,  1,  0,  1]
estonia_trunc = [ 1,  0,  0,  1,  0,  1,  1,  1,  1,  2]
estonia_inc   = [30 - p - d - t for p, d, t in
                 zip(estonia_pass, estonia_decoy, estonia_trunc)]

x = np.arange(len(runs))
w = 0.62
c_pass, c_decoy, c_trunc, c_inc = '#2ecc71', '#e74c3c', '#f1c40f', '#555555'

ax1.bar(x, estonia_pass, w, label='PASS (Estonia)', color=c_pass)
ax1.bar(x, estonia_decoy, w, bottom=estonia_pass, label='DECOY (planted country)', color=c_decoy)
b2 = [p + d for p, d in zip(estonia_pass, estonia_decoy)]
ax1.bar(x, estonia_trunc, w, bottom=b2, label='TRUNC (cut off mid-answer)', color=c_trunc)
b3 = [a + t for a, t in zip(b2, estonia_trunc)]
ax1.bar(x, estonia_inc, w, bottom=b3, label='Incomplete (no visible answer)', color=c_inc)

ax1.set_title("Estonia long-context tracking, 10 runs\n"
              "700k packet, 6-hop chain, C=30  -  FAIL = 0 in every run",
              fontsize=12, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(runs, fontsize=10)
ax1.set_ylabel("Streams (of 30)", fontsize=10)
ax1.set_ylim(0, 39)
ax1.axhline(30, color='#888888', lw=0.6, ls=':')
ax1.grid(True, axis='y', linestyle='--', alpha=0.15)
ax1.legend(loc='upper right', frameon=True, facecolor='#222222', edgecolor='none', fontsize=9)

for i, p in enumerate(estonia_pass):
    ax1.text(i, p / 2, str(p), ha='center', va='center',
             color='#111111', fontweight='bold', fontsize=9)

# -------------------------------------------------------------
# CHART 2: Hotel-lights concurrency. Completions are what the bench
# scored; the rest never returned a visible answer. This is a capacity
# observation (prompts are 60-100k tokens against a 700k slot), not a
# measured queue stall - hence the neutral label.
# -------------------------------------------------------------
hotel_done = [6, 4, 6, 6, 7, 6, 4, 4, 5, 4]
hotel_rest = [30 - c for c in hotel_done]

ax2.bar(x, hotel_done, w, label='Completed and scored', color='#3498db')
ax2.bar(x, hotel_rest, w, bottom=hotel_done,
        label='No visible answer returned', color='#e67e22', alpha=0.35)

ax2.set_title("Hotel-lights at C=30, 10 runs\n"
              "prompts 60-100k tokens, slot 700k  -  completions stay at 4-7",
              fontsize=12, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(runs, fontsize=10)
ax2.set_ylabel("Streams (of 30)", fontsize=10)
ax2.set_ylim(0, 39)
ax2.axhline(30, color='#888888', lw=0.6, ls=':')
ax2.grid(True, axis='y', linestyle='--', alpha=0.15)
ax2.legend(loc='upper right', frameon=True, facecolor='#222222', edgecolor='none', fontsize=9)

for i, c in enumerate(hotel_done):
    ax2.text(i, c / 2, str(c), ha='center', va='center',
             color='#ffffff', fontweight='bold', fontsize=9)
    ax2.text(i, c + hotel_rest[i] / 2, str(hotel_rest[i]), ha='center', va='center',
             color='#f39c12', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('estonia_hotel_benchmarks.svg', format='svg', bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig('estonia_hotel_benchmarks.png', format='png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
print("charts written")
