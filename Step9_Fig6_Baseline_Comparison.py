# GenerativeAURORA - Step 9: Updated Figure 6
# All 8 baselines + GenerativeAURORA + RCCC
# Uses ACTUAL numbers from step6 and step7 output
#
# Run:
#   python step9_fig6_baseline_comparison.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
DATASET_DIR = "C:/Users/user/Paper_10/Dataset"
FIG_DIR     = os.path.join(DATASET_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# STYLE
# -----------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         13,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   11,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.6,

})

# -----------------------------------------------------------------------
# DATA — ACTUAL numbers from your step6 + step7 + baseline runs
# -----------------------------------------------------------------------
methods = [
    "Gaussian baseline",
    "Historical bootstrap",
    "WGAN-GP",
    "Conditional VAE†",
    "MC Dropout LSTM‡",
    "TimeGrad-style",
    "DDPM (no regime)",
    "GenerativeAURORA + RCCC",
]

crps     = [99.39, 44.87, 46.62, 12.60, 11.08, 20.38, 20.64, 20.70]
coverage = [91.2,  90.6,  48.5,  15.4,  66.5,  92.6,  93.6,  96.1]
width    = [518.9, 254.0, 152.1,  3.98, 78.4,  120.5, 133.7, 151.6]
mae      = [146.79, 62.55, 61.56, 13.23, 13.70, 26.77, 27.77, 27.88]

n = len(methods)
colors_crps = ["#AAAAAA"] * (n-1) + ["#2563EB"]
colors_cov  = ["#AAAAAA"] * (n-1) + ["#16A34A"]

# -----------------------------------------------------------------------
# FIGURE: 2 panels side by side
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    "GenerativeAURORA + RCCC vs All Baselines: CRPS and 90% Coverage",
    fontsize=15, fontweight="bold", y=1.01
)

y_pos = np.arange(n)

# ── LEFT: CRPS ──────────────────────────────────────────────────────────
ax = axes[0]
bars = ax.barh(y_pos, crps, color=colors_crps,
               edgecolor="white", linewidth=0.5, height=0.65)

for bar, val, col in zip(bars, crps, colors_crps):
    text_x = bar.get_width() + 1.5
    ax.text(text_x, bar.get_y() + bar.get_height()/2,
            str(round(val, 2)),
            va="center", ha="left",
            fontsize=11, fontweight="bold",
            color="#1E40AF" if col == "#2563EB" else "#333333")

ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=12)
ax.set_xlabel("CRPS (W/m²,  lower is better)", fontsize=13)
ax.set_title("CRPS Comparison", fontsize=14, fontweight="bold")
ax.set_xlim(0, 115)
ax.axvline(20.70, color="#2563EB", lw=1.2, linestyle=":",
           alpha=0.5, label="Proposed (20.70)")
ax.legend(fontsize=10, loc="lower right")

# footnotes
ax.text(0.01, -0.09,
    "† Posterior collapse (coverage 15.4%)\n"
    "‡ Overconfident (coverage 66.5%)",
    transform=ax.transAxes, fontsize=9,
    color="#666666", va="top")

# ── RIGHT: COVERAGE ─────────────────────────────────────────────────────
ax2 = axes[1]
bars2 = ax2.barh(y_pos, coverage, color=colors_cov,
                 edgecolor="white", linewidth=0.5, height=0.65)

for bar, val, col in zip(bars2, coverage, colors_cov):
    text_x = bar.get_width() + 0.5
    ax2.text(text_x, bar.get_y() + bar.get_height()/2,
             str(round(val, 1)) + "%",
             va="center", ha="left",
             fontsize=11, fontweight="bold",
             color="#14532D" if col == "#16A34A" else "#333333")

ax2.set_yticks(y_pos)
ax2.set_yticklabels([""] * n)   # hide y labels (same as left)
ax2.set_xlabel("90% PI Coverage (%)  —  target: 90%", fontsize=13)
ax2.set_title("Coverage Comparison", fontsize=14, fontweight="bold")
ax2.set_xlim(0, 108)
ax2.axvline(90, color="#DC2626", lw=1.5, linestyle="--",
            alpha=0.8, label="Target 90%")
ax2.legend(fontsize=10, loc="lower right")

# highlight proposed
ax2.barh(n-1, coverage[n-1], color="#16A34A",
         edgecolor="#166534", linewidth=1.5, height=0.65)

plt.tight_layout()
out = os.path.join(FIG_DIR, "fig6_baseline_comparison_updated.png")
plt.savefig(out, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: " + out)

# -----------------------------------------------------------------------
# FIGURE 6B: Width comparison (additional panel)
# -----------------------------------------------------------------------
fig2, ax3 = plt.subplots(figsize=(10, 6))
ax3.set_title("Interval Width Comparison (90% PI)",
              fontsize=14, fontweight="bold")

colors_w = ["#AAAAAA"] * (n-1) + ["#7C3AED"]
bars3 = ax3.barh(y_pos, width, color=colors_w,
                 edgecolor="white", linewidth=0.5, height=0.65)

for bar, val, col in zip(bars3, width, colors_w):
    ax3.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
             str(round(val, 1)),
             va="center", ha="left", fontsize=11,
             fontweight="bold",
             color="#4C1D95" if col == "#7C3AED" else "#333333")

ax3.set_yticks(y_pos)
ax3.set_yticklabels(methods, fontsize=12)
ax3.set_xlabel("Average 90% PI Width (W/m²)", fontsize=13)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.grid(axis="x", alpha=0.3)

# note on VAE
ax3.text(0.01, -0.08,
    "† VAE width=3.98 W/m² indicates posterior collapse (degenerate scenario diversity)",
    transform=ax3.transAxes, fontsize=9, color="#666666")

plt.tight_layout()
out2 = os.path.join(FIG_DIR, "fig6b_width_comparison.png")
plt.savefig(out2, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: " + out2)

print("\nDone. Figure 6 updated with all 8 baselines.")