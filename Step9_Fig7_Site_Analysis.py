# GenerativeAURORA - Step 9: Figure 7
# CRPS and Coverage per Site
# Shows geographical generalization across 5 diverse US climates
#
# Run:
#   python step9_fig7_site_analysis.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "font.family":     "DejaVu Sans",
    "font.size":       13,
    "axes.titlesize":  14,
    "axes.labelsize":  13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linewidth":  0.6,
})

# -----------------------------------------------------------------------
# DATA — actual numbers from step7 output
# -----------------------------------------------------------------------
sites        = ["Phoenix AZ\n(Hot desert)",
                "Los Angeles CA\n(Mediterranean)",
                "Denver CO\n(Semi-arid)",
                "Miami FL\n(Subtropical)",
                "Seattle WA\n(Oceanic)"]

crps_vals    = [17.83, 15.43, 25.56, 27.20, 17.76]
cov90_vals   = [96.33, 96.81, 95.24, 95.82, 96.40]
cov80_vals   = [87.36, 88.89, 85.46, 88.15, 84.68]
width_vals   = [136.87, 119.46, 171.65, 197.07, 130.28]
mae_vals     = [23.78, 20.45, 34.82, 36.14, 24.57]

x     = np.arange(len(sites))
width = 0.38

SITE_COLORS = ["#D94F3D", "#E8993A", "#4A90D9", "#3DAE6B", "#7B4FA6"]

# -----------------------------------------------------------------------
# FIGURE 7A: CRPS per site
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "GenerativeAURORA + RCCC: Per-Site Performance Analysis",
    fontsize=15, fontweight="bold", y=1.02
)

# Left: CRPS per site
ax = axes[0]
bars = ax.bar(x, crps_vals, color=SITE_COLORS,
              edgecolor="white", linewidth=0.8, width=0.6)

for bar, val in zip(bars, crps_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            str(round(val, 2)),
            ha="center", va="bottom",
            fontsize=12, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(sites, fontsize=11)
ax.set_ylabel("CRPS (W/m²,  lower is better)", fontsize=13)
ax.set_title("CRPS by Site", fontsize=14, fontweight="bold")
ax.set_ylim(0, 35)
ax.axhline(20.70, color="#2563EB", lw=1.5, linestyle="--",
           alpha=0.7, label="Overall mean (20.70)")
ax.legend(fontsize=11)

# Right: Coverage 90% and 80% per site
ax2 = axes[1]
bars1 = ax2.bar(x - width/2, cov90_vals, width,
                color=SITE_COLORS, alpha=0.9,
                edgecolor="white", label="Coverage 90% PI")
bars2 = ax2.bar(x + width/2, cov80_vals, width,
                color=SITE_COLORS, alpha=0.45,
                edgecolor="white", label="Coverage 80% PI")

for bar, val in zip(bars1, cov90_vals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2,
             str(round(val, 1)) + "%",
             ha="center", va="bottom",
             fontsize=10, fontweight="bold")

for bar, val in zip(bars2, cov80_vals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2,
             str(round(val, 1)) + "%",
             ha="center", va="bottom",
             fontsize=10)

ax2.axhline(90, color="#DC2626", lw=1.5, linestyle="--",
            alpha=0.8, label="Target 90%")
ax2.axhline(80, color="#F97316", lw=1.2, linestyle=":",
            alpha=0.6, label="Target 80%")
ax2.set_xticks(x)
ax2.set_xticklabels(sites, fontsize=11)
ax2.set_ylabel("Coverage (%)", fontsize=13)
ax2.set_title("Coverage by Site", fontsize=14, fontweight="bold")
ax2.set_ylim(70, 102)
ax2.legend(fontsize=11, loc="lower right")

plt.tight_layout()
out = os.path.join(FIG_DIR, "fig7_site_analysis.png")
plt.savefig(out, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: " + out)

# -----------------------------------------------------------------------
# FIGURE 7B: Interval width per site (spider/radar alternative: grouped bar)
# -----------------------------------------------------------------------
fig3, ax4 = plt.subplots(figsize=(10, 5))
ax4.set_title(
    "90% PI Width and Median MAE by Site",
    fontsize=14, fontweight="bold"
)

ax4_twin = ax4.twinx()
b1 = ax4.bar(x - width/2, width_vals, width,
             color=SITE_COLORS, alpha=0.85,
             edgecolor="white", label="PI Width (W/m²)")
b2 = ax4_twin.bar(x + width/2, mae_vals, width,
                  color=SITE_COLORS, alpha=0.45,
                  edgecolor="white", label="Median MAE (W/m²)")

ax4.set_xticks(x)
ax4.set_xticklabels(sites, fontsize=11)
ax4.set_ylabel("90% PI Width (W/m²)", fontsize=13)
ax4_twin.set_ylabel("Median MAE (W/m²)", fontsize=13)
ax4.spines["top"].set_visible(False)

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2,
           fontsize=11, loc="upper left")

for bar, val in zip(b1, width_vals):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             str(round(val, 1)),
             ha="center", va="bottom", fontsize=10, fontweight="bold")

for bar, val in zip(b2, mae_vals):
    ax4_twin.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.3,
                  str(round(val, 2)),
                  ha="center", va="bottom", fontsize=10)

plt.tight_layout()
out2 = os.path.join(FIG_DIR, "fig7b_width_mae_site.png")
plt.savefig(out2, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: " + out2)

print("\nDone. Figure 7 complete.")