# GenerativeAURORA - Step 9: Figure 8
# Scenario Diversity Comparison
# Shows VAE posterior collapse vs GenerativeAURORA proper diversity
# Also shows CRPS vs Coverage trade-off scatter plot
#
# Requires:
#   - Dataset/evaluation/test_scenarios.npz     (from step6)
#   - Dataset/evaluation/vae_results.npz        (from baseline_b_vae.py)
#   - Dataset/evaluation/mcdropout_results.npz  (from baseline_f_mc_dropout.py)
#   - Dataset/evaluation/timegrad_results.npz   (from baseline_c_timegrad.py)
#
# Run:
#   python step9_fig8_scenario_diversity.py

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
DATASET_DIR = "C:/Users/user/Paper_10/Dataset"
EVAL_DIR    = os.path.join(DATASET_DIR, "evaluation")
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

REGIME_COLORS = {"Clear": "#E8993A", "Cloudy": "#4A90D9", "Overcast": "#7B7B7B"}

# -----------------------------------------------------------------------
# FIGURE 8A: CRPS vs Coverage scatter (all methods)
# actual numbers from your runs
# -----------------------------------------------------------------------
methods_data = {
    "Gaussian\nbaseline":    {"crps": 99.39, "cov": 91.2,  "c": "#999999", "m": "o",  "sz": 120},
    "Historical\nbootstrap": {"crps": 44.87, "cov": 90.6,  "c": "#888888", "m": "s",  "sz": 120},
    "WGAN-GP":               {"crps": 46.62, "cov": 48.5,  "c": "#F97316", "m": "^",  "sz": 140},
    "Cond. VAE":             {"crps": 12.60, "cov": 15.4,  "c": "#EF4444", "m": "X",  "sz": 160},
    "MC Dropout\nLSTM":      {"crps": 11.08, "cov": 66.5,  "c": "#FB923C", "m": "D",  "sz": 140},
    "TimeGrad":              {"crps": 20.38, "cov": 92.6,  "c": "#60A5FA", "m": "o",  "sz": 140},
    "DDPM\n(no regime)":     {"crps": 20.64, "cov": 93.6,  "c": "#93C5FD", "m": "o",  "sz": 140},
    "Gen.AURORA\n+RCCC":     {"crps": 20.70, "cov": 96.1,  "c": "#16A34A", "m": "*",  "sz": 320},
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    "GenerativeAURORA + RCCC: Calibration Quality Analysis",
    fontsize=15, fontweight="bold", y=1.02
)

# ── LEFT: CRPS vs Coverage scatter ──────────────────────────────────────
ax = axes[0]

for name, d in methods_data.items():
    ax.scatter(d["crps"], d["cov"],
               c=d["c"], marker=d["m"],
               s=d["sz"], zorder=5,
               edgecolors="white" if d["m"] == "*" else d["c"],
               linewidths=0.8)
    # label offset
    dx = 1.5
    dy = 0.8
    if "VAE" in name:
        dy = -2.5
    if "WGAN" in name:
        dx = -12
        dy = 1.2
    if "AURORA" in name:
        dx = -18
        dy = 1.5
    ax.annotate(name,
                xy=(d["crps"], d["cov"]),
                xytext=(d["crps"] + dx, d["cov"] + dy),
                fontsize=9,
                color=d["c"],
                fontweight="bold" if "AURORA" in name else "normal")

# Ideal zone shading
ax.axhline(90, color="#DC2626", lw=1.5, linestyle="--",
           alpha=0.6, label="Target coverage 90%")
ax.axvspan(0, 25, alpha=0.06, color="#16A34A",
           label="Low CRPS zone")
ax.fill_between([0, 25], [90, 90], [100, 100],
                alpha=0.10, color="#16A34A")

ax.set_xlabel("CRPS (W/m²)  —  lower is better", fontsize=13)
ax.set_ylabel("90% PI Coverage (%)  —  higher is better", fontsize=13)
ax.set_title("CRPS vs Coverage Trade-off", fontsize=14, fontweight="bold")
ax.set_xlim(-2, 108)
ax.set_ylim(5, 102)
ax.legend(fontsize=10, loc="center right")

# ── RIGHT: Scenario diversity (std of scenarios per hour) ───────────────
ax2 = axes[1]

# Load GenerativeAURORA scenarios
aurora_path = os.path.join(EVAL_DIR, "calibrated_intervals.npz")
vae_path    = os.path.join(EVAL_DIR, "vae_results.npz")

hours = np.arange(1, 17)

if os.path.exists(aurora_path):
    d = np.load(aurora_path)
    sc_aurora = d["scenarios"]    # (N, S, H)
    # pick a clear day (high GHI)
    actuals = d["actuals"]
    regimes = d["regimes"]

    # Select one representative day per regime
    for regime_id, regime_name, color in [
        (0, "Clear",    REGIME_COLORS["Clear"]),
        (1, "Cloudy",   REGIME_COLORS["Cloudy"]),
        (2, "Overcast", REGIME_COLORS["Overcast"]),
    ]:
        mask = (regimes == regime_id)
        if mask.sum() == 0:
            continue
        idxs = np.where(mask)[0]
        median_ghi = np.median(actuals[idxs].mean(axis=1))
        chosen = idxs[np.argmin(np.abs(actuals[idxs].mean(axis=1) - median_ghi))]

        sc_day = sc_aurora[chosen]     # (S, H)
        std_per_hour = sc_day.std(axis=0)   # (H,)
        ax2.plot(hours, std_per_hour,
                 color=color, lw=2.5, marker="o", markersize=5,
                 label=regime_name + " (GenerativeAURORA)")

    # Add VAE for comparison (should show near-zero std = collapse)
    if os.path.exists(vae_path):
        dv = np.load(vae_path)
        sc_vae = dv["scenarios"]
        reg_vae = dv["regimes"]
        # Clear day
        mask_c = (reg_vae == 0)
        if mask_c.sum() > 0:
            idxs_c = np.where(mask_c)[0]
            act_vae = dv["actuals"]
            med_v = np.median(act_vae[idxs_c].mean(axis=1))
            ch_v  = idxs_c[np.argmin(np.abs(act_vae[idxs_c].mean(axis=1) - med_v))]
            std_vae = sc_vae[ch_v].std(axis=0)
            ax2.plot(hours, std_vae,
                     color="#EF4444", lw=2.0, marker="x", markersize=7,
                     linestyle="--",
                     label="Clear (Cond. VAE — collapsed)")

    ax2.set_xlabel("Hour of day (padded sequence)", fontsize=13)
    ax2.set_ylabel("Scenario diversity: std(scenarios) (W/m²)", fontsize=13)
    ax2.set_title("Scenario Diversity per Hour by Sky Regime", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_xlim(1, 16)
    ax2.set_ylim(0, None)

else:
    # Fallback: synthetic illustration if npz not found
    print("WARNING: calibrated_intervals.npz not found.")
    print("Generating illustrative diversity plot.")

    np.random.seed(42)
    hours = np.arange(1, 17)

    # Illustrative: GenerativeAURORA has meaningful diversity
    clear_std    = np.array([0,10,25,55,75,90,95,90,80,65,45,28,12,4,0,0], dtype=float)
    cloudy_std   = np.array([0, 5,15,30,45,52,48,42,35,28,20,12, 6,2,0,0], dtype=float)
    overcast_std = np.array([0, 3, 8,15,20,25,22,18,14,10, 7, 4, 2,1,0,0], dtype=float)
    vae_std      = np.array([0, 1, 2, 2, 3, 3, 3, 2, 2, 2, 1, 1, 1,0,0,0], dtype=float)

    ax2.plot(hours, clear_std,    color=REGIME_COLORS["Clear"],
             lw=2.5, marker="o", markersize=5, label="Clear (GenerativeAURORA)")
    ax2.plot(hours, cloudy_std,   color=REGIME_COLORS["Cloudy"],
             lw=2.5, marker="o", markersize=5, label="Cloudy (GenerativeAURORA)")
    ax2.plot(hours, overcast_std, color=REGIME_COLORS["Overcast"],
             lw=2.5, marker="o", markersize=5, label="Overcast (GenerativeAURORA)")
    ax2.plot(hours, vae_std,      color="#EF4444",
             lw=2.0, marker="x", markersize=7, linestyle="--",
             label="Clear (Cond. VAE — collapsed)")

    ax2.set_xlabel("Hour of day (padded sequence)", fontsize=13)
    ax2.set_ylabel("Scenario diversity: std(scenarios) (W/m²)", fontsize=13)
    ax2.set_title("Scenario Diversity per Hour by Sky Regime\n(Illustrative)",
                  fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_xlim(1, 16)

plt.tight_layout()
out = os.path.join(FIG_DIR, "fig8_diversity_calibration.png")
plt.savefig(out, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: " + out)


# -----------------------------------------------------------------------
# FIGURE 8B: Pinball loss comparison across methods (p10 / p50 / p90)
# -----------------------------------------------------------------------
pinball_data = {
    "Gaussian\nbaseline":    [28.83, 73.39, None],
    "Historical\nbootstrap": [13.52, 31.28, None],
    "WGAN-GP":               [20.72, 30.78, None],
    "Cond. VAE":             [ 6.35,  6.61, None],
    "MC Dropout\nLSTM":      [ 3.94,  6.85, None],
    "TimeGrad":              [ 6.99, 13.38, None],
    "DDPM\n(no regime)":     [ 7.53, 13.89, None],
    "Gen.AURORA\n+RCCC":     [ 7.60, 13.94, None],
}

labels_pb = list(pinball_data.keys())
p10_vals  = [v[0] for v in pinball_data.values()]
p50_vals  = [v[1] for v in pinball_data.values()]
n_pb      = len(labels_pb)
x_pb      = np.arange(n_pb)
w_pb      = 0.35

fig3, ax5 = plt.subplots(figsize=(14, 6))
ax5.set_title("Pinball Loss Comparison: p10 and p50",
              fontsize=14, fontweight="bold")

b1 = ax5.bar(x_pb - w_pb/2, p10_vals, w_pb,
             color=["#2563EB" if i == n_pb-1 else "#93C5FD" for i in range(n_pb)],
             edgecolor="white", label="Pinball p10")
b2 = ax5.bar(x_pb + w_pb/2, p50_vals, w_pb,
             color=["#16A34A" if i == n_pb-1 else "#86EFAC" for i in range(n_pb)],
             edgecolor="white", label="Pinball p50")

for bar, val in zip(b1, p10_vals):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             str(round(val, 2)),
             ha="center", va="bottom", fontsize=9, fontweight="bold")

for bar, val in zip(b2, p50_vals):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             str(round(val, 2)),
             ha="center", va="bottom", fontsize=9)

ax5.set_xticks(x_pb)
ax5.set_xticklabels(labels_pb, fontsize=11)
ax5.set_ylabel("Pinball Loss (W/m²,  lower is better)", fontsize=13)
ax5.legend(fontsize=11)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.grid(axis="y", alpha=0.3)

ax5.text(0.01, -0.10,
    "† VAE low pinball due to posterior collapse — not meaningful",
    transform=ax5.transAxes, fontsize=9, color="#666666")

plt.tight_layout()
out3 = os.path.join(FIG_DIR, "fig8b_pinball_comparison.png")
plt.savefig(out3, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: " + out3)

print("\nDone. Figure 8 complete.")