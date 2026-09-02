# GenerativeAURORA - Step 8: Generate Paper Figures
#
# Produces publication-ready figures for the paper.
# All figures saved to Dataset/figures/ as high-resolution PNG (300 DPI).
#
# Figures produced:
#   fig1_training_loss.png        - Train vs val loss curve (paper Figure 1)
#   fig2_regime_distribution.png  - Sky regime counts per site (paper Figure 2)
#   fig3_scenario_examples.png    - Sample generated scenarios per regime (paper Figure 3)
#   fig4_coverage_comparison.png  - Before vs after RCCC coverage (paper Figure 4)
#   fig5_ghi_profile_by_site.png  - Average GHI profiles per site (paper Figure 5)
#
# No emoji, no special characters, pure matplotlib.
#
# Run:
#   python step8_figures.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, no GUI needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
DATASET_DIR = "C:/Users/user/Paper_10/Dataset"
EVAL_DIR    = os.path.join(DATASET_DIR, "evaluation")
FIG_DIR     = os.path.join(DATASET_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# STYLE SETTINGS  (clean, publication-ready)
# -----------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
})

REGIME_COLORS = {
    "Clear":    "#E8993A",
    "Cloudy":   "#4A90D9",
    "Overcast": "#7B7B7B",
}

SITE_COLORS = {
    "Phoenix_AZ":    "#D94F3D",
    "LosAngeles_CA": "#E8993A",
    "Denver_CO":     "#4A90D9",
    "Miami_FL":      "#3DAE6B",
    "Seattle_WA":    "#7B4FA6",
}

SITE_LABELS = {
    "Phoenix_AZ":    "Phoenix AZ",
    "LosAngeles_CA": "Los Angeles CA",
    "Denver_CO":     "Denver CO",
    "Miami_FL":      "Miami FL",
    "Seattle_WA":    "Seattle WA",
}


# -----------------------------------------------------------------------
# FIGURE 1: Training Loss Curve
# -----------------------------------------------------------------------
def fig1_training_loss():
    log_path = os.path.join(DATASET_DIR, "training_log.csv")
    if not os.path.exists(log_path):
        print("  [skip] training_log.csv not found")
        return

    df = pd.read_csv(log_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], color="#4A90D9", lw=1.5, label="Train loss")
    ax.plot(df["epoch"], df["val_loss"],   color="#D94F3D", lw=1.5, label="Validation loss", linestyle="--")

    best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
    best_val   = df["val_loss"].min()
    ax.axvline(best_epoch, color="#7B7B7B", lw=0.8, linestyle=":")
    ax.annotate(
        "Best: epoch " + str(best_epoch) + "\nval=" + str(round(best_val, 4)),
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 10, best_val + 0.01),
        fontsize=9,
        color="#7B7B7B",
        arrowprops=dict(arrowstyle="->", color="#7B7B7B", lw=0.8),
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized scale)")
    ax.set_title("GenerativeAURORA Training and Validation Loss")
    ax.legend(framealpha=0.8)
    ax.set_xlim(1, len(df))
    ax.set_ylim(0, df["train_loss"].max() * 1.1)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_training_loss.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("  Saved: " + out)


# -----------------------------------------------------------------------
# FIGURE 2: Sky Regime Distribution per Site
# -----------------------------------------------------------------------
def fig2_regime_distribution():
    regime_path = os.path.join(DATASET_DIR, "eda_output", "regime_distribution.csv")
    if not os.path.exists(regime_path):
        print("  [skip] regime_distribution.csv not found -- run eda_nsrdb.py first")
        return

    df = pd.read_csv(regime_path, index_col=0)

    pct_cols = [c for c in df.columns if c.endswith("_PCT")]
    sites    = [s.replace("_", " ") for s in df.index.tolist()]
    regimes  = [c.replace("_PCT", "") for c in pct_cols]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x     = np.arange(len(sites))
    width = 0.25

    for i, (regime, col) in enumerate(zip(regimes, pct_cols)):
        vals   = df[col].values
        offset = (i - 1) * width
        bars   = ax.bar(x + offset, vals, width,
                        label=regime,
                        color=REGIME_COLORS.get(regime, "#999999"),
                        alpha=0.85,
                        edgecolor="white",
                        linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        str(round(val, 1)) + "%",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=15, ha="right")
    ax.set_ylabel("Percentage of daytime hours (%)")
    ax.set_title("Sky Regime Distribution per Site (2018-2022, daytime hours)")
    ax.legend(title="Sky regime", framealpha=0.8)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_regime_distribution.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("  Saved: " + out)


# -----------------------------------------------------------------------
# FIGURE 3: Sample Generated Scenarios per Regime
# -----------------------------------------------------------------------
def fig3_scenario_examples():
    sc_path = os.path.join(EVAL_DIR, "test_scenarios.npz")
    if not os.path.exists(sc_path):
        print("  [skip] test_scenarios.npz not found -- run step6_evaluate.py first")
        return

    data      = np.load(sc_path)
    actuals   = data["actuals"]    # (N, H)
    scenarios = data["scenarios"]  # (N, S, H)
    regimes   = data["regimes"]    # (N,)

    regime_names = {0: "Clear", 1: "Cloudy", 2: "Overcast"}
    hours        = np.arange(1, actuals.shape[1] + 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

    for ax, (regime_id, regime_name) in zip(axes, regime_names.items()):
        mask  = (regimes == regime_id)
        idxs  = np.where(mask)[0]

        if len(idxs) == 0:
            ax.set_title(regime_name + " (no samples)")
            continue

        # Pick a representative day (closest to median GHI for this regime)
        median_ghi = np.median(actuals[idxs].mean(axis=1))
        day_ghi    = actuals[idxs].mean(axis=1)
        chosen_idx = idxs[np.argmin(np.abs(day_ghi - median_ghi))]

        actual_day    = actuals[chosen_idx]       # (H,)
        scenario_day  = scenarios[chosen_idx]     # (S, H)

        color = REGIME_COLORS.get(regime_name, "#999999")

        # Plot scenarios as light traces
        for s in range(scenario_day.shape[0]):
            ax.plot(hours, scenario_day[s], color=color, alpha=0.12, lw=0.8)

        # Plot 10th/90th quantile band
        q10 = np.quantile(scenario_day, 0.10, axis=0)
        q90 = np.quantile(scenario_day, 0.90, axis=0)
        ax.fill_between(hours, q10, q90, alpha=0.25, color=color, label="80% PI")

        # Plot median scenario
        med = np.median(scenario_day, axis=0)
        ax.plot(hours, med, color=color, lw=1.5, linestyle="--", label="Median scenario")

        # Plot actual
        ax.plot(hours, actual_day, color="black", lw=2.0, label="Actual GHI", zorder=5)

        ax.set_title(regime_name + " sky")
        ax.set_xlabel("Hour of day")
        if ax is axes[0]:
            ax.set_ylabel("GHI (W/m2)")
        ax.set_xlim(1, len(hours))
        ax.set_ylim(0, max(actual_day.max(), q90.max()) * 1.15)
        ax.legend(fontsize=8, framealpha=0.8)

    fig.suptitle("GenerativeAURORA: Generated GHI Scenarios by Sky Regime (50 scenarios per day)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_scenario_examples.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("  Saved: " + out)


# -----------------------------------------------------------------------
# FIGURE 4: Before vs After RCCC Coverage
# -----------------------------------------------------------------------
def fig4_coverage_comparison():
    results_path = os.path.join(EVAL_DIR, "rccc_results.csv")
    if not os.path.exists(results_path):
        print("  [skip] rccc_results.csv not found -- run step7_conformal.py first")
        return

    df = pd.read_csv(results_path)
    regime_df = df[df["group"] == "regime"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    regimes   = regime_df["name"].tolist()
    x         = np.arange(len(regimes))
    width     = 0.35
    colors_r  = [REGIME_COLORS.get(r, "#999999") for r in regimes]

    # Left: Coverage before vs after
    ax = axes[0]
    bars_before = ax.bar(x - width / 2, regime_df["coverage_before"],
                         width, label="Before RCCC",
                         color=colors_r, alpha=0.5, edgecolor="white")
    bars_after  = ax.bar(x + width / 2, regime_df["coverage_after"],
                         width, label="After RCCC",
                         color=colors_r, alpha=0.9, edgecolor="white")

    ax.axhline(90, color="#D94F3D", lw=1.2, linestyle="--", label="Target 90%")

    for bar in bars_before:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(round(bar.get_height(), 1)) + "%",
                ha="center", va="bottom", fontsize=8)
    for bar in bars_after:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(round(bar.get_height(), 1)) + "%",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage: Before vs After RCCC")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, framealpha=0.8)

    # Right: Interval width before vs after
    ax2 = axes[1]
    ax2.bar(x - width / 2, regime_df["width_before"],
            width, label="Before RCCC",
            color=colors_r, alpha=0.5, edgecolor="white")
    ax2.bar(x + width / 2, regime_df["width_after"],
            width, label="After RCCC",
            color=colors_r, alpha=0.9, edgecolor="white")

    for i, row in regime_df.iterrows():
        delta_cov = round(row["coverage_after"] - row["coverage_before"], 1)
        sign      = "+" if delta_cov >= 0 else ""
        ax2.text(x[list(regime_df["name"]).index(row["name"])],
                 max(row["width_before"], row["width_after"]) + 3,
                 sign + str(delta_cov) + " pp",
                 ha="center", va="bottom", fontsize=8,
                 color="#D94F3D" if delta_cov > 5 else "#3DAE6B")

    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes)
    ax2.set_ylabel("Average interval width (W/m2)")
    ax2.set_title("Interval Width: Before vs After RCCC")
    ax2.legend(fontsize=9, framealpha=0.8)

    fig.suptitle("Effect of Regime-Conditioned Conformal Calibration on 90% Prediction Intervals",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_coverage_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("  Saved: " + out)


# -----------------------------------------------------------------------
# FIGURE 5: Average GHI Profile per Site
# -----------------------------------------------------------------------
def fig5_ghi_profiles():
    profile_path = os.path.join(DATASET_DIR, "eda_output", "hourly_ghi_profile.csv")
    if not os.path.exists(profile_path):
        print("  [skip] hourly_ghi_profile.csv not found -- run eda_nsrdb.py first")
        return

    df = pd.read_csv(profile_path, index_col=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for col in df.columns:
        site_name  = col.strip()
        label      = SITE_LABELS.get(site_name, site_name)
        color      = SITE_COLORS.get(site_name, "#999999")
        valid_rows = df[col].dropna()
        ax.plot(valid_rows.index.astype(int), valid_rows.values,
                color=color, lw=2.0, label=label, marker="o", markersize=3)

    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("Mean GHI (W/m2)")
    ax.set_title("Average Daytime GHI Profile per Site (2018-2022)")
    ax.legend(title="Site", framealpha=0.8, loc="upper left")
    ax.set_xlim(4, 20)
    ax.set_ylim(0, None)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig5_ghi_profiles.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("  Saved: " + out)


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("GenerativeAURORA - Step 8: Generate Paper Figures")
    print("Output directory: " + FIG_DIR)
    print("")

    print("Figure 1: Training loss curve")
    fig1_training_loss()

    print("Figure 2: Regime distribution per site")
    fig2_regime_distribution()

    print("Figure 3: Scenario examples per regime")
    fig3_scenario_examples()

    print("Figure 4: Before vs after RCCC coverage")
    fig4_coverage_comparison()

    print("Figure 5: GHI profiles per site")
    fig5_ghi_profiles()

    print("")
    print("=== Step 8 complete ===")
    print("All figures saved to: " + FIG_DIR)
    print("")

    figs = [f for f in os.listdir(FIG_DIR) if f.endswith(".png")]
    for f in sorted(figs):
        fpath   = os.path.join(FIG_DIR, f)
        size_kb = round(os.path.getsize(fpath) / 1024, 1)
        print("  " + f.ljust(40) + str(size_kb) + " KB")

    print("")
    print("Next: review figures, then run step9_paper_tables.py to produce LaTeX tables.")


if __name__ == "__main__":
    main()