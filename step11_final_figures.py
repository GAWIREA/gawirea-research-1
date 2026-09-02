# GenerativeAURORA - Step 11: Final Publication-Ready Figures
#
# Fixes from Step 8:
#   - Fig2 and Fig5 were skipped because EDA files were in Coding/ not Dataset/
#     Fixed: now reads from wherever the files exist
#   - Fig3 x-axis shows padded index (1-16), not real hours
#     Fixed: maps to approximate real hours (6am-9pm)
#   - Adds Fig6: CRPS and coverage comparison bar chart (all methods)
#
# All figures saved at 300 DPI to Dataset/figures/
#
# Run:
#   python step11_final_figures.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_DIR = "C:/Users/user/Paper_10/Dataset"
CODING_DIR  = "C:/Users/user/Paper_10/Coding"
EVAL_DIR    = os.path.join(DATASET_DIR, "evaluation")
TABLE_DIR   = os.path.join(DATASET_DIR, "paper_tables")
FIG_DIR     = os.path.join(DATASET_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

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

REGIME_COLORS = {"Clear": "#E8993A", "Cloudy": "#4A90D9", "Overcast": "#7B7B7B"}
SITE_COLORS   = {
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

# Approximate real daytime hours for 16-slot padded sequences
# Phoenix/LA start ~5am, Seattle starts ~6am -- use a compromise
HOUR_LABELS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def find_eda_file(filename):
    candidates = [
        os.path.join(DATASET_DIR, "eda_output", filename),
        os.path.join(CODING_DIR,  "eda_output", filename),
        os.path.join(CODING_DIR,  filename),
        os.path.join(DATASET_DIR, filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# -----------------------------------------------------------------------
# FIG 1: Training loss (re-save with improved annotation)
# -----------------------------------------------------------------------
def fig1_training_loss():
    log_path = os.path.join(DATASET_DIR, "training_log.csv")
    if not os.path.exists(log_path):
        print("  [skip] training_log.csv not found")
        return

    df = pd.read_csv(log_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], color="#4A90D9", lw=1.5, label="Train loss")
    ax.plot(df["epoch"], df["val_loss"],   color="#D94F3D", lw=1.5,
            label="Validation loss", linestyle="--")

    best_ep  = int(df.loc[df["val_loss"].idxmin(), "epoch"])
    best_val = float(df["val_loss"].min())
    ax.axvline(best_ep, color="#888888", lw=0.8, linestyle=":")
    ax.annotate("Best epoch " + str(best_ep) + "\nval=" + str(round(best_val, 4)),
                xy=(best_ep, best_val),
                xytext=(best_ep + 8, best_val + 0.012),
                fontsize=9, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (normalized scale)")
    ax.set_title("GenerativeAURORA: Training and Validation Loss")
    ax.legend(framealpha=0.8)
    ax.set_xlim(1, len(df))
    ax.set_ylim(0, df["val_loss"].quantile(0.98) * 1.1)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_training_loss.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("  Saved: " + out)


# -----------------------------------------------------------------------
# FIG 2: Sky regime distribution per site
# -----------------------------------------------------------------------
def fig2_regime_distribution():
    fpath = find_eda_file("regime_distribution.csv")
    if fpath is None:
        print("  [skip] regime_distribution.csv not found")
        return

    df      = pd.read_csv(fpath, index_col=0)
    pct_cols = [c for c in df.columns if c.endswith("_PCT")]
    sites    = [s.replace("_", " ") for s in df.index.tolist()]
    regimes  = [c.replace("_PCT", "") for c in pct_cols]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x, width = np.arange(len(sites)), 0.25

    for i, (regime, col) in enumerate(zip(regimes, pct_cols)):
        vals   = df[col].values
        offset = (i - 1) * width
        bars   = ax.bar(x + offset, vals, width,
                        label=regime,
                        color=REGIME_COLORS.get(regime, "#999999"),
                        alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.4,
                        str(round(val, 1)) + "%",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=12, ha="right")
    ax.set_ylabel("Percentage of daytime hours (%)")
    ax.set_title("Sky Regime Distribution per Site (NSRDB 2018-2022)")
    ax.legend(title="Sky regime", framealpha=0.8)
    ax.set_ylim(0, 108)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_regime_distribution.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("  Saved: " + out)


# -----------------------------------------------------------------------
# FIG 3: Generated scenario examples per regime (fixed x-axis)
# -----------------------------------------------------------------------
def fig3_scenario_examples():
    sc_path = os.path.join(EVAL_DIR, "test_scenarios.npz")
    if not os.path.exists(sc_path):
        # Try calibrated intervals file which also has scenarios
        sc_path = os.path.join(EVAL_DIR, "calibrated_intervals.npz")
    if not os.path.exists(sc_path):
        print("  [skip] scenario file not found")
        return

    data      = np.load(sc_path)
    actuals   = data["actuals"]
    scenarios = data["scenarios"]
    regimes   = data["regimes"]
    hours     = HOUR_LABELS

    regime_names = {0: "Clear", 1: "Cloudy", 2: "Overcast"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

    for ax, (regime_id, regime_name) in zip(axes, regime_names.items()):
        mask = (regimes == regime_id)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            ax.set_title(regime_name + " (no samples)")
            continue

        median_ghi = np.median(actuals[idxs].mean(axis=1))
        chosen_idx = idxs[np.argmin(np.abs(actuals[idxs].mean(axis=1) - median_ghi))]
        actual_day = actuals[chosen_idx]
        sc_day     = scenarios[chosen_idx]
        color      = REGIME_COLORS.get(regime_name, "#999999")

        for s in range(sc_day.shape[0]):
            ax.plot(hours, sc_day[s], color=color, alpha=0.12, lw=0.8)

        q10 = np.quantile(sc_day, 0.10, axis=0)
        q90 = np.quantile(sc_day, 0.90, axis=0)
        ax.fill_between(hours, q10, q90, alpha=0.25, color=color, label="80% PI")

        med = np.median(sc_day, axis=0)
        ax.plot(hours, med, color=color, lw=1.5, linestyle="--", label="Median scenario")
        ax.plot(hours, actual_day, color="black", lw=2.0, label="Actual GHI", zorder=5)

        ax.set_title(regime_name + " sky")
        ax.set_xlabel("Hour of day")
        if ax is axes[0]:
            ax.set_ylabel("GHI (W/m2)")
        ax.set_xlim(hours[0], hours[-1])
        ax.set_ylim(0, max(float(actual_day.max()), float(q90.max())) * 1.15)
        ax.legend(fontsize=8, framealpha=0.8)

    fig.suptitle("GenerativeAURORA: Generated GHI Scenarios by Sky Regime (50 scenarios per day)",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_scenario_examples.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("  Saved: " + out)


# -----------------------------------------------------------------------
# FIG 4: Before vs After RCCC (re-save cleaner version)
# -----------------------------------------------------------------------
def fig4_coverage_comparison():
    results_path = os.path.join(EVAL_DIR, "rccc_results.csv")
    if not os.path.exists(results_path):
        print("  [skip] rccc_results.csv not found")
        return

    df         = pd.read_csv(results_path)
    regime_df  = df[df["group"] == "regime"].copy().reset_index(drop=True)
    regimes    = regime_df["name"].tolist()
    x          = np.arange(len(regimes))
    width      = 0.35
    colors_r   = [REGIME_COLORS.get(r, "#999999") for r in regimes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.bar(x - width / 2, regime_df["coverage_before"], width,
           label="Before RCCC", color=colors_r, alpha=0.45, edgecolor="white")
    ax.bar(x + width / 2, regime_df["coverage_after"],  width,
           label="After RCCC",  color=colors_r, alpha=0.90, edgecolor="white")
    ax.axhline(90, color="#D94F3D", lw=1.2, linestyle="--", label="Target 90%")

    for i, row in regime_df.iterrows():
        ax.text(x[i] - width / 2, row["coverage_before"] + 0.5,
                str(round(row["coverage_before"], 1)) + "%",
                ha="center", va="bottom", fontsize=8.5, color="#555555")
        ax.text(x[i] + width / 2, row["coverage_after"] + 0.5,
                str(round(row["coverage_after"], 1)) + "%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage: Before vs After RCCC")
    ax.set_ylim(0, 107)
    ax.legend(fontsize=9, framealpha=0.8)

    ax2 = axes[1]
    ax2.bar(x - width / 2, regime_df["width_before"], width,
            label="Before RCCC", color=colors_r, alpha=0.45, edgecolor="white")
    ax2.bar(x + width / 2, regime_df["width_after"],  width,
            label="After RCCC",  color=colors_r, alpha=0.90, edgecolor="white")

    for i, row in regime_df.iterrows():
        delta = round(row["delta_coverage"] if "delta_coverage" in row else
                      row["coverage_after"] - row["coverage_before"], 2)
        color_d = "#D94F3D" if delta > 5 else "#3DAE6B"
        ypos = max(row["width_before"], row["width_after"]) + 3
        ax2.text(x[i], ypos, "+" + str(delta) + " pp",
                 ha="center", va="bottom", fontsize=9, color=color_d, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes)
    ax2.set_ylabel("Average interval width (W/m2)")
    ax2.set_title("Interval Width: Before vs After RCCC")
    ax2.legend(fontsize=9, framealpha=0.8)

    fig.suptitle("Regime-Conditioned Conformal Calibration (RCCC): Effect on 90% Prediction Intervals",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_coverage_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("  Saved: " + out)


# -----------------------------------------------------------------------
# FIG 5: Average GHI profiles per site
# -----------------------------------------------------------------------
def fig5_ghi_profiles():
    fpath = find_eda_file("hourly_ghi_profile.csv")
    if fpath is None:
        print("  [skip] hourly_ghi_profile.csv not found")
        return

    df  = pd.read_csv(fpath, index_col=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for col in df.columns:
        site_name = col.strip()
        label     = SITE_LABELS.get(site_name, site_name.replace("_", " "))
        color     = SITE_COLORS.get(site_name, "#999999")
        valid     = df[col].dropna()
        ax.plot(valid.index.astype(int), valid.values,
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
# FIG 6: Method comparison bar chart (CRPS + Coverage)
# -----------------------------------------------------------------------
def fig6_method_comparison():
    table_path = os.path.join(TABLE_DIR, "table2_overall_metrics.csv")
    if not os.path.exists(table_path):
        print("  [skip] table2_overall_metrics.csv not found")
        return

    df = pd.read_csv(table_path, index_col=0)

    methods = list(df.index)
    crps    = df["CRPS"].values.astype(float)
    cov90   = df["Coverage 90%"].values.astype(float)

    x     = np.arange(len(methods))
    width = 0.38

    colors_crps = ["#B0B0B0"] * (len(methods) - 1) + ["#4A90D9"]
    colors_cov  = ["#B0B0B0"] * (len(methods) - 1) + ["#3DAE6B"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    bars = ax.barh(x, crps, color=colors_crps, edgecolor="white", height=0.55)
    for bar, val in zip(bars, crps):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                str(round(val, 2)), va="center", fontsize=9)
    ax.set_yticks(x)
    ax.set_yticklabels([m.replace(" (proposed)", "") for m in methods], fontsize=9)
    ax.set_xlabel("CRPS (W/m2, lower is better)")
    ax.set_title("CRPS Comparison (all methods)")
    ax.invert_yaxis()

    ax2 = axes[1]
    bars2 = ax2.barh(x, cov90, color=colors_cov, edgecolor="white", height=0.55)
    ax2.axvline(90, color="#D94F3D", lw=1.2, linestyle="--", label="Target 90%")
    for bar, val in zip(bars2, cov90):
        ax2.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                 str(round(val, 1)) + "%", va="center", fontsize=9)
    ax2.set_yticks(x)
    ax2.set_yticklabels([m.replace(" (proposed)", "") for m in methods], fontsize=9)
    ax2.set_xlabel("Coverage at 90% PI (%)")
    ax2.set_title("Coverage Comparison (all methods)")
    ax2.legend(fontsize=9)
    ax2.invert_yaxis()

    fig.suptitle("GenerativeAURORA vs Baselines: CRPS and 90% Coverage", fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig6_method_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("  Saved: " + out)


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():
    print("GenerativeAURORA - Step 11: Final Publication Figures")
    print("Output: " + FIG_DIR)
    print("")

    print("Fig 1: Training loss")
    fig1_training_loss()

    print("Fig 2: Regime distribution per site")
    fig2_regime_distribution()

    print("Fig 3: Scenario examples (fixed hour axis)")
    fig3_scenario_examples()

    print("Fig 4: RCCC before vs after")
    fig4_coverage_comparison()

    print("Fig 5: GHI profiles per site")
    fig5_ghi_profiles()

    print("Fig 6: Method comparison bar chart")
    fig6_method_comparison()

    print("")
    print("=== Step 11 complete ===")
    figs = sorted([f for f in os.listdir(FIG_DIR) if f.endswith(".png")])
    for f in figs:
        kb = round(os.path.getsize(os.path.join(FIG_DIR, f)) / 1024, 1)
        print("  " + f.ljust(40) + str(kb) + " KB")
    print("")
    print("Next: run step12_paper_draft.py to generate the full paper draft.")


if __name__ == "__main__":
    main()