# GenerativeAURORA - Step 6: Evaluate and Generate Scenarios
#
# What this does:
#   1. Loads best_model.pt from training
#   2. Generates 50 GHI scenarios per test day per site
#   3. Computes evaluation metrics:
#        - CRPS  (Continuous Ranked Probability Score) -- main metric
#        - Energy Score (multivariate CRPS generalization)
#        - Pinball loss at 10th, 50th, 90th quantiles
#        - Mean scenario MAE vs actual GHI
#        - Coverage at 80% and 90% prediction intervals
#   4. Breaks down all metrics by sky regime (Clear / Cloudy / Overcast)
#   5. Saves generated scenarios and metric tables to Dataset/
#
# Run:
#   python step6_evaluate.py

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
DATASET_DIR = "C:/Users/user/Paper_10/Dataset"
CODING_DIR  = "C:/Users/user/Paper_10/Coding"
OUTPUT_DIR  = os.path.join(DATASET_DIR, "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------
N_SCENARIOS  = 50      # scenarios to generate per test day
BATCH_SIZE   = 32      # batch size during generation
HORIZON      = 16
LOOKBACK     = 16
N_FEATURES   = 10
T_DIFFUSION  = 200

REGIME_NAMES = {0: "Clear", 1: "Cloudy", 2: "Overcast"}
SITE_NAMES   = {0: "Phoenix_AZ", 1: "LosAngeles_CA", 2: "Denver_CO",
                3: "Miami_FL",   4: "Seattle_WA"}

# -----------------------------------------------------------------------
# IMPORT MODEL
# -----------------------------------------------------------------------
import sys
sys.path.insert(0, CODING_DIR)
from step4_build_model import GenerativeAURORA


# -----------------------------------------------------------------------
# DATASET (reuse from step 5)
# -----------------------------------------------------------------------
class SolarSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path):
        data        = np.load(npz_path)
        self.X      = torch.tensor(data["X"],      dtype=torch.float32)
        self.Y      = torch.tensor(data["Y"],      dtype=torch.float32)
        self.regime = torch.tensor(data["regime"], dtype=torch.long)
        self.site   = torch.tensor(data["site"],   dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.regime[idx], self.site[idx]


# -----------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------

def crps_gaussian_approx(actuals, scenarios):
    """
    CRPS approximated from ensemble scenarios using the energy form:
    CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
    actuals  : (N, H)  actual GHI trajectories
    scenarios: (N, S, H) generated scenarios
    Returns  : mean CRPS scalar
    """
    N, S, H = scenarios.shape

    # E|X - y| : mean absolute error between each scenario and actual
    diff_actual = np.abs(scenarios - actuals[:, np.newaxis, :])   # (N, S, H)
    term1 = diff_actual.mean(axis=1)                               # (N, H)

    # E|X - X'| : mean pairwise distance between scenarios
    # Efficient computation: use the identity E|X-X'| = 2 * sum_i sum_j<i |xi - xj| / S^2
    term2_list = []
    for n in range(N):
        sc = scenarios[n]        # (S, H)
        diff = np.abs(sc[:, np.newaxis, :] - sc[np.newaxis, :, :])   # (S, S, H)
        term2_list.append(diff.mean(axis=(0, 1)))                     # (H,)
    term2 = np.stack(term2_list, axis=0)   # (N, H)

    crps_per_step = term1 - 0.5 * term2    # (N, H)
    return float(crps_per_step.mean())


def pinball_loss(actuals, scenarios, quantile):
    """
    Pinball loss at a given quantile.
    actuals  : (N, H)
    scenarios: (N, S, H)
    """
    q_forecast = np.quantile(scenarios, quantile, axis=1)   # (N, H)
    error      = actuals - q_forecast
    loss       = np.where(error >= 0, quantile * error, (quantile - 1) * error)
    return float(loss.mean())


def coverage_and_width(actuals, scenarios, alpha=0.10):
    """
    Empirical coverage and average interval width at (1-alpha) prediction interval.
    alpha=0.10 -> 90% PI,  alpha=0.20 -> 80% PI
    """
    lower = np.quantile(scenarios, alpha / 2,     axis=1)   # (N, H)
    upper = np.quantile(scenarios, 1 - alpha / 2, axis=1)   # (N, H)
    covered = ((actuals >= lower) & (actuals <= upper)).mean()
    width   = (upper - lower).mean()
    return float(covered), float(width)


def mean_scenario_mae(actuals, scenarios):
    """Average MAE between median scenario and actual."""
    median_scenario = np.median(scenarios, axis=1)   # (N, H)
    return float(np.abs(median_scenario - actuals).mean())


def compute_all_metrics(actuals, scenarios, label=""):
    crps  = crps_gaussian_approx(actuals, scenarios)
    pb10  = pinball_loss(actuals, scenarios, 0.10)
    pb50  = pinball_loss(actuals, scenarios, 0.50)
    pb90  = pinball_loss(actuals, scenarios, 0.90)
    cov90, w90 = coverage_and_width(actuals, scenarios, alpha=0.10)
    cov80, w80 = coverage_and_width(actuals, scenarios, alpha=0.20)
    mae   = mean_scenario_mae(actuals, scenarios)

    prefix = (label + " ") if label else ""
    print("  " + prefix + "CRPS          : " + str(round(crps,  6)))
    print("  " + prefix + "Pinball p10   : " + str(round(pb10,  6)))
    print("  " + prefix + "Pinball p50   : " + str(round(pb50,  6)))
    print("  " + prefix + "Pinball p90   : " + str(round(pb90,  6)))
    print("  " + prefix + "Coverage 90%  : " + str(round(cov90 * 100, 2)) + "%")
    print("  " + prefix + "Coverage 80%  : " + str(round(cov80 * 100, 2)) + "%")
    print("  " + prefix + "Interval width: " + str(round(w90,  4)) + "  (90% PI)")
    print("  " + prefix + "Median MAE    : " + str(round(mae,  6)))

    return {
        "crps":         crps,
        "pinball_p10":  pb10,
        "pinball_p50":  pb50,
        "pinball_p90":  pb90,
        "coverage_90":  round(cov90 * 100, 4),
        "coverage_80":  round(cov80 * 100, 4),
        "interval_width_90": round(w90, 6),
        "median_mae":   mae,
        "n_samples":    len(actuals),
    }


# -----------------------------------------------------------------------
# INVERSE SCALE (convert normalized back to W/m2)
# -----------------------------------------------------------------------

def load_scalers(dataset_dir):
    path = os.path.join(dataset_dir, "scaler_params.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inverse_scale_ghi(arr, scalers):
    s_min = scalers["GHI"]["min"]
    s_max = scalers["GHI"]["max"]
    return arr * (s_max - s_min) + s_min


# -----------------------------------------------------------------------
# GENERATION LOOP
# -----------------------------------------------------------------------

def generate_scenarios(model, loader, device, n_scenarios):
    model.eval()

    all_actuals   = []
    all_scenarios = []
    all_regimes   = []
    all_sites     = []

    total_batches = len(loader)
    print("  Generating scenarios... (" + str(total_batches) + " batches)")

    with torch.no_grad():
        for batch_idx, (X, Y, regime, site) in enumerate(loader):
            X      = X.to(device)
            regime = regime.to(device)
            site   = site.to(device)

            B = X.shape[0]

            # Generate: returns (B * n_scenarios, HORIZON)
            sc = model.sample(X, regime, site, n_scenarios=n_scenarios)
            sc = sc.cpu().numpy().reshape(B, n_scenarios, HORIZON)   # (B, S, H)

            all_actuals.append(Y.numpy())                            # (B, H)
            all_scenarios.append(sc)                                 # (B, S, H)
            all_regimes.append(regime.cpu().numpy())                 # (B,)
            all_sites.append(site.cpu().numpy())                     # (B,)

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                print("    Batch " + str(batch_idx + 1) + "/" + str(total_batches))

    actuals   = np.concatenate(all_actuals,   axis=0)   # (N, H)
    scenarios = np.concatenate(all_scenarios, axis=0)   # (N, S, H)
    regimes   = np.concatenate(all_regimes,   axis=0)   # (N,)
    sites     = np.concatenate(all_sites,     axis=0)   # (N,)

    return actuals, scenarios, regimes, sites


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("GenerativeAURORA - Step 6: Evaluate")
    print("")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))
    print("")

    # Load scalers
    scalers = load_scalers(DATASET_DIR)
    ghi_min = scalers["GHI"]["min"]
    ghi_max = scalers["GHI"]["max"]
    print("GHI scale range: " + str(round(ghi_min, 1)) + " to " + str(round(ghi_max, 1)) + " W/m2")
    print("")

    # Load model
    model_path = os.path.join(DATASET_DIR, "best_model.pt")
    print("Loading model from: " + model_path)
    model = GenerativeAURORA(T=T_DIFFUSION).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: " + str(total_params))
    print("")

    # Load test dataset
    test_path = os.path.join(DATASET_DIR, "sequences_test.npz")
    test_ds   = SolarSequenceDataset(test_path)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("Test sequences: " + str(len(test_ds)))
    print("Generating " + str(N_SCENARIOS) + " scenarios per sequence...")
    print("")

    # Generate
    actuals, scenarios, regimes, sites = generate_scenarios(
        model, test_loader, device, N_SCENARIOS
    )

    print("")
    print("Generated shapes:")
    print("  Actuals  : " + str(actuals.shape))
    print("  Scenarios: " + str(scenarios.shape))
    print("")

    # Inverse scale to W/m2 for interpretable metrics
    actuals_wm2   = inverse_scale_ghi(actuals,   scalers)
    scenarios_wm2 = inverse_scale_ghi(scenarios, scalers)

    # -----------------------------------------------------------------------
    # OVERALL METRICS
    # -----------------------------------------------------------------------
    print("=== Overall Test Metrics (W/m2 scale) ===")
    overall_metrics = compute_all_metrics(actuals_wm2, scenarios_wm2, label="Overall")
    print("")

    # -----------------------------------------------------------------------
    # METRICS BY SKY REGIME
    # -----------------------------------------------------------------------
    print("=== Metrics by Sky Regime ===")
    regime_metrics = {}
    for regime_id, regime_name in REGIME_NAMES.items():
        mask = (regimes == regime_id)
        if mask.sum() == 0:
            print("  " + regime_name + ": no samples in test set")
            continue
        print("  " + regime_name + " (" + str(mask.sum()) + " days):")
        m = compute_all_metrics(actuals_wm2[mask], scenarios_wm2[mask], label="")
        regime_metrics[regime_name] = m
        print("")

    # -----------------------------------------------------------------------
    # METRICS BY SITE
    # -----------------------------------------------------------------------
    print("=== Metrics by Site ===")
    site_metrics = {}
    for site_id, site_name in SITE_NAMES.items():
        mask = (sites == site_id)
        if mask.sum() == 0:
            continue
        print("  " + site_name + " (" + str(mask.sum()) + " days):")
        m = compute_all_metrics(actuals_wm2[mask], scenarios_wm2[mask], label="")
        site_metrics[site_name] = m
        print("")

    # -----------------------------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------------------------

    # Save scenario arrays (compressed, for paper figures)
    sc_path = os.path.join(OUTPUT_DIR, "test_scenarios.npz")
    np.savez_compressed(
        sc_path,
        actuals=actuals_wm2,
        scenarios=scenarios_wm2,
        regimes=regimes,
        sites=sites,
    )
    print("Scenarios saved to: " + sc_path)

    # Save metrics as JSON
    all_metrics = {
        "overall":         overall_metrics,
        "by_regime":       regime_metrics,
        "by_site":         site_metrics,
        "n_scenarios":     N_SCENARIOS,
        "ghi_scale_min":   ghi_min,
        "ghi_scale_max":   ghi_max,
    }
    metrics_json = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print("Metrics saved to : " + metrics_json)

    # Save metrics as CSV tables (for paper Table II and Table III)
    rows = []
    for rname, rm in regime_metrics.items():
        row = {"group": "regime", "name": rname}
        row.update(rm)
        rows.append(row)
    for sname, sm in site_metrics.items():
        row = {"group": "site", "name": sname}
        row.update(sm)
        rows.append(row)
    overall_row = {"group": "overall", "name": "All"}
    overall_row.update(overall_metrics)
    rows.append(overall_row)

    metrics_df = pd.DataFrame(rows)
    metrics_csv = os.path.join(OUTPUT_DIR, "metrics_table.csv")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8")
    print("Metrics table    : " + metrics_csv)

    print("")
    print("=== Step 6 complete ===")
    print("Best val loss during training : 0.025614")
    print("Overall test CRPS             : " + str(round(overall_metrics["crps"], 6)))
    print("Overall coverage 90%          : " + str(overall_metrics["coverage_90"]) + "%")
    print("")
    print("Next: run step7_conformal.py to apply regime-conditioned conformal calibration.")


if __name__ == "__main__":
    main()