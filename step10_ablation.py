# GenerativeAURORA - Step 10: Ablation Study
#
# Tests the contribution of each model component systematically.
# This produces Table VI in the paper (ablation table).
#
# Ablation variants tested:
#   A1. Full model + RCCC          (GenerativeAURORA + RCCC) -- proposed
#   A2. Full model, no RCCC        (GenerativeAURORA)        -- ablate RCCC
#   A3. No regime conditioning     (DDPM no regime)          -- ablate sky gating
#   A4. No condition encoder       (blind DDPM)              -- ablate weather context
#   A5. No site embedding          (site-agnostic DDPM)      -- ablate site identity
#   A6. Gaussian baseline          (trivial)                 -- lower bound
#
# For each variant we report:
#   CRPS, Coverage 90% (overall), Coverage 90% Overcast, Median MAE
#
# Run:
#   python step10_ablation.py

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
from torch.utils.data import DataLoader

DATASET_DIR = "C:/Users/user/Paper_10/Dataset"
CODING_DIR  = "C:/Users/user/Paper_10/Coding"
EVAL_DIR    = os.path.join(DATASET_DIR, "evaluation")
TABLE_DIR   = os.path.join(DATASET_DIR, "paper_tables")
os.makedirs(TABLE_DIR, exist_ok=True)

N_SCENARIOS = 50
BATCH_SIZE  = 32
HORIZON     = 16
LOOKBACK    = 16
N_FEATURES  = 10
N_REGIMES   = 3
N_SITES     = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 128
T_DIFFUSION = 200
ALPHA       = 0.10

REGIME_NAMES = {0: "Clear", 1: "Cloudy", 2: "Overcast"}

import sys
sys.path.insert(0, CODING_DIR)
from step4_build_model import (
    GenerativeAURORA, DiffusionUNet, TimestepEmbedding,
    ConditionEncoder, ResBlock1D, make_beta_schedule,
    HORIZON, LOOKBACK, N_FEATURES, N_REGIMES, N_SITES,
    EMBED_DIM, HIDDEN_DIM, T_DIFFUSION,
)


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


def load_scalers():
    with open(os.path.join(DATASET_DIR, "scaler_params.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def inv_scale(arr, scalers):
    s = scalers["GHI"]
    return arr * (s["max"] - s["min"]) + s["min"]


# -----------------------------------------------------------------------
# ABLATION VARIANT: NO CONDITION ENCODER
# Replace ConditionEncoder output with zeros -- model gets no weather context
# -----------------------------------------------------------------------
class BlindConditionEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        B = x.shape[0]
        return torch.zeros(B, self.embed_dim, device=x.device)


class GenerativeAURORA_NoContext(GenerativeAURORA):
    def __init__(self, T=T_DIFFUSION):
        super().__init__(T=T)
        self.unet.cond_enc = BlindConditionEncoder(embed_dim=EMBED_DIM)


# -----------------------------------------------------------------------
# ABLATION VARIANT: NO SITE EMBEDDING
# Replace site embedding with zeros -- model is site-agnostic
# -----------------------------------------------------------------------
class GenerativeAURORA_NoSite(GenerativeAURORA):
    def __init__(self, T=T_DIFFUSION):
        super().__init__(T=T)

    def forward(self, x0, cond_x, regime, site):
        site_zeros = torch.zeros_like(site)
        return super().forward(x0, cond_x, regime, site_zeros)

    @torch.no_grad()
    def sample(self, cond_x, regime, site, n_scenarios=50):
        site_zeros = torch.zeros_like(site)
        return super().sample(cond_x, regime, site_zeros, n_scenarios)


# -----------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------
def crps_score(actuals, scenarios):
    N, S, H = scenarios.shape
    t1 = np.abs(scenarios - actuals[:, np.newaxis, :]).mean(axis=1)
    t2 = np.zeros((N, H))
    for n in range(N):
        sc = scenarios[n]
        t2[n] = np.abs(sc[:, np.newaxis, :] - sc[np.newaxis, :, :]).mean(axis=(0, 1))
    return float((t1 - 0.5 * t2).mean())


def coverage_from_scenarios(actuals, scenarios, alpha=ALPHA):
    lower = np.quantile(scenarios, alpha / 2,     axis=1)
    upper = np.quantile(scenarios, 1 - alpha / 2, axis=1)
    return float(((actuals >= lower) & (actuals <= upper)).mean()) * 100


def coverage_from_intervals(actuals, lower, upper):
    return float(((actuals >= lower) & (actuals <= upper)).mean()) * 100


def med_mae(actuals, scenarios):
    return float(np.abs(np.median(scenarios, axis=1) - actuals).mean())


def width_from_scenarios(actuals, scenarios, alpha=ALPHA):
    lower = np.quantile(scenarios, alpha / 2,     axis=1)
    upper = np.quantile(scenarios, 1 - alpha / 2, axis=1)
    return float((upper - lower).mean())


# -----------------------------------------------------------------------
# GENERATE SCENARIOS FOR A MODEL
# -----------------------------------------------------------------------
def generate(model, loader, device, scalers, n_sc, regime_override=None, site_override=None):
    model.eval()
    all_act, all_sc, all_reg, all_sit = [], [], [], []

    with torch.no_grad():
        for X, Y, regime, site in loader:
            X      = X.to(device)
            B      = X.shape[0]

            r = torch.zeros(B, dtype=torch.long, device=device) if regime_override == 0 \
                else regime.to(device)
            s = torch.zeros(B, dtype=torch.long, device=device) if site_override == 0 \
                else site.to(device)

            sc = model.sample(X, r, s, n_scenarios=n_sc)
            sc = sc.cpu().numpy().reshape(B, n_sc, HORIZON)

            all_act.append(Y.numpy())
            all_sc.append(sc)
            all_reg.append(regime.numpy())
            all_sit.append(site.numpy())

    act = inv_scale(np.concatenate(all_act, axis=0), scalers)
    sc  = inv_scale(np.concatenate(all_sc,  axis=0), scalers)
    reg = np.concatenate(all_reg, axis=0)
    sit = np.concatenate(all_sit, axis=0)
    return act, sc, reg, sit


def ablation_metrics(act, sc, reg, cal_lower=None, cal_upper=None):
    crps     = round(crps_score(act, sc), 4)
    cov_all  = round(coverage_from_scenarios(act, sc), 2)
    mae      = round(med_mae(act, sc), 4)
    wid      = round(width_from_scenarios(act, sc), 2)

    # Overcast coverage
    mask_ov  = (reg == 2)
    if mask_ov.sum() > 0:
        if cal_lower is not None and cal_upper is not None:
            cov_ov = round(coverage_from_intervals(act[mask_ov], cal_lower[mask_ov], cal_upper[mask_ov]), 2)
        else:
            cov_ov = round(coverage_from_scenarios(act[mask_ov], sc[mask_ov]), 2)
    else:
        cov_ov = float("nan")

    # Cloudy coverage
    mask_cl = (reg == 1)
    if mask_cl.sum() > 0:
        if cal_lower is not None and cal_upper is not None:
            cov_cl = round(coverage_from_intervals(act[mask_cl], cal_lower[mask_cl], cal_upper[mask_cl]), 2)
        else:
            cov_cl = round(coverage_from_scenarios(act[mask_cl], sc[mask_cl]), 2)
    else:
        cov_cl = float("nan")

    return {
        "CRPS":               crps,
        "Coverage 90% (All)": cov_all if cal_lower is None else round(
            coverage_from_intervals(act, cal_lower, cal_upper), 2),
        "Coverage (Cloudy)":  cov_cl,
        "Coverage (Overcast)": cov_ov,
        "Width 90% PI":       wid if cal_lower is None else round(
            float((cal_upper - cal_lower).mean()), 2),
        "Median MAE":         mae,
    }


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():
    print("GenerativeAURORA - Step 10: Ablation Study")
    print("")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scalers = load_scalers()
    np.random.seed(42)
    print("Device: " + str(device))
    print("")

    test_ds = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_test.npz"))
    train_ds = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_train.npz"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model_path = os.path.join(DATASET_DIR, "best_model.pt")

    # Load calibrated intervals from Step 7
    cal_data  = np.load(os.path.join(EVAL_DIR, "calibrated_intervals.npz"))
    l_cal     = cal_data["lower_cal"]
    u_cal     = cal_data["upper_cal"]
    reg_cal   = cal_data["regimes"]

    results = {}

    # -----------------------------------------------------------------------
    # A1: Full model + RCCC (proposed)
    # -----------------------------------------------------------------------
    print("A1: GenerativeAURORA + RCCC (proposed)...")
    model = GenerativeAURORA(T=T_DIFFUSION).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    act, sc, reg, sit = generate(model, test_loader, device, scalers, N_SCENARIOS)
    results["A1: Full + RCCC (proposed)"] = ablation_metrics(act, sc, reg, l_cal, u_cal)
    print("  CRPS=" + str(results["A1: Full + RCCC (proposed)"]["CRPS"])
          + "  Overcast cov=" + str(results["A1: Full + RCCC (proposed)"]["Coverage (Overcast)"]) + "%")

    # -----------------------------------------------------------------------
    # A2: Full model, no RCCC
    # -----------------------------------------------------------------------
    print("A2: GenerativeAURORA (no RCCC)...")
    results["A2: Full, no RCCC"] = ablation_metrics(act, sc, reg)
    print("  CRPS=" + str(results["A2: Full, no RCCC"]["CRPS"])
          + "  Overcast cov=" + str(results["A2: Full, no RCCC"]["Coverage (Overcast)"]) + "%")

    # -----------------------------------------------------------------------
    # A3: No regime conditioning (force all regime=0)
    # -----------------------------------------------------------------------
    print("A3: DDPM no regime conditioning...")
    act_nr, sc_nr, reg_nr, _ = generate(model, test_loader, device, scalers, N_SCENARIOS,
                                         regime_override=0)
    results["A3: No regime conditioning"] = ablation_metrics(act_nr, sc_nr, reg_nr)
    print("  CRPS=" + str(results["A3: No regime conditioning"]["CRPS"])
          + "  Overcast cov=" + str(results["A3: No regime conditioning"]["Coverage (Overcast)"]) + "%")

    # -----------------------------------------------------------------------
    # A4: No condition encoder (blind -- no weather context)
    # -----------------------------------------------------------------------
    print("A4: No condition encoder (blind DDPM)...")
    model_blind = GenerativeAURORA_NoContext(T=T_DIFFUSION).to(device)
    model_blind.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True),
        strict=False   # blind encoder has different architecture, load shared weights only
    )
    act_b, sc_b, reg_b, _ = generate(model_blind, test_loader, device, scalers, N_SCENARIOS)
    results["A4: No condition encoder"] = ablation_metrics(act_b, sc_b, reg_b)
    print("  CRPS=" + str(results["A4: No condition encoder"]["CRPS"])
          + "  Overcast cov=" + str(results["A4: No condition encoder"]["Coverage (Overcast)"]) + "%")

    # -----------------------------------------------------------------------
    # A5: No site embedding
    # -----------------------------------------------------------------------
    print("A5: No site embedding...")
    model_ns = GenerativeAURORA_NoSite(T=T_DIFFUSION).to(device)
    model_ns.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    act_ns, sc_ns, reg_ns, _ = generate(model_ns, test_loader, device, scalers, N_SCENARIOS)
    results["A5: No site embedding"] = ablation_metrics(act_ns, sc_ns, reg_ns)
    print("  CRPS=" + str(results["A5: No site embedding"]["CRPS"])
          + "  Overcast cov=" + str(results["A5: No site embedding"]["Coverage (Overcast)"]) + "%")

    # -----------------------------------------------------------------------
    # A6: Gaussian baseline (lower bound)
    # -----------------------------------------------------------------------
    print("A6: Gaussian baseline...")
    train_Y_wm2 = inv_scale(train_ds.Y.numpy(), scalers)
    test_Y_wm2  = inv_scale(test_ds.Y.numpy(),  scalers)
    mu  = train_Y_wm2.mean(axis=0)
    sig = np.maximum(train_Y_wm2.std(axis=0), 1.0)
    N   = len(test_Y_wm2)
    sc_g = np.random.normal(
        mu[np.newaxis, np.newaxis, :],
        sig[np.newaxis, np.newaxis, :],
        size=(N, N_SCENARIOS, HORIZON)
    ).clip(0, None).astype(np.float32)
    results["A6: Gaussian baseline"] = ablation_metrics(test_Y_wm2, sc_g, test_ds.regime.numpy())
    print("  CRPS=" + str(results["A6: Gaussian baseline"]["CRPS"])
          + "  Overcast cov=" + str(results["A6: Gaussian baseline"]["Coverage (Overcast)"]) + "%")

    # -----------------------------------------------------------------------
    # BUILD ABLATION TABLE
    # -----------------------------------------------------------------------
    print("")
    print("=== Ablation Table (Table VI) ===")
    table_df = pd.DataFrame(results).T
    print(table_df.to_string())

    # Save CSV
    csv_path = os.path.join(TABLE_DIR, "table6_ablation.csv")
    table_df.to_csv(csv_path, encoding="utf-8")
    print("")
    print("Saved: " + csv_path)

    # Save LaTeX
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Ablation study. Each row removes one component from the full GenerativeAURORA + RCCC model. "
        "Best result per column in bold. Overcast coverage is the key diagnostic metric.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Variant & CRPS & Cov. 90\\% & Cov. Cloudy & Cov. Overcast & Width & MAE \\\\",
        "\\midrule",
    ]
    for name, row in results.items():
        short_name = name.split(":")[1].strip()
        is_proposed = "proposed" in name
        prefix = "\\textbf{" if is_proposed else ""
        suffix = "}" if is_proposed else ""
        line = (prefix + short_name + suffix + " & "
                + prefix + str(row["CRPS"]) + suffix + " & "
                + prefix + str(row["Coverage 90% (All)"]) + "\\%" + suffix + " & "
                + prefix + str(row["Coverage (Cloudy)"]) + "\\%" + suffix + " & "
                + prefix + str(row["Coverage (Overcast)"]) + "\\%" + suffix + " & "
                + prefix + str(row["Width 90% PI"]) + suffix + " & "
                + prefix + str(row["Median MAE"]) + suffix + " \\\\")
        lines.append(line)

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_content = "\n".join(lines)

    tex_path = os.path.join(TABLE_DIR, "table6_ablation.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print("Saved: " + tex_path)

    # -----------------------------------------------------------------------
    # PRINT PAPER-READY SUMMARY
    # -----------------------------------------------------------------------
    print("")
    print("=== Paper narrative (Section V, ablation paragraph) ===")
    print("")
    a1 = results["A1: Full + RCCC (proposed)"]
    a2 = results["A2: Full, no RCCC"]
    a3 = results["A3: No regime conditioning"]
    a4 = results["A4: No condition encoder"]

    print("Removing RCCC (A2 vs A1):")
    print("  Overall coverage: " + str(a2["Coverage 90% (All)"]) + "% vs "
          + str(a1["Coverage 90% (All)"]) + "%")
    print("  Overcast coverage: " + str(a2["Coverage (Overcast)"]) + "% vs "
          + str(a1["Coverage (Overcast)"]) + "%")
    delta_ov = round(a1["Coverage (Overcast)"] - a2["Coverage (Overcast)"], 2)
    print("  Overcast gain from RCCC: +" + str(delta_ov) + " pp")

    print("")
    print("Removing regime conditioning (A3 vs A2):")
    delta_crps = round(a2["CRPS"] - a3["CRPS"], 4)
    sign = "+" if delta_crps >= 0 else ""
    print("  CRPS change: " + sign + str(delta_crps)
          + "  (small because 93% data is Clear sky)")

    print("")
    print("Removing condition encoder (A4 vs A2):")
    delta_crps_blind = round(a2["CRPS"] - a4["CRPS"], 4)
    sign2 = "+" if delta_crps_blind >= 0 else ""
    print("  CRPS change: " + sign2 + str(delta_crps_blind)
          + "  (weather context contribution)")

    print("")
    print("=== Step 10 complete ===")
    print("Next: run step11_final_figures.py")


if __name__ == "__main__":
    main()