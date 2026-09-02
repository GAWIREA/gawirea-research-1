# GenerativeAURORA - Step 7 (Fixed): Regime-Conditioned Conformal Calibration
#
# Root cause of Step 9 failure:
#   Step 9 tried to reconstruct scenarios from calibrated intervals by shifting
#   scenarios by (lower_shift + upper_shift) / 2. This distorted the scenario
#   distribution and destroyed coverage metrics.
#
# Correct approach:
#   - Scenario-based metrics (CRPS, pinball, MAE): computed from RAW scenarios
#   - Interval-based metrics (coverage, width): computed from CALIBRATED intervals
#   - Both are saved together so Step 9 uses them correctly
#
# Run:
#   python step7_conformal.py

import os
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

DATASET_DIR = "C:/Users/user/Paper_10/Dataset"
CODING_DIR  = "C:/Users/user/Paper_10/Coding"
EVAL_DIR    = os.path.join(DATASET_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

N_SCENARIOS  = 50
BATCH_SIZE   = 32
HORIZON      = 16
T_DIFFUSION  = 200
ALPHA        = 0.10

REGIME_NAMES = {0: "Clear", 1: "Cloudy", 2: "Overcast"}
SITE_NAMES   = {0: "Phoenix_AZ", 1: "LosAngeles_CA", 2: "Denver_CO",
                3: "Miami_FL",   4: "Seattle_WA"}

import sys
sys.path.insert(0, CODING_DIR)
from step4_build_model import GenerativeAURORA


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


def generate_split(model, loader, device, scalers, n_sc, label):
    model.eval()
    all_act, all_sc, all_reg, all_sit = [], [], [], []
    print("  Generating " + label + "...")
    with torch.no_grad():
        for X, Y, regime, site in loader:
            X = X.to(device); regime = regime.to(device); site = site.to(device)
            B  = X.shape[0]
            sc = model.sample(X, regime, site, n_scenarios=n_sc)
            sc = sc.cpu().numpy().reshape(B, n_sc, HORIZON)
            all_act.append(Y.numpy())
            all_sc.append(sc)
            all_reg.append(regime.cpu().numpy())
            all_sit.append(site.cpu().numpy())

    act = np.concatenate(all_act, axis=0)
    sc  = np.concatenate(all_sc,  axis=0)
    reg = np.concatenate(all_reg, axis=0)
    sit = np.concatenate(all_sit, axis=0)
    print("    actuals: " + str(act.shape) + "  scenarios: " + str(sc.shape))
    return inv_scale(act, scalers), inv_scale(sc, scalers), reg, sit


def nonconformity_scores(actuals, scenarios):
    lower = np.quantile(scenarios, ALPHA / 2,     axis=1)
    upper = np.quantile(scenarios, 1 - ALPHA / 2, axis=1)
    score = (np.maximum(lower - actuals, 0) + np.maximum(actuals - upper, 0)).mean(axis=1)
    return score, lower, upper


def calibrate(val_act, val_sc, val_reg):
    scores, _, _ = nonconformity_scores(val_act, val_sc)
    cal = {}
    print("  Per-regime calibration quantiles:")
    for rid, rname in REGIME_NAMES.items():
        mask = (val_reg == rid)
        n    = mask.sum()
        if n == 0:
            cal[rid] = None
            continue
        rs     = scores[mask]
        n_conf = min(int(np.ceil((n + 1) * (1 - ALPHA))), n)
        q      = float(np.sort(rs)[n_conf - 1])
        cal[rid] = q
        print("    " + rname.ljust(10) + " n=" + str(n).rjust(4)
              + "  q=" + str(round(q, 4))
              + "  mean_score=" + str(round(float(rs.mean()), 4)))
    global_q = float(np.quantile(scores, 1 - ALPHA))
    for rid in cal:
        if cal[rid] is None:
            cal[rid] = global_q
    return cal


def apply_cal(actuals, scenarios, regimes, cal):
    lower_raw = np.quantile(scenarios, ALPHA / 2,     axis=1)
    upper_raw = np.quantile(scenarios, 1 - ALPHA / 2, axis=1)
    lower_cal = np.copy(lower_raw)
    upper_cal = np.copy(upper_raw)
    for rid, q in cal.items():
        mask = (regimes == rid)
        if mask.sum() == 0:
            continue
        lower_cal[mask] -= q
        upper_cal[mask] += q
    return lower_raw, upper_raw, np.maximum(lower_cal, 0), np.maximum(upper_cal, 0)


def cov_width(actuals, lower, upper):
    cov = float(((actuals >= lower) & (actuals <= upper)).mean()) * 100
    wid = float((upper - lower).mean())
    return round(cov, 4), round(wid, 4)


def crps_score(actuals, scenarios):
    N, S, H = scenarios.shape
    t1 = np.abs(scenarios - actuals[:, np.newaxis, :]).mean(axis=1)
    t2 = np.zeros((N, H))
    for n in range(N):
        sc = scenarios[n]
        t2[n] = np.abs(sc[:, np.newaxis, :] - sc[np.newaxis, :, :]).mean(axis=(0, 1))
    return float((t1 - 0.5 * t2).mean())


def pinball(actuals, scenarios, q):
    fc = np.quantile(scenarios, q, axis=1)
    e  = actuals - fc
    return float(np.where(e >= 0, q * e, (q - 1) * e).mean())


def med_mae(actuals, scenarios):
    return float(np.abs(np.median(scenarios, axis=1) - actuals).mean())


def main():
    print("GenerativeAURORA - Step 7 (Fixed): RCCC")
    print("")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scalers = load_scalers()
    print("Device: " + str(device))
    print("")

    model = GenerativeAURORA(T=T_DIFFUSION).to(device)
    model.load_state_dict(torch.load(
        os.path.join(DATASET_DIR, "best_model.pt"),
        map_location=device, weights_only=True))
    model.eval()
    print("Model loaded.")
    print("")

    val_ds  = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_val.npz"))
    test_ds = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_test.npz"))
    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("--- 7A: Generate validation scenarios ---")
    val_act, val_sc, val_reg, val_sit = generate_split(
        model, val_loader, device, scalers, N_SCENARIOS, "validation")
    print("")

    print("--- 7B: Compute calibration quantiles ---")
    cal = calibrate(val_act, val_sc, val_reg)
    with open(os.path.join(EVAL_DIR, "rccc_calibration.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in cal.items()}, f, indent=2)
    print("")

    print("--- 7C: Generate test scenarios ---")
    test_act, test_sc, test_reg, test_sit = generate_split(
        model, test_loader, device, scalers, N_SCENARIOS, "test")
    print("")

    print("--- 7D: Apply calibration and evaluate ---")
    lower_raw, upper_raw, lower_cal, upper_cal = apply_cal(
        test_act, test_sc, test_reg, cal)

    # Scenario-based metrics (not affected by RCCC -- uses raw scenarios)
    sc_metrics = {
        "CRPS":        round(crps_score(test_act, test_sc), 4),
        "Pinball_p10": round(pinball(test_act, test_sc, 0.10), 4),
        "Pinball_p50": round(pinball(test_act, test_sc, 0.50), 4),
        "Pinball_p90": round(pinball(test_act, test_sc, 0.90), 4),
        "Median_MAE":  round(med_mae(test_act, test_sc), 4),
    }

    print("")
    print("Scenario-based metrics (GenerativeAURORA raw, 50 scenarios):")
    for k, v in sc_metrics.items():
        print("  " + k.ljust(15) + str(v))

    # Interval-based metrics before and after RCCC
    results = []
    groups = [
        ("overall", "All", np.ones(len(test_reg), dtype=bool)),
    ]
    for rid, rname in REGIME_NAMES.items():
        groups.append(("regime", rname, test_reg == rid))
    for sid, sname in SITE_NAMES.items():
        groups.append(("site", sname, test_sit == sid))

    print("")
    print("Coverage and width (Before vs After RCCC):")
    print("")

    for group, name, mask in groups:
        n = mask.sum()
        if n == 0:
            continue
        cr, wr = cov_width(test_act[mask], lower_raw[mask], upper_raw[mask])
        cc, wc = cov_width(test_act[mask], lower_cal[mask], upper_cal[mask])
        print("  " + (group + "/" + name).ljust(22)
              + " n=" + str(n).rjust(4)
              + "  Before: " + str(cr) + "% / " + str(wr) + " W/m2"
              + "  After: "  + str(cc) + "% / " + str(wc) + " W/m2"
              + "  delta: +" + str(round(cc - cr, 2)) + " pp")
        results.append({
            "group": group, "name": name, "n": int(n),
            "coverage_before": cr, "width_before": wr,
            "coverage_after":  cc, "width_after":  wc,
            "delta_coverage":  round(cc - cr, 2),
        })

    # Save everything
    pd.DataFrame(results).to_csv(
        os.path.join(EVAL_DIR, "rccc_results.csv"), index=False, encoding="utf-8")

    np.savez_compressed(
        os.path.join(EVAL_DIR, "calibrated_intervals.npz"),
        actuals=test_act, scenarios=test_sc,
        lower_raw=lower_raw, upper_raw=upper_raw,
        lower_cal=lower_cal, upper_cal=upper_cal,
        regimes=test_reg, sites=test_sit,
    )

    with open(os.path.join(EVAL_DIR, "scenario_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(sc_metrics, f, indent=2)

    print("")
    print("Files saved to: " + EVAL_DIR)
    print("")
    print("=== Key result for paper (Table IV) ===")
    for row in results:
        if row["group"] == "regime":
            sign = "+" if row["delta_coverage"] >= 0 else ""
            print("  " + row["name"].ljust(10)
                  + " coverage: " + str(row["coverage_before"])
                  + "% -> " + str(row["coverage_after"])
                  + "%  (" + sign + str(row["delta_coverage"]) + " pp)")
    print("")
    print("Next: run step9_tables.py")


if __name__ == "__main__":
    main()