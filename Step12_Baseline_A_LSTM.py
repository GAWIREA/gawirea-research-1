# GenerativeAURORA - Baseline A (FIXED): Quantile Regression LSTM
#
# Fix applied vs original:
#   The original script computed CRPS on normalized [0,1] scale instead of W/m2.
#   Root cause: quantiles_to_scenarios() was called on normalized q_preds,
#   then inv_scale was applied to scenarios -- but CRPS was already computed.
#   Fix: inv_scale is now applied to q_preds BEFORE scenario reconstruction,
#   so all metrics are in W/m2, consistent with GenerativeAURORA's evaluation.
#
# Run:
#   python baseline_a_quantile_lstm_fixed.py

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
DATASET_DIR = "C:/Users/user/Paper_10/Dataset"
CODING_DIR  = "C:/Users/user/Paper_10/Coding"
EVAL_DIR    = os.path.join(DATASET_DIR, "evaluation")
CKPT_DIR    = os.path.join(DATASET_DIR, "checkpoints_qlstm")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# HYPERPARAMETERS
# -----------------------------------------------------------------------
HORIZON     = 16
LOOKBACK    = 16
N_FEATURES  = 10
BATCH_SIZE  = 128
EPOCHS      = 150
LR          = 1e-3
WEIGHT_DECAY= 1e-5
GRAD_CLIP   = 1.0
N_SCENARIOS = 50
QUANTILES   = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
               0.60, 0.70, 0.80, 0.90, 0.95]

REGIME_NAMES = {0: "Clear", 1: "Cloudy", 2: "Overcast"}
SITE_NAMES   = {0: "Phoenix_AZ", 1: "LosAngeles_CA", 2: "Denver_CO",
                3: "Miami_FL",   4: "Seattle_WA"}


# -----------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------
class SolarSequenceDataset(Dataset):
    def __init__(self, npz_path):
        data        = np.load(npz_path)
        self.X      = torch.tensor(data["X"],      dtype=torch.float32)
        self.Y      = torch.tensor(data["Y"],      dtype=torch.float32)
        self.regime = torch.tensor(data["regime"], dtype=torch.long)
        self.site   = torch.tensor(data["site"],   dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.regime[idx], self.site[idx]


def make_weighted_sampler(dataset):
    labels = dataset.regime.numpy()
    unique, counts = np.unique(labels, return_counts=True)
    freq    = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.array([1.0 / freq[int(r)] for r in labels], dtype=np.float32)
    return WeightedRandomSampler(torch.from_numpy(weights), len(weights), replacement=True)


# -----------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------
class QuantileLSTM(nn.Module):
    def __init__(self, n_features=N_FEATURES, lookback=LOOKBACK,
                 horizon=HORIZON, hidden=128, n_quantiles=len(QUANTILES),
                 n_regimes=3, n_sites=5, embed_dim=32):
        super().__init__()
        self.regime_emb = nn.Embedding(n_regimes, embed_dim)
        self.site_emb   = nn.Embedding(n_sites,   embed_dim)
        self.lstm = nn.LSTM(
            input_size=n_features + embed_dim * 2,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.norm  = nn.LayerNorm(hidden * 2)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden * 2, 128),
                nn.ReLU(),
                nn.Linear(128, horizon),
            )
            for _ in range(n_quantiles)
        ])

    def forward(self, x, regime, site):
        B, L, F = x.shape
        r_e = self.regime_emb(regime).unsqueeze(1).expand(-1, L, -1)
        s_e = self.site_emb(site).unsqueeze(1).expand(-1, L, -1)
        inp = torch.cat([x, r_e, s_e], dim=-1)
        out, _ = self.lstm(inp)
        h      = self.norm(out.mean(1))
        preds  = [head(h) for head in self.heads]
        return torch.stack(preds, dim=1)   # (B, Q, H)


# -----------------------------------------------------------------------
# PINBALL LOSS
# -----------------------------------------------------------------------
def pinball_loss_batch(pred, target, quantiles):
    q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=pred.device)
    target_  = target.unsqueeze(1).expand_as(pred)
    error    = target_ - pred
    loss     = torch.max(
        q_tensor.view(1, -1, 1) * error,
        (q_tensor.view(1, -1, 1) - 1) * error
    )
    return loss.mean()


# -----------------------------------------------------------------------
# SCENARIOS FROM QUANTILES
# -----------------------------------------------------------------------
def quantiles_to_scenarios(q_preds_wm2, n_scenarios=N_SCENARIOS):
    """
    q_preds_wm2: (N, Q, H) already in W/m2 scale
    Returns    : (N, S, H) scenarios in W/m2
    """
    N, Q, H  = q_preds_wm2.shape
    q_levels = np.array(QUANTILES)
    scenarios = np.zeros((N, n_scenarios, H), dtype=np.float32)
    u_samples = np.random.uniform(0, 1, size=(N, n_scenarios))

    for n in range(N):
        for s in range(n_scenarios):
            u = u_samples[n, s]
            for h in range(H):
                scenarios[n, s, h] = float(np.interp(u, q_levels, q_preds_wm2[n, :, h]))

    return np.clip(scenarios, 0, None)


# -----------------------------------------------------------------------
# SCALERS
# -----------------------------------------------------------------------
def load_scalers():
    with open(os.path.join(DATASET_DIR, "scaler_params.json"), "r") as f:
        return json.load(f)

def inv_scale_ghi(arr, scalers):
    """arr can be any shape; applies GHI min-max inverse transform."""
    s = scalers["GHI"]
    return arr * (s["max"] - s["min"]) + s["min"]


# -----------------------------------------------------------------------
# METRICS  (all computed in W/m2)
# -----------------------------------------------------------------------
def crps_score(actuals, scenarios):
    N, S, H = scenarios.shape
    t1 = np.abs(scenarios - actuals[:, np.newaxis, :]).mean(axis=1)
    t2 = np.zeros((N, H))
    for n in range(N):
        sc = scenarios[n]
        t2[n] = np.abs(sc[:, np.newaxis, :] - sc[np.newaxis, :, :]).mean(axis=(0, 1))
    return float((t1 - 0.5 * t2).mean())

def pinball_np(actuals, scenarios, q):
    fc = np.quantile(scenarios, q, axis=1)
    e  = actuals - fc
    return float(np.where(e >= 0, q * e, (q - 1) * e).mean())

def coverage_width(actuals, scenarios, alpha=0.10):
    lo  = np.quantile(scenarios, alpha / 2,     axis=1)
    hi  = np.quantile(scenarios, 1 - alpha / 2, axis=1)
    cov = float(((actuals >= lo) & (actuals <= hi)).mean()) * 100
    wid = float((hi - lo).mean())
    return round(cov, 4), round(wid, 4)

def med_mae(actuals, scenarios):
    return float(np.abs(np.median(scenarios, axis=1) - actuals).mean())

def print_metrics(actuals, scenarios, label=""):
    crps       = crps_score(actuals, scenarios)
    pb10       = pinball_np(actuals, scenarios, 0.10)
    pb50       = pinball_np(actuals, scenarios, 0.50)
    pb90       = pinball_np(actuals, scenarios, 0.90)
    cov90, w90 = coverage_width(actuals, scenarios, 0.10)
    cov80, _   = coverage_width(actuals, scenarios, 0.20)
    mae        = med_mae(actuals, scenarios)
    if label: print("  " + label)
    print("    CRPS         : " + str(round(crps, 4)) + " W/m2")
    print("    Pinball p10  : " + str(round(pb10, 4)) + " W/m2")
    print("    Pinball p50  : " + str(round(pb50, 4)) + " W/m2")
    print("    Pinball p90  : " + str(round(pb90, 4)) + " W/m2")
    print("    Coverage 90% : " + str(cov90) + "%")
    print("    Coverage 80% : " + str(cov80) + "%")
    print("    Width 90% PI : " + str(w90) + " W/m2")
    print("    Median MAE   : " + str(round(mae, 4)) + " W/m2")
    return {"crps": crps, "pinball_p10": pb10, "pinball_p50": pb50,
            "pinball_p90": pb90, "coverage_90": cov90, "coverage_80": cov80,
            "width_90": w90, "median_mae": mae}


# -----------------------------------------------------------------------
# TRAIN / VALIDATE
# -----------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0; n = 0
    for X, Y, regime, site in loader:
        X, Y = X.to(device), Y.to(device)
        regime, site = regime.to(device), site.to(device)
        optimizer.zero_grad()
        pred = model(X, regime, site)
        loss = pinball_loss_batch(pred, Y, QUANTILES)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total += loss.item(); n += 1
    return total / n

def validate_epoch(model, loader, device):
    model.eval()
    total = 0.0; n = 0
    with torch.no_grad():
        for X, Y, regime, site in loader:
            X, Y = X.to(device), Y.to(device)
            regime, site = regime.to(device), site.to(device)
            pred = model(X, regime, site)
            loss = pinball_loss_batch(pred, Y, QUANTILES)
            total += loss.item(); n += 1
    return total / n


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():
    print("GenerativeAURORA - Baseline A (FIXED): Quantile Regression LSTM")
    print("Fix: inverse-scale q_preds to W/m2 BEFORE scenario reconstruction")
    print("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device : " + str(device))
    print("")

    train_ds = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_train.npz"))
    val_ds   = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_val.npz"))
    test_ds  = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_test.npz"))
    print("Train: " + str(len(train_ds)) + "  Val: " + str(len(val_ds)) + "  Test: " + str(len(test_ds)))

    sampler      = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Check if best checkpoint already exists (skip retraining) ----
    best_path = os.path.join(CKPT_DIR, "qlstm_best.pt")

    model = QuantileLSTM().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Parameters: " + str(total_params))

    if os.path.exists(best_path):
        print("Found existing checkpoint: " + best_path)
        print("Loading saved model -- skipping retraining.")
        print("(Delete " + best_path + " to retrain from scratch)")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    else:
        print("")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        best_val  = float("inf")
        log_path  = os.path.join(DATASET_DIR, "training_log_qlstm.csv")

        print("Training for " + str(EPOCHS) + " epochs...")
        print("-" * 55)
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")

        t0 = time.time()
        for epoch in range(EPOCHS):
            tr = train_epoch(model, train_loader, optimizer, device)
            vl = validate_epoch(model, val_loader, device)
            scheduler.step()
            is_best = vl < best_val
            if is_best:
                best_val = vl
                torch.save(model.state_dict(), best_path)
            with open(log_path, "a") as f:
                f.write(str(epoch+1) + "," + str(round(tr,6)) + "," + str(round(vl,6)) + "\n")
            if (epoch + 1) % 10 == 0 or is_best:
                marker = "  <-- best" if is_best else ""
                print("Epoch " + str(epoch+1).rjust(3) + "/" + str(EPOCHS)
                      + "  train: " + str(round(tr,6)).ljust(10)
                      + "  val: "   + str(round(vl,6)).ljust(10) + marker)

        elapsed = round((time.time() - t0) / 60, 1)
        print("-" * 55)
        print("Training complete. Time: " + str(elapsed) + " min")
        print("Best val loss: " + str(round(best_val, 6)))
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    # -----------------------------------------------------------------------
    # EVALUATION  -- THE FIX IS HERE
    # -----------------------------------------------------------------------
    print("")
    print("Evaluating on test set (W/m2 scale)...")
    model.eval()
    scalers = load_scalers()

    all_qpreds, all_actuals, all_regimes, all_sites = [], [], [], []
    with torch.no_grad():
        for X, Y, regime, site in test_loader:
            X = X.to(device); regime = regime.to(device); site = site.to(device)
            q_pred = model(X, regime, site).cpu().numpy()   # (B, Q, H) normalized
            all_qpreds.append(q_pred)
            all_actuals.append(Y.numpy())                   # (B, H)   normalized
            all_regimes.append(regime.cpu().numpy())
            all_sites.append(site.cpu().numpy())

    q_preds_norm = np.concatenate(all_qpreds,  axis=0)   # (N, Q, H) normalized
    actuals_norm = np.concatenate(all_actuals, axis=0)   # (N, H)    normalized
    regimes      = np.concatenate(all_regimes, axis=0)
    sites        = np.concatenate(all_sites,   axis=0)

    # *** FIX: inverse-scale BEFORE building scenarios ***
    # q_preds_norm shape: (N, Q, H)
    # inv_scale_ghi expects any shape -- reshape to (N*Q, H), scale, reshape back
    N, Q, H = q_preds_norm.shape
    q_preds_wm2 = inv_scale_ghi(
        q_preds_norm.reshape(N * Q, H), scalers
    ).reshape(N, Q, H)
    q_preds_wm2 = np.clip(q_preds_wm2, 0, None)

    actuals_wm2 = inv_scale_ghi(actuals_norm, scalers)
    actuals_wm2 = np.clip(actuals_wm2, 0, None)

    # Build scenarios in W/m2
    print("Reconstructing " + str(N_SCENARIOS) + " scenarios from quantile forecasts (W/m2 scale)...")
    scenarios_wm2 = quantiles_to_scenarios(q_preds_wm2, N_SCENARIOS)   # (N, S, H) W/m2

    # Sanity check
    print("  Actuals   mean GHI : " + str(round(float(actuals_wm2.mean()), 1)) + " W/m2  (expect ~400)")
    print("  Scenarios mean GHI : " + str(round(float(scenarios_wm2.mean()), 1)) + " W/m2  (should match)")
    print("  Scenarios 90th pct : " + str(round(float(np.percentile(scenarios_wm2, 90)), 1)) + " W/m2")
    print("")

    # Overall metrics
    print("=== Quantile LSTM (FIXED): Test Metrics (W/m2) ===")
    overall = print_metrics(actuals_wm2, scenarios_wm2, "Overall (920 days, 5 sites)")

    # Per-regime
    print("")
    print("=== By Sky Regime ===")
    regime_metrics = {}
    for rid, rname in REGIME_NAMES.items():
        mask = (regimes == rid)
        if mask.sum() == 0: continue
        print("  " + rname + " (n=" + str(mask.sum()) + ")")
        regime_metrics[rname] = print_metrics(actuals_wm2[mask], scenarios_wm2[mask])

    # Per-site
    print("")
    print("=== By Site ===")
    site_metrics = {}
    for sid, sname in SITE_NAMES.items():
        mask = (sites == sid)
        if mask.sum() == 0: continue
        print("  " + sname + " (n=" + str(mask.sum()) + ")")
        site_metrics[sname] = print_metrics(actuals_wm2[mask], scenarios_wm2[mask])

    # Save (overwrites original qlstm_results.npz with corrected values)
    np.savez_compressed(
        os.path.join(EVAL_DIR, "qlstm_results.npz"),
        actuals=actuals_wm2, scenarios=scenarios_wm2,
        q_preds=q_preds_wm2, regimes=regimes, sites=sites,
    )
    with open(os.path.join(EVAL_DIR, "qlstm_metrics.json"), "w") as f:
        json.dump({"model": "QuantileLSTM_fixed", "overall": overall,
                   "by_regime": regime_metrics, "by_site": site_metrics}, f, indent=2)

    print("")
    print("=== COPY THESE INTO TABLE II ===")
    print("Quantile LSTM  |  CRPS: " + str(round(overall["crps"], 2))
          + "  | Pb p10: " + str(round(overall["pinball_p10"], 2))
          + "  | Pb p50: " + str(round(overall["pinball_p50"], 2))
          + "  | Cov 90%: " + str(overall["coverage_90"])
          + "%  | Width: "  + str(round(overall["width_90"], 1))
          + "  | MAE: "     + str(round(overall["median_mae"], 2)))
    print("")
    print("Results saved to: " + EVAL_DIR)
    print("Now re-run compile_results_table.py to regenerate the final table.")


if __name__ == "__main__":
    main()