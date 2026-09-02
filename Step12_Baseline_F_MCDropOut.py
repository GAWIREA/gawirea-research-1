# GenerativeAURORA - Baseline F: Monte Carlo Dropout LSTM
#
# MC Dropout approximates Bayesian uncertainty by keeping dropout active
# at inference time and sampling multiple forward passes as scenarios.
# This is a widely used probabilistic baseline in energy forecasting papers.
#
# Known weakness: underestimates uncertainty under distribution shift
# (e.g. rare Overcast days not seen during training) -- exactly where
# GenerativeAURORA + RCCC excels.
#
# Architecture:
#   2-layer BiLSTM with dropout=0.3 (kept ON at inference)
#   Regime + site embeddings (same as GenerativeAURORA)
#   Output: point prediction per forward pass
#   Scenarios: 50 stochastic forward passes = 50 scenario samples
#
# Run:
#   python baseline_f_mc_dropout.py

import os
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
EVAL_DIR    = os.path.join(DATASET_DIR, "evaluation")
CKPT_DIR    = os.path.join(DATASET_DIR, "checkpoints_mcdropout")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# HYPERPARAMETERS
# -----------------------------------------------------------------------
HORIZON     = 16
LOOKBACK    = 16
N_FEATURES  = 10
EMBED_DIM   = 32
HIDDEN_DIM  = 128
DROPOUT_P   = 0.3     # kept active at inference
BATCH_SIZE  = 128
EPOCHS      = 150
LR          = 1e-3
WEIGHT_DECAY= 1e-5
GRAD_CLIP   = 1.0
N_SCENARIOS = 50      # = number of MC forward passes

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
# MODEL: MC Dropout LSTM
# -----------------------------------------------------------------------
class MCDropoutLSTM(nn.Module):
    def __init__(self, n_features=N_FEATURES, lookback=LOOKBACK,
                 horizon=HORIZON, hidden=HIDDEN_DIM, embed_dim=EMBED_DIM,
                 dropout_p=DROPOUT_P, n_regimes=3, n_sites=5):
        super().__init__()
        self.dropout_p  = dropout_p
        self.regime_emb = nn.Embedding(n_regimes, embed_dim)
        self.site_emb   = nn.Embedding(n_sites,   embed_dim)

        # BiLSTM -- note: nn.LSTM dropout only applies between layers,
        # so we add explicit dropout layers for MC Dropout at output
        self.lstm = nn.LSTM(
            input_size  = n_features + embed_dim * 2,
            hidden_size = hidden,
            num_layers  = 2,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout_p,   # between LSTM layers
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.norm    = nn.LayerNorm(hidden * 2)
        self.head    = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),   # MC Dropout: also in head
            nn.Linear(hidden, horizon),
            nn.Sigmoid(),
        )

    def forward(self, x, regime, site):
        B, L, F = x.shape
        r_e = self.regime_emb(regime).unsqueeze(1).expand(-1, L, -1)
        s_e = self.site_emb(site).unsqueeze(1).expand(-1, L, -1)
        inp = torch.cat([x, r_e, s_e], dim=-1)
        out, _ = self.lstm(inp)
        h      = self.dropout(self.norm(out.mean(1)))   # MC Dropout here
        return self.head(h)   # (B, H)

    def mc_sample(self, x, regime, site, n_scenarios=N_SCENARIOS):
        """
        Keep model in TRAIN mode to activate dropout.
        Run n_scenarios forward passes to get scenario ensemble.
        """
        self.train()   # CRITICAL: keeps dropout active
        scenarios = []
        with torch.no_grad():
            for _ in range(n_scenarios):
                pred = self.forward(x, regime, site)   # (B, H)
                scenarios.append(pred.unsqueeze(1))     # (B, 1, H)
        return torch.cat(scenarios, dim=1)              # (B, S, H)


# -----------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------
def load_scalers():
    with open(os.path.join(DATASET_DIR, "scaler_params.json"), "r") as f:
        return json.load(f)

def inv_scale_ghi(arr, scalers):
    s = scalers["GHI"]
    return arr * (s["max"] - s["min"]) + s["min"]

def crps_score(actuals, scenarios):
    N, S, H = scenarios.shape
    t1 = np.abs(scenarios - actuals[:, np.newaxis, :]).mean(axis=1)
    t2 = np.zeros((N, H))
    for n in range(N):
        sc = scenarios[n]
        t2[n] = np.abs(sc[:, np.newaxis, :] - sc[np.newaxis, :, :]).mean(axis=(0,1))
    return float((t1 - 0.5 * t2).mean())

def pinball_np(actuals, scenarios, q):
    fc = np.quantile(scenarios, q, axis=1)
    e  = actuals - fc
    return float(np.where(e >= 0, q*e, (q-1)*e).mean())

def coverage_width(actuals, scenarios, alpha=0.10):
    lo  = np.quantile(scenarios, alpha/2,   axis=1)
    hi  = np.quantile(scenarios, 1-alpha/2, axis=1)
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
# MAIN
# -----------------------------------------------------------------------
def main():
    print("GenerativeAURORA - Baseline F: MC Dropout LSTM")
    print("Uncertainty via " + str(N_SCENARIOS) + " stochastic forward passes")
    print("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device : " + str(device))

    train_ds = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_train.npz"))
    val_ds   = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_val.npz"))
    test_ds  = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_test.npz"))
    print("Train: " + str(len(train_ds)) + "  Val: " + str(len(val_ds)) + "  Test: " + str(len(test_ds)))

    sampler      = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MCDropoutLSTM().to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print("Parameters: " + str(total_p))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val  = float("inf")
    best_path = os.path.join(CKPT_DIR, "mcdropout_best.pt")
    log_path  = os.path.join(DATASET_DIR, "training_log_mcdropout.csv")

    print("\nTraining for " + str(EPOCHS) + " epochs (MSE loss)...")
    print("-" * 55)

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        tr_total = 0.0; n = 0
        for X, Y, regime, site in train_loader:
            X, Y   = X.to(device), Y.to(device)
            regime = regime.to(device); site = site.to(device)
            pred   = model(X, regime, site)
            loss   = F.mse_loss(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            tr_total += loss.item(); n += 1
        tr = tr_total / n

        model.eval()
        vl_total = 0.0; n = 0
        with torch.no_grad():
            for X, Y, regime, site in val_loader:
                X, Y   = X.to(device), Y.to(device)
                regime = regime.to(device); site = site.to(device)
                pred   = model(X, regime, site)
                loss   = F.mse_loss(pred, Y)
                vl_total += loss.item(); n += 1
        vl = vl_total / n

        scheduler.step()
        is_best = vl < best_val
        if is_best:
            best_val = vl
            torch.save(model.state_dict(), best_path)

        with open(log_path, "a") as f:
            f.write(str(epoch+1) + "," + str(round(tr,6)) + "," + str(round(vl,6)) + "\n")

        if (epoch+1) % 10 == 0 or is_best:
            marker = "  <-- best" if is_best else ""
            print("Epoch " + str(epoch+1).rjust(3) + "/" + str(EPOCHS)
                  + "  train: " + str(round(tr,6)).ljust(10)
                  + "  val: "   + str(round(vl,6)).ljust(10) + marker)

    elapsed = round((time.time() - t0) / 60, 1)
    print("-" * 55)
    print("Training complete. Time: " + str(elapsed) + " min")
    print("Best val loss: " + str(round(best_val, 6)))
    print("")

    # Evaluation with MC Dropout
    print("Evaluating with " + str(N_SCENARIOS) + " MC Dropout forward passes...")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    # NOTE: model.train() is called inside mc_sample() to keep dropout active
    scalers = load_scalers()

    all_sc, all_act, all_reg, all_sit = [], [], [], []
    for X, Y, regime, site in test_loader:
        X      = X.to(device)
        regime = regime.to(device); site = site.to(device)
        sc     = model.mc_sample(X, regime, site, N_SCENARIOS).cpu().numpy()
        all_sc.append(sc)
        all_act.append(Y.numpy())
        all_reg.append(regime.cpu().numpy())
        all_sit.append(site.cpu().numpy())

    scenarios = np.concatenate(all_sc,  axis=0)
    actuals   = np.concatenate(all_act, axis=0)
    regimes   = np.concatenate(all_reg, axis=0)
    sites     = np.concatenate(all_sit, axis=0)

    actuals_wm2   = inv_scale_ghi(actuals,                   scalers)
    scenarios_wm2 = inv_scale_ghi(np.clip(scenarios, 0, 1), scalers)

    print("\n=== MC Dropout LSTM: Test Metrics ===")
    overall = print_metrics(actuals_wm2, scenarios_wm2, "Overall (920 days, 5 sites)")

    print("\n=== By Sky Regime ===")
    regime_metrics = {}
    for rid, rname in REGIME_NAMES.items():
        mask = (regimes == rid)
        if mask.sum() == 0: continue
        print("  " + rname + " (n=" + str(mask.sum()) + ")")
        regime_metrics[rname] = print_metrics(actuals_wm2[mask], scenarios_wm2[mask])

    print("\n=== By Site ===")
    site_metrics = {}
    for sid, sname in SITE_NAMES.items():
        mask = (sites == sid)
        if mask.sum() == 0: continue
        print("  " + sname + " (n=" + str(mask.sum()) + ")")
        site_metrics[sname] = print_metrics(actuals_wm2[mask], scenarios_wm2[mask])

    np.savez_compressed(os.path.join(EVAL_DIR, "mcdropout_results.npz"),
                        actuals=actuals_wm2, scenarios=scenarios_wm2,
                        regimes=regimes, sites=sites)
    with open(os.path.join(EVAL_DIR, "mcdropout_metrics.json"), "w") as f:
        json.dump({"model": "MCDropoutLSTM", "overall": overall,
                   "by_regime": regime_metrics, "by_site": site_metrics}, f, indent=2)

    print("\n=== COPY THESE INTO TABLE II ===")
    print("MC Dropout LSTM  |  CRPS: " + str(round(overall["crps"],2))
          + "  | Cov 90%: " + str(overall["coverage_90"])
          + "%  | Width: "  + str(round(overall["width_90"],1))
          + "  | MAE: "     + str(round(overall["median_mae"],2)))
    print("\nResults saved to: " + EVAL_DIR)
    print("All 3 new baselines complete. Run compile_results_table.py to see full table.")

if __name__ == "__main__":
    main()