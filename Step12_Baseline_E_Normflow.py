# GenerativeAURORA - Baseline E: Conditional Normalizing Flow (RealNVP)
#
# RealNVP-style normalizing flow for solar GHI scenario generation.
# Normalizing flows learn an exact bijective mapping between a simple
# prior (Gaussian) and the data distribution. They achieve exact
# likelihood but cannot do per-regime conformal calibration.
#
# Architecture:
#   8 coupling layers (RealNVP style), alternating mask patterns
#   Conditioning: GRU context encoder + regime + site embeddings
#   Each coupling layer: affine transform s(x_A, cond), t(x_A, cond)
#
# Reference:
#   Dinh et al., "Density estimation using Real-valued Non-Volume Preserving
#   (Real NVP) transformations," ICLR 2017.
#
# Run:
#   python baseline_e_normflow.py

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
CKPT_DIR    = os.path.join(DATASET_DIR, "checkpoints_normflow")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# HYPERPARAMETERS
# -----------------------------------------------------------------------
HORIZON     = 16
LOOKBACK    = 16
N_FEATURES  = 10
EMBED_DIM   = 64
HIDDEN_DIM  = 256
N_FLOWS     = 8       # number of coupling layers
BATCH_SIZE  = 128
EPOCHS      = 200
LR          = 1e-4
WEIGHT_DECAY= 1e-5
GRAD_CLIP   = 1.0
N_SCENARIOS = 50

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
# CONDITION ENCODER
# -----------------------------------------------------------------------
class ConditionEncoder(nn.Module):
    def __init__(self, n_features=N_FEATURES, embed_dim=EMBED_DIM,
                 n_regimes=3, n_sites=5):
        super().__init__()
        self.regime_emb = nn.Embedding(n_regimes, embed_dim)
        self.site_emb   = nn.Embedding(n_sites,   embed_dim)
        self.gru  = nn.GRU(n_features, embed_dim, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, regime, site):
        _, h = self.gru(x)
        h    = torch.cat([h[-2], h[-1]], dim=-1)
        ctx  = self.norm(self.proj(h))
        r_e  = self.regime_emb(regime)
        s_e  = self.site_emb(site)
        return ctx + r_e + s_e   # (B, embed_dim)


# -----------------------------------------------------------------------
# AFFINE COUPLING LAYER (RealNVP style)
# -----------------------------------------------------------------------
class AffineCouplingLayer(nn.Module):
    def __init__(self, dim, cond_dim, mask, hidden=HIDDEN_DIM):
        super().__init__()
        self.register_buffer("mask", mask)
        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(dim + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),         nn.ReLU(),
            nn.Linear(hidden, dim),            nn.Tanh(),
        )
        self.trans_net = nn.Sequential(
            nn.Linear(dim + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),         nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, cond):
        # Forward: x -> z  (data to latent)
        x_masked  = x * self.mask
        inp       = torch.cat([x_masked, cond], dim=-1)
        s         = self.scale_net(inp) * (1 - self.mask)
        t         = self.trans_net(inp) * (1 - self.mask)
        z         = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det   = (s * (1 - self.mask)).sum(dim=-1)
        return z, log_det

    def inverse(self, z, cond):
        # Inverse: z -> x  (latent to data, for sampling)
        z_masked  = z * self.mask
        inp       = torch.cat([z_masked, cond], dim=-1)
        s         = self.scale_net(inp) * (1 - self.mask)
        t         = self.trans_net(inp) * (1 - self.mask)
        x         = z_masked + (1 - self.mask) * ((z - t) * torch.exp(-s))
        return x


# -----------------------------------------------------------------------
# FULL NORMALIZING FLOW MODEL
# -----------------------------------------------------------------------
class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, dim=HORIZON, cond_dim=EMBED_DIM, n_flows=N_FLOWS,
                 n_regimes=3, n_sites=5):
        super().__init__()
        self.cond_enc = ConditionEncoder(embed_dim=cond_dim,
                                         n_regimes=n_regimes, n_sites=n_sites)
        # Alternating masks: first half vs second half
        masks = []
        for i in range(n_flows):
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[:dim//2] = 1.0
            else:
                mask[dim//2:] = 1.0
            masks.append(mask)

        self.flows = nn.ModuleList([
            AffineCouplingLayer(dim, cond_dim, masks[i])
            for i in range(n_flows)
        ])

    def forward(self, x, regime, site):
        # Returns negative log-likelihood (training loss)
        cond    = self.cond_enc(x[:, :, :].clone() if x.dim()==3 else x,
                                regime, site)
        # x here is Y (GHI trajectory)
        return cond

    def log_prob(self, y, cond):
        z       = y
        log_det = torch.zeros(y.shape[0], device=y.device)
        for flow in self.flows:
            z, ld = flow(z, cond)
            log_det = log_det + ld
        # Gaussian log prob
        log_pz  = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * 3.14159265))).sum(dim=-1)
        return log_pz + log_det

    @torch.no_grad()
    def sample(self, cond, n_scenarios=N_SCENARIOS):
        B      = cond.shape[0]
        device = cond.device
        # Expand conditioning
        cond_r = cond.unsqueeze(1).expand(-1, n_scenarios, -1).reshape(B*n_scenarios, -1)
        # Sample from prior
        z = torch.randn(B * n_scenarios, HORIZON, device=device)
        # Inverse flows
        x = z
        for flow in reversed(self.flows):
            x = flow.inverse(x, cond_r)
        return x.reshape(B, n_scenarios, HORIZON)


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
    print("GenerativeAURORA - Baseline E: Conditional Normalizing Flow (RealNVP)")
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

    model     = ConditionalNormalizingFlow().to(device)
    total_p   = sum(p.numel() for p in model.parameters())
    print("Parameters: " + str(total_p))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val  = float("inf")
    best_path = os.path.join(CKPT_DIR, "normflow_best.pt")
    log_path  = os.path.join(DATASET_DIR, "training_log_normflow.csv")

    print("\nTraining for " + str(EPOCHS) + " epochs...")
    print("-" * 60)

    with open(log_path, "w") as f:
        f.write("epoch,train_nll,val_nll\n")

    t0 = time.time()
    for epoch in range(EPOCHS):
        # Train
        model.train()
        tr_total = 0.0; n = 0
        for X, Y, regime, site in train_loader:
            X, Y   = X.to(device), Y.to(device)
            regime = regime.to(device); site = site.to(device)
            cond   = model.cond_enc(X, regime, site)
            nll    = -model.log_prob(Y, cond).mean()
            optimizer.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            tr_total += nll.item(); n += 1
        tr = tr_total / n

        # Validate
        model.eval()
        vl_total = 0.0; n = 0
        with torch.no_grad():
            for X, Y, regime, site in val_loader:
                X, Y   = X.to(device), Y.to(device)
                regime = regime.to(device); site = site.to(device)
                cond   = model.cond_enc(X, regime, site)
                nll    = -model.log_prob(Y, cond).mean()
                vl_total += nll.item(); n += 1
        vl = vl_total / n

        scheduler.step()
        is_best = vl < best_val
        if is_best:
            best_val = vl
            torch.save(model.state_dict(), best_path)

        with open(log_path, "a") as f:
            f.write(str(epoch+1) + "," + str(round(tr,4)) + "," + str(round(vl,4)) + "\n")

        if (epoch+1) % 20 == 0 or is_best:
            marker = "  <-- best" if is_best else ""
            print("Epoch " + str(epoch+1).rjust(3) + "/" + str(EPOCHS)
                  + "  NLL train: " + str(round(tr,4)).ljust(10)
                  + "  val: "       + str(round(vl,4)).ljust(10) + marker)

    elapsed = round((time.time() - t0) / 60, 1)
    print("-" * 60)
    print("Training complete. Time: " + str(elapsed) + " min")
    print("Best val NLL: " + str(round(best_val, 4)))
    print("")

    # Evaluation
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    scalers = load_scalers()

    all_sc, all_act, all_reg, all_sit = [], [], [], []
    with torch.no_grad():
        for X, Y, regime, site in test_loader:
            X      = X.to(device)
            regime = regime.to(device); site = site.to(device)
            cond   = model.cond_enc(X, regime, site)
            sc     = model.sample(cond, N_SCENARIOS).cpu().numpy()
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

    print("=== Normalizing Flow: Test Metrics ===")
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

    np.savez_compressed(os.path.join(EVAL_DIR, "normflow_results.npz"),
                        actuals=actuals_wm2, scenarios=scenarios_wm2,
                        regimes=regimes, sites=sites)
    with open(os.path.join(EVAL_DIR, "normflow_metrics.json"), "w") as f:
        json.dump({"model": "NormalizingFlow", "overall": overall,
                   "by_regime": regime_metrics, "by_site": site_metrics}, f, indent=2)

    print("\n=== COPY THESE INTO TABLE II ===")
    print("Norm. Flow  |  CRPS: " + str(round(overall["crps"],2))
          + "  | Cov 90%: " + str(overall["coverage_90"])
          + "%  | Width: "  + str(round(overall["width_90"],1))
          + "  | MAE: "     + str(round(overall["median_mae"],2)))
    print("\nResults saved to: " + EVAL_DIR)
    print("Next: run baseline_f_mc_dropout.py")

if __name__ == "__main__":
    main()