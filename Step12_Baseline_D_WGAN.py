# GenerativeAURORA - Baseline D: Conditional WGAN-GP
#
# Wasserstein GAN with Gradient Penalty for solar GHI scenario generation.
# WGAN-GP is the standard GAN baseline for energy scenario generation papers.
# Known issues: mode collapse, training instability, no coverage guarantees.
#
# Architecture:
#   Generator : MLP(z + regime_emb + site_emb + context) -> GHI trajectory
#   Critic    : MLP(GHI + regime_emb + site_emb + context) -> scalar score
#   Training  : WGAN-GP (lambda=10), 5 critic steps per generator step
#
# Run:
#   python baseline_d_wgan.py

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
CKPT_DIR    = os.path.join(DATASET_DIR, "checkpoints_wgan")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# HYPERPARAMETERS
# -----------------------------------------------------------------------
HORIZON     = 16
LOOKBACK    = 16
N_FEATURES  = 10
LATENT_DIM  = 64
EMBED_DIM   = 32
HIDDEN_DIM  = 256
BATCH_SIZE  = 128
EPOCHS      = 200
LR_G        = 1e-4
LR_D        = 1e-4
LAMBDA_GP   = 10
N_CRITIC    = 5       # critic steps per generator step
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
# CONTEXT ENCODER (shared by G and D)
# -----------------------------------------------------------------------
class ContextEncoder(nn.Module):
    def __init__(self, n_features=N_FEATURES, embed_dim=EMBED_DIM,
                 n_regimes=3, n_sites=5):
        super().__init__()
        self.regime_emb = nn.Embedding(n_regimes, embed_dim)
        self.site_emb   = nn.Embedding(n_sites,   embed_dim)
        self.gru = nn.GRU(n_features, HIDDEN_DIM // 2, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.1)
        self.proj = nn.Linear(HIDDEN_DIM, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, regime, site):
        _, h   = self.gru(x)
        h      = torch.cat([h[-2], h[-1]], dim=-1)
        ctx    = self.norm(self.proj(h))
        r_e    = self.regime_emb(regime)
        s_e    = self.site_emb(site)
        return torch.cat([ctx, r_e, s_e], dim=-1)   # (B, 3*embed_dim)


# -----------------------------------------------------------------------
# GENERATOR
# -----------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, cond_dim=EMBED_DIM * 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + cond_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.LayerNorm(HIDDEN_DIM // 2), nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM // 2, HORIZON),
            nn.Sigmoid(),
        )

    def forward(self, z, cond):
        return self.net(torch.cat([z, cond], dim=-1))


# -----------------------------------------------------------------------
# CRITIC (Discriminator for WGAN)
# -----------------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, cond_dim=EMBED_DIM * 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HORIZON + cond_dim, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, y, cond):
        return self.net(torch.cat([y, cond], dim=-1))


# -----------------------------------------------------------------------
# GRADIENT PENALTY
# -----------------------------------------------------------------------
def gradient_penalty(critic, real, fake, cond, device):
    B     = real.shape[0]
    alpha = torch.rand(B, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interp, cond)
    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True,
    )[0]
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


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
    print("GenerativeAURORA - Baseline D: Conditional WGAN-GP")
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
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ctx_enc = ContextEncoder().to(device)
    G       = Generator().to(device)
    D       = Critic().to(device)

    opt_G = torch.optim.Adam(list(G.parameters()) + list(ctx_enc.parameters()),
                             lr=LR_G, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.0, 0.9))

    best_g_loss = float("inf")
    best_path   = os.path.join(CKPT_DIR, "wgan_best.pt")
    log_path    = os.path.join(DATASET_DIR, "training_log_wgan.csv")

    print("\nTraining WGAN-GP for " + str(EPOCHS) + " epochs...")
    print("-" * 60)

    with open(log_path, "w") as f:
        f.write("epoch,g_loss,d_loss\n")

    t0 = time.time()
    for epoch in range(EPOCHS):
        G.train(); D.train(); ctx_enc.train()
        g_total = 0.0; d_total = 0.0; nb = 0

        for X, Y, regime, site in train_loader:
            X, Y   = X.to(device), Y.to(device)
            regime = regime.to(device); site = site.to(device)
            B      = X.shape[0]
            cond   = ctx_enc(X, regime, site)

            # --- Critic steps ---
            for _ in range(N_CRITIC):
                z    = torch.randn(B, LATENT_DIM, device=device)
                fake = G(z, cond).detach()
                gp   = gradient_penalty(D, Y, fake, cond.detach(), device)
                d_loss = D(fake, cond.detach()).mean() - D(Y, cond.detach()).mean() + LAMBDA_GP * gp
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # --- Generator step ---
            z      = torch.randn(B, LATENT_DIM, device=device)
            fake   = G(z, cond)
            g_loss = -D(fake, cond).mean()
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            g_total += g_loss.item(); d_total += d_loss.item(); nb += 1

        g_avg = g_total / nb; d_avg = d_total / nb
        is_best = g_avg < best_g_loss
        if is_best:
            best_g_loss = g_avg
            torch.save({"G": G.state_dict(), "ctx": ctx_enc.state_dict()}, best_path)

        with open(log_path, "a") as f:
            f.write(str(epoch+1) + "," + str(round(g_avg,4)) + "," + str(round(d_avg,4)) + "\n")

        if (epoch+1) % 20 == 0 or is_best:
            marker = "  <-- best" if is_best else ""
            print("Epoch " + str(epoch+1).rjust(3) + "/" + str(EPOCHS)
                  + "  G: " + str(round(g_avg,4)).ljust(10)
                  + "  D: " + str(round(d_avg,4)).ljust(10) + marker)

    elapsed = round((time.time() - t0) / 60, 1)
    print("-" * 60)
    print("Training complete. Time: " + str(elapsed) + " min")
    print("")

    # Evaluation
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    G.load_state_dict(ckpt["G"]); ctx_enc.load_state_dict(ckpt["ctx"])
    G.eval(); ctx_enc.eval()
    scalers = load_scalers()

    all_sc, all_act, all_reg, all_sit = [], [], [], []
    with torch.no_grad():
        for X, Y, regime, site in test_loader:
            X      = X.to(device)
            regime = regime.to(device); site = site.to(device)
            B      = X.shape[0]
            cond   = ctx_enc(X, regime, site)
            cond_r = cond.unsqueeze(1).expand(-1, N_SCENARIOS, -1).reshape(B*N_SCENARIOS, -1)
            z      = torch.randn(B * N_SCENARIOS, LATENT_DIM, device=device)
            sc     = G(z, cond_r).cpu().numpy().reshape(B, N_SCENARIOS, HORIZON)
            all_sc.append(sc)
            all_act.append(Y.numpy())
            all_reg.append(regime.cpu().numpy())
            all_sit.append(site.cpu().numpy())

    scenarios = np.concatenate(all_sc,  axis=0)
    actuals   = np.concatenate(all_act, axis=0)
    regimes   = np.concatenate(all_reg, axis=0)
    sites     = np.concatenate(all_sit, axis=0)

    actuals_wm2   = inv_scale_ghi(actuals,                    scalers)
    scenarios_wm2 = inv_scale_ghi(np.clip(scenarios, 0, 1),  scalers)

    print("=== WGAN-GP: Test Metrics ===")
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

    np.savez_compressed(os.path.join(EVAL_DIR, "wgan_results.npz"),
                        actuals=actuals_wm2, scenarios=scenarios_wm2,
                        regimes=regimes, sites=sites)
    with open(os.path.join(EVAL_DIR, "wgan_metrics.json"), "w") as f:
        json.dump({"model": "WGAN-GP", "overall": overall,
                   "by_regime": regime_metrics, "by_site": site_metrics}, f, indent=2)

    print("\n=== COPY THESE INTO TABLE II ===")
    print("WGAN-GP  |  CRPS: " + str(round(overall["crps"],2))
          + "  | Cov 90%: " + str(overall["coverage_90"])
          + "%  | Width: "  + str(round(overall["width_90"],1))
          + "  | MAE: "     + str(round(overall["median_mae"],2)))
    print("\nResults saved to: " + EVAL_DIR)
    print("Next: run baseline_e_normflow.py")

if __name__ == "__main__":
    main()