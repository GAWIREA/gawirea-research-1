# GenerativeAURORA - Baseline B: Conditional VAE Scenario Generator
#
# A Conditional Variational Autoencoder (CVAE) for solar GHI scenario generation.
# This is the standard generative baseline before diffusion models, commonly used
# in energy scenario generation literature (Applied Energy, IEEE TPWRS).
#
# Architecture:
#   Encoder : BiGRU(weather context) -> mu, log_var in latent space (dim=32)
#   Decoder : MLP(z + regime_emb + site_emb) -> GHI trajectory (length 16)
#   Training: ELBO = reconstruction loss + KL divergence (beta-VAE with beta=0.5)
#
# At inference: sample z ~ N(0,I) and decode -> scenario
#
# Run:
#   python baseline_b_vae.py

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
CKPT_DIR    = os.path.join(DATASET_DIR, "checkpoints_vae")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# HYPERPARAMETERS
# -----------------------------------------------------------------------
HORIZON     = 16
LOOKBACK    = 16
N_FEATURES  = 10
LATENT_DIM  = 32
EMBED_DIM   = 32
HIDDEN_DIM  = 128
BATCH_SIZE  = 128
EPOCHS      = 150
LR          = 1e-3
WEIGHT_DECAY= 1e-5
BETA_KL     = 0.5        # beta-VAE coefficient: balance reconstruction vs KL
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
# MODEL: Conditional VAE
# -----------------------------------------------------------------------
class ConditionEncoder(nn.Module):
    """Encodes the 16x10 weather context into a fixed vector."""
    def __init__(self, n_features=N_FEATURES, hidden=HIDDEN_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.gru  = nn.GRU(n_features, hidden, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.proj = nn.Linear(hidden * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, L, F)
        _, h = self.gru(x)          # h: (4, B, hidden) -- 2 layers x bidirectional
        h    = torch.cat([h[-2], h[-1]], dim=-1)   # (B, hidden*2)
        return self.norm(self.proj(h))              # (B, embed_dim)


class VAEEncoder(nn.Module):
    """Encodes (Y, context) -> (mu, log_var) in latent space."""
    def __init__(self, horizon=HORIZON, embed_dim=EMBED_DIM,
                 latent_dim=LATENT_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        # Y encoder: simple MLP on the GHI trajectory
        self.y_enc = nn.Sequential(
            nn.Linear(horizon, hidden),
            nn.SiLU(),
            nn.Linear(hidden, embed_dim),
        )
        # Fuse Y encoding with context
        self.fuse  = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden),   # y + context + regime+site
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mu_head      = nn.Linear(hidden, latent_dim)
        self.logvar_head  = nn.Linear(hidden, latent_dim)

    def forward(self, y, ctx, regime_e, site_e):
        y_e  = self.y_enc(y)
        fused = self.fuse(torch.cat([y_e, ctx, regime_e + site_e], dim=-1))
        return self.mu_head(fused), self.logvar_head(fused)


class VAEDecoder(nn.Module):
    """Decodes (z, context, regime, site) -> GHI trajectory."""
    def __init__(self, horizon=HORIZON, embed_dim=EMBED_DIM,
                 latent_dim=LATENT_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + embed_dim * 3, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, horizon),
            nn.Sigmoid(),    # output in [0,1] normalized scale
        )

    def forward(self, z, ctx, regime_e, site_e):
        inp = torch.cat([z, ctx, regime_e, site_e], dim=-1)
        return self.net(inp)


class ConditionalVAE(nn.Module):
    def __init__(self, n_regimes=3, n_sites=5,
                 embed_dim=EMBED_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.regime_emb = nn.Embedding(n_regimes, embed_dim)
        self.site_emb   = nn.Embedding(n_sites,   embed_dim)
        self.ctx_enc    = ConditionEncoder(embed_dim=embed_dim)
        self.vae_enc    = VAEEncoder(embed_dim=embed_dim, latent_dim=latent_dim)
        self.vae_dec    = VAEDecoder(embed_dim=embed_dim, latent_dim=latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y, regime, site):
        # Encode conditioning
        r_e = self.regime_emb(regime)
        s_e = self.site_emb(site)
        ctx = self.ctx_enc(x)

        # Encode posterior
        mu, log_var = self.vae_enc(y, ctx, r_e, s_e)
        z           = self.reparameterize(mu, log_var)

        # Decode
        y_hat = self.vae_dec(z, ctx, r_e, s_e)

        # ELBO
        recon_loss = F.mse_loss(y_hat, y)
        kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss       = recon_loss + BETA_KL * kl_loss
        return loss, recon_loss.item(), kl_loss.item()

    @torch.no_grad()
    def sample(self, x, regime, site, n_scenarios=N_SCENARIOS):
        # x      : (B, L, F)
        # Returns: (B, n_scenarios, H)
        B      = x.shape[0]
        device = x.device

        r_e = self.regime_emb(regime)
        s_e = self.site_emb(site)
        ctx = self.ctx_enc(x)

        # Expand for scenarios
        ctx = ctx.unsqueeze(1).expand(-1, n_scenarios, -1).reshape(B * n_scenarios, -1)
        r_e = r_e.unsqueeze(1).expand(-1, n_scenarios, -1).reshape(B * n_scenarios, -1)
        s_e = s_e.unsqueeze(1).expand(-1, n_scenarios, -1).reshape(B * n_scenarios, -1)

        z     = torch.randn(B * n_scenarios, self.latent_dim, device=device)
        y_hat = self.vae_dec(z, ctx, r_e, s_e)   # (B*S, H)
        return y_hat.reshape(B, n_scenarios, -1)   # (B, S, H)


# -----------------------------------------------------------------------
# METRICS (identical to step6_evaluate.py)
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
    crps = crps_score(actuals, scenarios)
    pb10 = pinball_np(actuals, scenarios, 0.10)
    pb50 = pinball_np(actuals, scenarios, 0.50)
    pb90 = pinball_np(actuals, scenarios, 0.90)
    cov90, w90 = coverage_width(actuals, scenarios, 0.10)
    cov80, _   = coverage_width(actuals, scenarios, 0.20)
    mae  = med_mae(actuals, scenarios)
    if label: print("  " + label)
    print("    CRPS         : " + str(round(crps, 4)))
    print("    Pinball p10  : " + str(round(pb10, 4)))
    print("    Pinball p50  : " + str(round(pb50, 4)))
    print("    Pinball p90  : " + str(round(pb90, 4)))
    print("    Coverage 90% : " + str(cov90) + "%")
    print("    Coverage 80% : " + str(cov80) + "%")
    print("    Width 90% PI : " + str(w90) + " W/m2")
    print("    Median MAE   : " + str(round(mae, 4)))
    return {"crps": crps, "pinball_p10": pb10, "pinball_p50": pb50,
            "pinball_p90": pb90, "coverage_90": cov90, "coverage_80": cov80,
            "width_90": w90, "median_mae": mae}


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():
    print("GenerativeAURORA - Baseline B: Conditional VAE")
    print("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device : " + str(device))
    print("")

    train_ds = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_train.npz"))
    val_ds   = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_val.npz"))
    test_ds  = SolarSequenceDataset(os.path.join(DATASET_DIR, "sequences_test.npz"))
    print("Train: " + str(len(train_ds)) + "  Val: " + str(len(val_ds)) + "  Test: " + str(len(test_ds)))

    sampler      = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,    num_workers=0, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,    num_workers=0)

    model = ConditionalVAE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Parameters: " + str(total_params))
    print("")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val  = float("inf")
    best_path = os.path.join(CKPT_DIR, "vae_best.pt")
    log_path  = os.path.join(DATASET_DIR, "training_log_vae.csv")

    print("Training for " + str(EPOCHS) + " epochs...")
    print("-" * 60)

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    t0 = time.time()
    for epoch in range(EPOCHS):
        # -- Train --
        model.train()
        tr_total = 0.0; n = 0
        for X, Y, regime, site in train_loader:
            X, Y = X.to(device), Y.to(device)
            regime, site = regime.to(device), site.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(X, Y, regime, site)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            tr_total += loss.item(); n += 1
        tr = tr_total / n

        # -- Validate --
        model.eval()
        vl_total = 0.0; n = 0
        with torch.no_grad():
            for X, Y, regime, site in val_loader:
                X, Y = X.to(device), Y.to(device)
                regime, site = regime.to(device), site.to(device)
                loss, _, _ = model(X, Y, regime, site)
                vl_total += loss.item(); n += 1
        vl = vl_total / n

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
    print("-" * 60)
    print("Training complete. Time: " + str(elapsed) + " min")
    print("Best val loss: " + str(round(best_val, 6)))
    print("")

    # Evaluation
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    scalers = load_scalers()

    all_sc, all_act, all_reg, all_sit = [], [], [], []
    with torch.no_grad():
        for X, Y, regime, site in test_loader:
            X = X.to(device); regime = regime.to(device); site = site.to(device)
            sc = model.sample(X, regime, site, N_SCENARIOS).cpu().numpy()
            all_sc.append(sc)
            all_act.append(Y.numpy())
            all_reg.append(regime.cpu().numpy())
            all_sit.append(site.cpu().numpy())

    scenarios = np.concatenate(all_sc,  axis=0)
    actuals   = np.concatenate(all_act, axis=0)
    regimes   = np.concatenate(all_reg, axis=0)
    sites     = np.concatenate(all_sit, axis=0)

    actuals_wm2   = inv_scale_ghi(actuals,   scalers)
    scenarios_wm2 = inv_scale_ghi(scenarios, scalers)

    print("")
    print("=== Conditional VAE: Test Metrics ===")
    overall = print_metrics(actuals_wm2, scenarios_wm2, "Overall (920 days, 5 sites)")

    print("")
    print("=== By Sky Regime ===")
    regime_metrics = {}
    for rid, rname in REGIME_NAMES.items():
        mask = (regimes == rid)
        if mask.sum() == 0: continue
        print("  " + rname + " (n=" + str(mask.sum()) + ")")
        regime_metrics[rname] = print_metrics(actuals_wm2[mask], scenarios_wm2[mask])

    print("")
    print("=== By Site ===")
    site_metrics = {}
    for sid, sname in SITE_NAMES.items():
        mask = (sites == sid)
        if mask.sum() == 0: continue
        print("  " + sname + " (n=" + str(mask.sum()) + ")")
        site_metrics[sname] = print_metrics(actuals_wm2[mask], scenarios_wm2[mask])

    # Save
    np.savez_compressed(
        os.path.join(EVAL_DIR, "vae_results.npz"),
        actuals=actuals_wm2, scenarios=scenarios_wm2,
        regimes=regimes, sites=sites,
    )
    with open(os.path.join(EVAL_DIR, "vae_metrics.json"), "w") as f:
        json.dump({"model": "ConditionalVAE", "overall": overall,
                   "by_regime": regime_metrics, "by_site": site_metrics}, f, indent=2)

    print("")
    print("=== COPY THESE INTO TABLE II ===")
    print("Conditional VAE  |  CRPS: " + str(round(overall["crps"],2))
          + "  | Pb p10: " + str(round(overall["pinball_p10"],2))
          + "  | Pb p50: " + str(round(overall["pinball_p50"],2))
          + "  | Cov 90%: " + str(overall["coverage_90"])
          + "%  | Width: " + str(round(overall["width_90"],1))
          + "  | MAE: " + str(round(overall["median_mae"],2)))
    print("")
    print("Results saved to: " + EVAL_DIR)
    print("Next: run baseline_c_timegrad.py")


if __name__ == "__main__":
    main()