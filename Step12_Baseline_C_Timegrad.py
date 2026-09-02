# GenerativeAURORA - Baseline C: TimeGrad-Style Diffusion
#
# TimeGrad (Rasul et al., NeurIPS 2021) is the closest published DDPM competitor
# to GenerativeAURORA. It uses an autoregressive RNN to condition a DDPM on
# past observations, generating probabilistic forecasts step-by-step.
#
# Our implementation adapts TimeGrad's core idea to the solar scenario generation
# setting, making it a fair direct DDPM competitor:
#   - Same T=200 diffusion steps and linear noise schedule as GenerativeAURORA
#   - Same training data, same regime-weighted sampler
#   - Same 50-scenario evaluation protocol
#   - Key difference: uses GRU-based autoregressive conditioning instead of
#     GenerativeAURORA's CNN-GRU + U-Net architecture with explicit regime/site embeddings
#
# This isolates the contribution of GenerativeAURORA's architecture choices over
# a standard DDPM baseline in the time series domain.
#
# Reference:
#   Rasul et al., "Autoregressive Denoising Diffusion Models for Multivariate
#   Probabilistic Time Series Forecasting," ICML 2021.
#
# Run:
#   python baseline_c_timegrad.py

import os
import sys
import json
import math
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
CKPT_DIR    = os.path.join(DATASET_DIR, "checkpoints_timegrad")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# HYPERPARAMETERS (match GenerativeAURORA where applicable for fair comparison)
# -----------------------------------------------------------------------
HORIZON     = 16
LOOKBACK    = 16
N_FEATURES  = 10
T_DIFFUSION = 200        # same as GenerativeAURORA
HIDDEN_DIM  = 128
EMBED_DIM   = 64
BATCH_SIZE  = 128
EPOCHS      = 200        # same as GenerativeAURORA
LR          = 1e-4       # same as GenerativeAURORA
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
# NOISE SCHEDULE (identical to GenerativeAURORA)
# -----------------------------------------------------------------------
def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas     = torch.linspace(beta_start, beta_end, T)
    alphas    = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


# -----------------------------------------------------------------------
# SINUSOIDAL TIMESTEP EMBEDDING (identical to GenerativeAURORA)
# -----------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def sinusoidal(self, t):
        half  = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / (half - 1)
        ).to(t.device)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        return self.proj(self.sinusoidal(t))


# -----------------------------------------------------------------------
# TIMEGRAD DENOISER
# Key architectural difference from GenerativeAURORA:
#   - Uses a GRU to process the noisy sequence autoregressively
#   - No explicit regime/site conditioning (pure architectural comparison)
#   - No U-Net skip connections (flat GRU denoiser)
# -----------------------------------------------------------------------
class TimeGradDenoiser(nn.Module):
    """
    Denoises a full GHI trajectory conditioned only on:
      (1) the weather context encoded by a GRU
      (2) the diffusion timestep
    No explicit regime or site embeddings -- that is the key architectural
    difference vs GenerativeAURORA.
    """
    def __init__(self, horizon=HORIZON, n_features=N_FEATURES,
                 lookback=LOOKBACK, hidden=HIDDEN_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.horizon   = horizon

        # Context encoder: plain GRU (no CNN, no regime/site emb)
        self.ctx_gru   = nn.GRU(n_features, hidden, num_layers=2,
                                batch_first=True, bidirectional=True, dropout=0.1)
        self.ctx_proj  = nn.Linear(hidden * 2, embed_dim)
        self.ctx_norm  = nn.LayerNorm(embed_dim)

        # Timestep embedding
        self.t_emb     = TimestepEmbedding(embed_dim)

        # Denoising GRU: processes noisy trajectory + context
        # Input at each step: (noisy_ghi_t, context, t_emb)
        self.den_gru   = nn.GRU(
            input_size  = 1 + embed_dim * 2,
            hidden_size = hidden * 2,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.1,
        )
        self.out_norm  = nn.LayerNorm(hidden * 2)
        self.out_proj  = nn.Linear(hidden * 2, 1)

    def forward(self, x_noisy, t, cond_x):
        # x_noisy : (B, H)   noisy GHI trajectory
        # t       : (B,)     diffusion timestep
        # cond_x  : (B, L, F) weather context

        B, H = x_noisy.shape

        # Encode context
        _, h_ctx  = self.ctx_gru(cond_x)          # h_ctx: (4, B, hidden)
        h_ctx     = torch.cat([h_ctx[-2], h_ctx[-1]], dim=-1)  # (B, hidden*2)
        ctx       = self.ctx_norm(self.ctx_proj(h_ctx))         # (B, embed_dim)

        # Timestep embedding
        t_e = self.t_emb(t)   # (B, embed_dim)

        # Prepare denoising GRU input:
        # For each horizon step, concatenate noisy value + context + t_emb
        x_seq  = x_noisy.unsqueeze(-1)                          # (B, H, 1)
        ctx_ex = ctx.unsqueeze(1).expand(-1, H, -1)             # (B, H, embed_dim)
        t_ex   = t_e.unsqueeze(1).expand(-1, H, -1)             # (B, H, embed_dim)
        inp    = torch.cat([x_seq, ctx_ex, t_ex], dim=-1)       # (B, H, 1+2E)

        out, _ = self.den_gru(inp)                               # (B, H, 2*hidden)
        out    = self.out_norm(out)
        eps    = self.out_proj(out).squeeze(-1)                  # (B, H) predicted noise
        return eps


# -----------------------------------------------------------------------
# FULL TIMEGRAD MODEL
# -----------------------------------------------------------------------
class TimeGrad(nn.Module):
    def __init__(self, T=T_DIFFUSION):
        super().__init__()
        self.T       = T
        self.denoiser = TimeGradDenoiser()

        betas, alphas, alpha_bar = make_beta_schedule(T)
        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].unsqueeze(1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    def forward(self, x0, cond_x):
        B  = x0.shape[0]
        t  = torch.randint(0, self.T, (B,), device=x0.device)
        x_noisy, noise = self.q_sample(x0, t)
        noise_pred     = self.denoiser(x_noisy, t, cond_x)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, cond_x, n_scenarios=N_SCENARIOS):
        B      = cond_x.shape[0]
        device = cond_x.device

        cond_x = cond_x.repeat_interleave(n_scenarios, dim=0)   # (B*S, L, F)
        x      = torch.randn(B * n_scenarios, self.denoiser.horizon, device=device)

        for step in reversed(range(self.T)):
            t_batch = torch.full((B * n_scenarios,), step, dtype=torch.long, device=device)
            noise_pred = self.denoiser(x, t_batch, cond_x)

            beta  = self.betas[step]
            alpha = self.alphas[step]
            ab    = self.alpha_bar[step]

            x = (1.0 / alpha.sqrt()) * (x - (beta / (1 - ab).sqrt()) * noise_pred)
            if step > 0:
                x = x + beta.sqrt() * torch.randn_like(x)

        return x.reshape(B, n_scenarios, -1)   # (B, S, H)


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
    print("GenerativeAURORA - Baseline C: TimeGrad-Style Diffusion")
    print("")
    print("Key difference vs GenerativeAURORA:")
    print("  - GRU denoiser (no U-Net skip connections)")
    print("  - No regime or site conditioning")
    print("  - Same T=200, same noise schedule, same training protocol")
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

    model = TimeGrad(T=T_DIFFUSION).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Parameters: " + str(total_params))
    print("")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val  = float("inf")
    best_path = os.path.join(CKPT_DIR, "timegrad_best.pt")
    log_path  = os.path.join(DATASET_DIR, "training_log_timegrad.csv")

    print("Training for " + str(EPOCHS) + " epochs (same as GenerativeAURORA)...")
    print("-" * 65)

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    t0 = time.time()
    for epoch in range(EPOCHS):
        # -- Train --
        model.train()
        tr_total = 0.0; n = 0
        for X, Y, regime, site in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = model(Y, X)    # TimeGrad: no regime/site
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
                loss = model(Y, X)
                vl_total += loss.item(); n += 1
        vl = vl_total / n

        scheduler.step()
        is_best = vl < best_val
        if is_best:
            best_val = vl
            torch.save(model.state_dict(), best_path)

        with open(log_path, "a") as f:
            f.write(str(epoch+1) + "," + str(round(tr,6)) + "," + str(round(vl,6)) + "\n")

        if (epoch + 1) % 20 == 0 or is_best:
            marker = "  <-- best" if is_best else ""
            print("Epoch " + str(epoch+1).rjust(3) + "/" + str(EPOCHS)
                  + "  train: " + str(round(tr,6)).ljust(10)
                  + "  val: "   + str(round(vl,6)).ljust(10) + marker)

    elapsed = round((time.time() - t0) / 60, 1)
    print("-" * 65)
    print("Training complete. Time: " + str(elapsed) + " min")
    print("Best val loss: " + str(round(best_val, 6)))
    print("")

    # Evaluation
    print("Evaluating on test set (" + str(N_SCENARIOS) + " scenarios per day)...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    scalers = load_scalers()

    all_sc, all_act, all_reg, all_sit = [], [], [], []
    total_batches = len(test_loader)
    with torch.no_grad():
        for i, (X, Y, regime, site) in enumerate(test_loader):
            X = X.to(device)
            sc = model.sample(X, N_SCENARIOS).cpu().numpy()   # (B, S, H)
            all_sc.append(sc)
            all_act.append(Y.numpy())
            all_reg.append(regime.numpy())
            all_sit.append(site.numpy())
            if (i+1) % 5 == 0 or (i+1) == total_batches:
                print("  Batch " + str(i+1) + "/" + str(total_batches))

    scenarios = np.concatenate(all_sc,  axis=0)
    actuals   = np.concatenate(all_act, axis=0)
    regimes   = np.concatenate(all_reg, axis=0)
    sites     = np.concatenate(all_sit, axis=0)

    actuals_wm2   = inv_scale_ghi(actuals,   scalers)
    scenarios_wm2 = inv_scale_ghi(np.clip(scenarios, 0, 1), scalers)

    print("")
    print("=== TimeGrad-Style: Test Metrics ===")
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
        os.path.join(EVAL_DIR, "timegrad_results.npz"),
        actuals=actuals_wm2, scenarios=scenarios_wm2,
        regimes=regimes, sites=sites,
    )
    with open(os.path.join(EVAL_DIR, "timegrad_metrics.json"), "w") as f:
        json.dump({"model": "TimeGrad", "overall": overall,
                   "by_regime": regime_metrics, "by_site": site_metrics}, f, indent=2)

    print("")
    print("=== COPY THESE INTO TABLE II ===")
    print("TimeGrad-style  |  CRPS: " + str(round(overall["crps"],2))
          + "  | Pb p10: " + str(round(overall["pinball_p10"],2))
          + "  | Pb p50: " + str(round(overall["pinball_p50"],2))
          + "  | Cov 90%: " + str(overall["coverage_90"])
          + "%  | Width: " + str(round(overall["width_90"],1))
          + "  | MAE: " + str(round(overall["median_mae"],2)))
    print("")
    print("Results saved to: " + EVAL_DIR)
    print("")
    print("=== All 3 baselines complete ===")
    print("Now run compile_results_table.py to generate the updated Table II.")


if __name__ == "__main__":
    main()