# GenerativeAURORA - Step 4: Diffusion Model Architecture
#
# Model: Conditional Denoising Diffusion Probabilistic Model (DDPM)
# Task : Generate full-day GHI trajectories conditioned on:
#        (1) 16-hour weather context window (X)
#        (2) Sky regime label (Clear / Cloudy / Overcast)
#        (3) Site identity (5 US sites)
#
# Architecture overview:
#   Condition encoder : 1D CNN + GRU encodes the 16 x 10 weather context
#   Regime embedding  : learned embedding for 3 regimes
#   Site embedding    : learned embedding for 5 sites
#   Diffusion U-Net   : 1D U-Net denoises the GHI trajectory at each timestep
#   Noise schedule    : linear beta schedule, T=200 steps
#
# This file defines the model classes only -- no training here.
# Import this in step5_train.py.
#
# Run this file directly to verify architecture shapes:
#   python step4_build_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------------------------------------------------
# HYPERPARAMETERS
# -----------------------------------------------------------------------
HORIZON          = 16     # GHI trajectory length (hours per day)
LOOKBACK         = 16     # condition window length
N_FEATURES       = 10     # number of weather input features
N_REGIMES        = 3      # Clear, Cloudy, Overcast
N_SITES          = 5      # Phoenix, LA, Denver, Miami, Seattle
T_DIFFUSION      = 200    # number of diffusion timesteps
EMBED_DIM        = 64     # embedding dimension throughout the model
HIDDEN_DIM       = 128    # hidden dimension for U-Net channels


# -----------------------------------------------------------------------
# NOISE SCHEDULE
# -----------------------------------------------------------------------

def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas  = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


# -----------------------------------------------------------------------
# SINUSOIDAL TIMESTEP EMBEDDING
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
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / (half - 1)
        ).to(t.device)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb

    def forward(self, t):
        emb = self.sinusoidal(t)
        return self.proj(emb)


# -----------------------------------------------------------------------
# CONDITION ENCODER
# Encodes the 16 x 10 weather context into a fixed-size vector
# -----------------------------------------------------------------------

class ConditionEncoder(nn.Module):
    def __init__(self, n_features=N_FEATURES, lookback=LOOKBACK, embed_dim=EMBED_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.gru   = nn.GRU(64, embed_dim, batch_first=True, num_layers=2)
        self.norm  = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, LOOKBACK, N_FEATURES)
        x = x.permute(0, 2, 1)              # (B, N_FEATURES, LOOKBACK)
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = x.permute(0, 2, 1)              # (B, LOOKBACK, 64)
        _, h = self.gru(x)                  # h: (2, B, embed_dim)
        cond = self.norm(h[-1])             # (B, embed_dim) -- last GRU layer
        return cond


# -----------------------------------------------------------------------
# 1D RESBLOCK FOR U-NET
# -----------------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, channels, embed_dim):
        super().__init__()
        self.norm1  = nn.GroupNorm(8, channels)
        self.conv1  = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2  = nn.GroupNorm(8, channels)
        self.conv2  = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(embed_dim, channels)   # timestep injection
        self.c_proj = nn.Linear(embed_dim, channels)   # condition injection

    def forward(self, x, t_emb, c_emb):
        # x: (B, C, L)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # inject timestep and condition embeddings
        h = h + self.t_proj(t_emb).unsqueeze(-1)
        h = h + self.c_proj(c_emb).unsqueeze(-1)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h   # residual connection


# -----------------------------------------------------------------------
# DIFFUSION U-NET
# 1D U-Net that denoises a GHI trajectory of length HORIZON
# -----------------------------------------------------------------------

class DiffusionUNet(nn.Module):
    def __init__(
        self,
        horizon    = HORIZON,
        embed_dim  = EMBED_DIM,
        hidden_dim = HIDDEN_DIM,
        n_regimes  = N_REGIMES,
        n_sites    = N_SITES,
    ):
        super().__init__()

        self.horizon    = horizon
        self.embed_dim  = embed_dim

        # --- Embeddings ---
        self.t_emb      = TimestepEmbedding(embed_dim)
        self.regime_emb = nn.Embedding(n_regimes, embed_dim)
        self.site_emb   = nn.Embedding(n_sites,   embed_dim)
        self.cond_enc   = ConditionEncoder(embed_dim=embed_dim)

        # Fuse all conditioning signals into one vector
        self.cond_fuse  = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # --- U-Net Encoder ---
        self.input_proj  = nn.Conv1d(1, hidden_dim, kernel_size=1)
        self.enc1        = ResBlock1D(hidden_dim, embed_dim)
        self.down1       = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=2, stride=2)

        self.enc2        = ResBlock1D(hidden_dim * 2, embed_dim)
        self.down2       = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck  = ResBlock1D(hidden_dim * 4, embed_dim)

        # --- U-Net Decoder ---
        self.up1         = nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=2, stride=2)
        self.dec1        = ResBlock1D(hidden_dim * 4, embed_dim)   # skip connection doubles channels

        self.up2         = nn.ConvTranspose1d(hidden_dim * 4, hidden_dim, kernel_size=2, stride=2)
        self.dec2        = ResBlock1D(hidden_dim * 2, embed_dim)   # skip connection doubles channels

        # --- Output projection ---
        self.out_norm    = nn.GroupNorm(8, hidden_dim * 2)
        self.out_proj    = nn.Conv1d(hidden_dim * 2, 1, kernel_size=1)

    def forward(self, x_noisy, t, cond_x, regime, site):
        # x_noisy : (B, HORIZON)       noisy GHI trajectory
        # t       : (B,)               diffusion timestep
        # cond_x  : (B, LOOKBACK, F)   weather context
        # regime  : (B,)               sky regime label
        # site    : (B,)               site label

        # Build conditioning vector
        t_e   = self.t_emb(t)                        # (B, E)
        r_e   = self.regime_emb(regime)              # (B, E)
        s_e   = self.site_emb(site)                  # (B, E)
        c_e   = self.cond_enc(cond_x)                # (B, E)
        cond  = self.cond_fuse(torch.cat([t_e, r_e, s_e, c_e], dim=-1))  # (B, E)

        # Reshape noisy input for 1D conv: (B, 1, HORIZON)
        h = x_noisy.unsqueeze(1)
        h = self.input_proj(h)                       # (B, H, HORIZON)

        # Encoder
        e1 = self.enc1(h,  t_e, cond)               # (B, H,   HORIZON)
        h  = self.down1(e1)                          # (B, 2H,  HORIZON//2)
        e2 = self.enc2(h,  t_e, cond)               # (B, 2H,  HORIZON//2)
        h  = self.down2(e2)                          # (B, 4H,  HORIZON//4)

        # Bottleneck
        h  = self.bottleneck(h, t_e, cond)          # (B, 4H,  HORIZON//4)

        # Decoder with skip connections
        h  = self.up1(h)                             # (B, 2H,  HORIZON//2)
        h  = torch.cat([h, e2], dim=1)              # (B, 4H,  HORIZON//2)
        h  = self.dec1(h, t_e, cond)                # (B, 4H,  HORIZON//2)

        h  = self.up2(h)                             # (B, H,   HORIZON)
        h  = torch.cat([h, e1], dim=1)              # (B, 2H,  HORIZON)
        h  = self.dec2(h, t_e, cond)                # (B, 2H,  HORIZON)

        # Output: predict the noise epsilon
        h  = F.silu(self.out_norm(h))
        h  = self.out_proj(h)                        # (B, 1, HORIZON)
        return h.squeeze(1)                          # (B, HORIZON)


# -----------------------------------------------------------------------
# FULL GENERATIVE MODEL WRAPPER
# Handles forward diffusion (add noise) and reverse (denoise)
# -----------------------------------------------------------------------

class GenerativeAURORA(nn.Module):
    def __init__(self, T=T_DIFFUSION):
        super().__init__()
        self.T      = T
        self.unet   = DiffusionUNet()

        betas, alphas, alpha_bar = make_beta_schedule(T)
        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0, t, noise=None):
        # Forward process: add noise to clean GHI trajectory x0
        # x0: (B, HORIZON)  t: (B,) integer timesteps
        if noise is None:
            noise = torch.randn_like(x0)
        ab  = self.alpha_bar[t].unsqueeze(1)         # (B, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    def forward(self, x0, cond_x, regime, site):
        # Training forward pass: sample random t, add noise, predict noise
        B  = x0.shape[0]
        t  = torch.randint(0, self.T, (B,), device=x0.device)
        x_noisy, noise = self.q_sample(x0, t)
        noise_pred     = self.unet(x_noisy, t, cond_x, regime, site)
        loss           = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, cond_x, regime, site, n_scenarios=50):
        # Reverse process: generate n_scenarios GHI trajectories
        # cond_x : (1, LOOKBACK, F) or (B, LOOKBACK, F)
        # Returns: (B * n_scenarios, HORIZON)
        B      = cond_x.shape[0]
        device = cond_x.device

        # Expand conditioning for n_scenarios
        cond_x = cond_x.repeat_interleave(n_scenarios, dim=0)   # (B*S, L, F)
        regime = regime.repeat_interleave(n_scenarios)           # (B*S,)
        site   = site.repeat_interleave(n_scenarios)             # (B*S,)

        # Start from pure noise
        x = torch.randn(B * n_scenarios, self.unet.horizon, device=device)

        # Reverse diffusion loop
        for step in reversed(range(self.T)):
            t_batch = torch.full((B * n_scenarios,), step,
                                 dtype=torch.long, device=device)
            noise_pred = self.unet(x, t_batch, cond_x, regime, site)

            beta  = self.betas[step]
            alpha = self.alphas[step]
            ab    = self.alpha_bar[step]

            # DDPM reverse step
            x = (1.0 / alpha.sqrt()) * (
                x - (beta / (1 - ab).sqrt()) * noise_pred
            )
            if step > 0:
                x = x + beta.sqrt() * torch.randn_like(x)

        return x   # (B * n_scenarios, HORIZON)


# -----------------------------------------------------------------------
# SHAPE VERIFICATION  (run this file directly to check)
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("GenerativeAURORA - Step 4: Architecture Verification")
    print("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))
    print("")

    model = GenerativeAURORA(T=T_DIFFUSION).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters    : " + str(total_params))
    print("Trainable parameters: " + str(trainable))
    print("")

    B = 4  # batch size for test
    x0     = torch.randn(B, HORIZON).to(device)
    cond_x = torch.randn(B, LOOKBACK, N_FEATURES).to(device)
    regime = torch.randint(0, N_REGIMES, (B,)).to(device)
    site   = torch.randint(0, N_SITES,   (B,)).to(device)

    print("Forward pass (training)...")
    loss = model(x0, cond_x, regime, site)
    print("  Loss value : " + str(round(loss.item(), 4)))
    print("  Loss shape : scalar -- OK")
    print("")

    print("Sampling (inference, 10 scenarios per sample)...")
    cond_x_single = torch.randn(2, LOOKBACK, N_FEATURES).to(device)
    regime_single = torch.randint(0, N_REGIMES, (2,)).to(device)
    site_single   = torch.randint(0, N_SITES,   (2,)).to(device)
    scenarios = model.sample(cond_x_single, regime_single, site_single, n_scenarios=10)
    print("  Scenarios shape: " + str(scenarios.shape) + "  expected: (20, " + str(HORIZON) + ")")
    print("")

    if scenarios.shape == (20, HORIZON):
        print("All shape checks passed. Ready for Step 5 (training).")
    else:
        print("WARNING: Shape mismatch. Check architecture.")