# GenerativeAURORA - Step 5: Train Diffusion Model
#
# Features:
#   - Runs on RTX 4080 Super (CUDA) automatically
#   - Weighted sampler to handle regime imbalance (Clear 93%, Cloudy 6%, Overcast 1%)
#   - Checkpoint saved after every epoch to Dataset/checkpoints/
#   - Resumes automatically from last checkpoint if power cuts
#   - Validation loss tracked every epoch
#   - Best model saved separately as best_model.pt
#
# Run:
#   python step5_train.py

import os
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
DATASET_DIR  = "C:/Users/user/Paper_10/Dataset"
CODING_DIR   = "C:/Users/user/Paper_10/Coding"
CKPT_DIR     = os.path.join(DATASET_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# -----------------------------------------------------------------------
BATCH_SIZE    = 128
EPOCHS        = 200
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
GRAD_CLIP     = 1.0
T_DIFFUSION   = 200

# Model dimensions (must match step4_build_model.py)
HORIZON     = 16
LOOKBACK    = 16
N_FEATURES  = 10
N_REGIMES   = 3
N_SITES     = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 128

# -----------------------------------------------------------------------
# IMPORT MODEL FROM STEP 4
# -----------------------------------------------------------------------
import sys
sys.path.insert(0, CODING_DIR)
from step4_build_model import GenerativeAURORA


# -----------------------------------------------------------------------
# DATASET CLASS
# -----------------------------------------------------------------------

class SolarSequenceDataset(Dataset):
    def __init__(self, npz_path):
        data       = np.load(npz_path)
        self.X      = torch.tensor(data["X"],      dtype=torch.float32)
        self.Y      = torch.tensor(data["Y"],      dtype=torch.float32)
        self.regime = torch.tensor(data["regime"], dtype=torch.long)
        self.site   = torch.tensor(data["site"],   dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.regime[idx], self.site[idx]


# -----------------------------------------------------------------------
# WEIGHTED SAMPLER (handles Clear 93% / Cloudy 6% / Overcast 1%)
# -----------------------------------------------------------------------

def make_weighted_sampler(dataset):
    regime_labels = dataset.regime.numpy()
    unique, counts = np.unique(regime_labels, return_counts=True)
    freq = dict(zip(unique.tolist(), counts.tolist()))

    # Weight = inverse frequency so rare regimes are sampled more
    weights = np.array([1.0 / freq[int(r)] for r in regime_labels], dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )

    regime_names = {0: "Clear", 1: "Cloudy", 2: "Overcast"}
    print("  Regime weights (inverse frequency):")
    for k, v in freq.items():
        w = 1.0 / v
        print("    " + regime_names.get(k, str(k)).ljust(10)
              + " count: " + str(v).rjust(5)
              + "  weight: " + str(round(w, 6)))
    return sampler


# -----------------------------------------------------------------------
# CHECKPOINT UTILITIES
# -----------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss":   val_loss,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    best_val    = ckpt["val_loss"]
    print("Resumed from epoch " + str(ckpt["epoch"])
          + "  val_loss: " + str(round(best_val, 6)))
    return start_epoch, best_val


def find_latest_checkpoint(ckpt_dir):
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pt")]
    if not files:
        return None
    epochs = [int(f.replace("epoch_", "").replace(".pt", "")) for f in files]
    latest = max(epochs)
    return os.path.join(ckpt_dir, "epoch_" + str(latest) + ".pt")


# -----------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for X, Y, regime, site in loader:
        X      = X.to(device)
        Y      = Y.to(device)
        regime = regime.to(device)
        site   = site.to(device)

        optimizer.zero_grad()
        loss = model(Y, X, regime, site)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for X, Y, regime, site in loader:
            X      = X.to(device)
            Y      = Y.to(device)
            regime = regime.to(device)
            site   = site.to(device)

            loss = model(Y, X, regime, site)
            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("GenerativeAURORA - Step 5: Training")
    print("")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print("Device : " + gpu_name + "  (" + str(vram) + " GB VRAM)")
    else:
        device = torch.device("cpu")
        print("Device : CPU  (CUDA not available -- install PyTorch with CUDA)")
        print("         pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("")

    # Datasets
    train_path = os.path.join(DATASET_DIR, "sequences_train.npz")
    val_path   = os.path.join(DATASET_DIR, "sequences_val.npz")

    print("Loading datasets...")
    train_ds = SolarSequenceDataset(train_path)
    val_ds   = SolarSequenceDataset(val_path)
    print("  Train: " + str(len(train_ds)) + " sequences")
    print("  Val  : " + str(len(val_ds))   + " sequences")
    print("")

    # Weighted sampler for train only
    print("Building weighted sampler for regime imbalance...")
    sampler = make_weighted_sampler(train_ds)
    print("")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = GenerativeAURORA(T=T_DIFFUSION).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: " + str(total_params))
    print("")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # Resume from checkpoint if available
    start_epoch = 0
    best_val    = float("inf")
    latest_ckpt = find_latest_checkpoint(CKPT_DIR)

    if latest_ckpt is not None:
        print("Found checkpoint: " + latest_ckpt)
        start_epoch, best_val = load_checkpoint(model, optimizer, latest_ckpt, device)
        print("")
    else:
        print("No checkpoint found. Starting from scratch.")
        print("")

    # Training log
    log_path = os.path.join(DATASET_DIR, "training_log.csv")
    if start_epoch == 0:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,lr,elapsed_sec\n")

    print("Starting training...")
    print("Epochs       : " + str(EPOCHS))
    print("Batch size   : " + str(BATCH_SIZE))
    print("LR           : " + str(LR))
    print("Grad clip    : " + str(GRAD_CLIP))
    print("T diffusion  : " + str(T_DIFFUSION))
    print("Checkpoints  : " + CKPT_DIR)
    print("")
    print("-" * 65)

    t0_total = time.time()

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = validate(model, val_loader, device)
        scheduler.step()

        elapsed = round(time.time() - t0, 1)
        current_lr = round(scheduler.get_last_lr()[0], 8)

        # Print progress
        is_best = val_loss < best_val
        marker  = "  <-- best" if is_best else ""
        print("Epoch " + str(epoch + 1).rjust(3) + "/" + str(EPOCHS)
              + "  train: " + str(round(train_loss, 6)).ljust(10)
              + "  val: "   + str(round(val_loss,   6)).ljust(10)
              + "  lr: "    + str(current_lr).ljust(12)
              + "  " + str(elapsed) + "s"
              + marker)

        # Save checkpoint every epoch (enables resume after power cut)
        ckpt_path = os.path.join(CKPT_DIR, "epoch_" + str(epoch + 1) + ".pt")
        save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, ckpt_path)

        # Keep only last 3 checkpoints to save disk space
        all_ckpts = sorted(
            [f for f in os.listdir(CKPT_DIR) if f.startswith("epoch_") and f.endswith(".pt")],
            key=lambda x: int(x.replace("epoch_", "").replace(".pt", ""))
        )
        for old in all_ckpts[:-3]:
            os.remove(os.path.join(CKPT_DIR, old))

        # Save best model separately
        if is_best:
            best_val = val_loss
            best_path = os.path.join(DATASET_DIR, "best_model.pt")
            torch.save(model.state_dict(), best_path)

        # Append to training log
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(str(epoch + 1) + ","
                    + str(round(train_loss, 6)) + ","
                    + str(round(val_loss, 6)) + ","
                    + str(current_lr) + ","
                    + str(elapsed) + "\n")

    total_time = round((time.time() - t0_total) / 60, 1)
    print("-" * 65)
    print("")
    print("Training complete.")
    print("Total time    : " + str(total_time) + " minutes")
    print("Best val loss : " + str(round(best_val, 6)))
    print("Best model    : " + os.path.join(DATASET_DIR, "best_model.pt"))
    print("Training log  : " + log_path)
    print("")
    print("Next: run step6_evaluate.py to generate scenarios and compute metrics.")


if __name__ == "__main__":
    main()