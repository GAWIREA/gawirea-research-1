# GenerativeAURORA - Step 3: Build Sequences (Fixed)
#
# Root cause of previous failure:
#   Daytime filtering removed nighttime rows, creating hour gaps.
#   The consecutive-hour check rejected every window.
#
# Fix:
#   Build one sequence per day per site.
#   Condition: all daytime hours of day D (padded to LOOKBACK length)
#   Target   : all daytime hours of day D (padded to HORIZON length)
#   This is the natural structure for solar scenario generation:
#   "given today's weather context, generate today's full GHI trajectory"
#
# Output saved to C:/Users/user/Paper_10/Dataset/
#
# Run:
#     python step3_build_sequences.py

import os
import json
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
PARQUET_FILE = "nsrdb_merged.parquet"
OUTPUT_DIR   = "C:/Users/user/Paper_10/Dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# SEQUENCE PARAMETERS
# -----------------------------------------------------------------------
# Each sequence = one day of daytime hours
# We pad/truncate to fixed length so all sequences have the same shape
LOOKBACK = 16    # max daytime hours per day (padded with 0 if shorter)
HORIZON  = 16    # same: we generate the full daytime GHI trajectory

# Features used as the conditioning input
CONDITION_FEATURES = [
    "GHI",
    "DHI",
    "DNI",
    "WIND SPEED",
    "TEMPERATURE",
    "RELATIVE HUMIDITY",
    "PRESSURE",
    "CLEARSKY GHI",
    "SOLAR ZENITH ANGLE",
    "CLEARNESS_INDEX",
]

TARGET_FEATURE = "GHI"

REGIME_MAP = {"Clear": 0, "Cloudy": 1, "Overcast": 2}

SITE_MAP = {
    "Phoenix_AZ":    0,
    "LosAngeles_CA": 1,
    "Denver_CO":     2,
    "Miami_FL":      3,
    "Seattle_WA":    4,
}

TRAIN_YEARS = [2018, 2019, 2020, 2021]
# Val : 2022 Jan-Jun
# Test: 2022 Jul-Dec


# -----------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------

def load_data(path):
    print("Loading " + path + "...")
    df = pd.read_parquet(path)
    print("Loaded " + str(len(df)) + " rows, " + str(len(df.columns)) + " columns.")
    missing = [f for f in CONDITION_FEATURES if f not in df.columns]
    if missing:
        print("WARNING: Missing columns: " + str(missing))
    else:
        print("All condition features present.")
    print("")
    return df


def compute_scalers(df):
    train_df = df[df["YEAR"].isin(TRAIN_YEARS)]
    scalers  = {}
    all_feats = list(dict.fromkeys(CONDITION_FEATURES + [TARGET_FEATURE]))
    for feat in all_feats:
        if feat in df.columns:
            col_min = float(train_df[feat].min())
            col_max = float(train_df[feat].max())
            if col_max == col_min:
                col_max = col_min + 1.0
            scalers[feat] = {"min": col_min, "max": col_max}
    print("Scaler fitted on " + str(len(train_df)) + " training rows.")
    return scalers


def scale_col(series, scalers, feat):
    s_min = scalers[feat]["min"]
    s_max = scalers[feat]["max"]
    return (series - s_min) / (s_max - s_min)


def pad_or_truncate(arr, target_len):
    # arr shape: (T,) or (T, F)
    # Pad with zeros at end, or truncate
    if arr.ndim == 1:
        if len(arr) >= target_len:
            return arr[:target_len]
        pad = np.zeros(target_len - len(arr), dtype=arr.dtype)
        return np.concatenate([arr, pad])
    else:
        if arr.shape[0] >= target_len:
            return arr[:target_len]
        pad = np.zeros((target_len - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)


def build_sequences_for_site(site_df, site_name, scalers):
    # Scale features
    for feat in CONDITION_FEATURES:
        if feat in site_df.columns and feat in scalers:
            site_df = site_df.copy()
            site_df[feat + "_SC"] = scale_col(site_df[feat], scalers, feat)

    tgt_sc = TARGET_FEATURE + "_SC"
    if tgt_sc not in site_df.columns:
        site_df = site_df.copy()
        site_df[tgt_sc] = scale_col(site_df[TARGET_FEATURE], scalers, TARGET_FEATURE)

    cond_cols = [f + "_SC" for f in CONDITION_FEATURES if f in site_df.columns]
    site_id   = SITE_MAP.get(site_name, -1)

    # Group by calendar date
    site_df["DATE"] = pd.to_datetime(site_df["DATETIME"]).dt.date
    grouped = site_df.groupby("DATE")

    X_list, Y_list, R_list, S_list, DT_list = [], [], [], [], []
    skipped = 0

    for date, day_df in grouped:
        day_df = day_df.sort_values("HOUR")

        # Require at least 6 daytime hours to form a meaningful sequence
        if len(day_df) < 6:
            skipped += 1
            continue

        # Condition: all daytime hours of this day (scaled features)
        cond_arr = day_df[cond_cols].values.astype(np.float32)
        cond_arr = pad_or_truncate(cond_arr, LOOKBACK)   # shape: (LOOKBACK, n_features)

        # Target: GHI trajectory of this day (scaled)
        tgt_arr = day_df[tgt_sc].values.astype(np.float32)
        tgt_arr = pad_or_truncate(tgt_arr, HORIZON)       # shape: (HORIZON,)

        # Sky regime: dominant regime of the day
        regime_vals = day_df["SKY_REGIME"].map(REGIME_MAP).dropna().values.astype(int)
        if len(regime_vals) == 0:
            skipped += 1
            continue
        regime_mode = int(np.bincount(regime_vals).argmax())

        X_list.append(cond_arr)
        Y_list.append(tgt_arr)
        R_list.append(regime_mode)
        S_list.append(site_id)
        DT_list.append(str(date))

    if not X_list:
        print("  WARNING: No sequences for " + site_name)
        return None

    result = {
        "X":         np.stack(X_list, axis=0),
        "Y":         np.stack(Y_list, axis=0),
        "regime":    np.array(R_list, dtype=np.int32),
        "site":      np.array(S_list, dtype=np.int32),
        "datetimes": DT_list,
    }

    regime_names = {v: k for k, v in REGIME_MAP.items()}
    unique, counts = np.unique(result["regime"], return_counts=True)
    regime_str = ", ".join(
        regime_names.get(int(u), str(u)) + ": " + str(int(c))
        for u, c in zip(unique, counts)
    )

    print("  " + site_name.ljust(20)
          + " days: " + str(len(X_list))
          + "  skipped: " + str(skipped)
          + "  X:" + str(result["X"].shape)
          + "  regimes: [" + regime_str + "]")

    return result


def split_and_save(all_data):
    splits = {
        "train": ([], [], [], []),
        "val":   ([], [], [], []),
        "test":  ([], [], [], []),
    }

    for site_name, data in all_data.items():
        if data is None:
            continue

        dts    = pd.to_datetime(data["datetimes"])
        years  = dts.year.values
        months = dts.month.values

        masks = {
            "train": np.isin(years, TRAIN_YEARS),
            "val":   (years == 2022) & (months <= 6),
            "test":  (years == 2022) & (months >= 7),
        }

        for split_name, mask in masks.items():
            if mask.sum() == 0:
                continue
            Xl, Yl, Rl, Sl = splits[split_name]
            Xl.append(data["X"][mask])
            Yl.append(data["Y"][mask])
            Rl.append(data["regime"][mask])
            Sl.append(data["site"][mask])

    counts = {}
    print("")
    print("Saving to " + OUTPUT_DIR + " ...")
    print("")

    for split_name, (Xl, Yl, Rl, Sl) in splits.items():
        if not Xl:
            print("  " + split_name + ": empty, skipping.")
            counts[split_name] = 0
            continue

        X = np.concatenate(Xl, axis=0)
        Y = np.concatenate(Yl, axis=0)
        R = np.concatenate(Rl, axis=0)
        S = np.concatenate(Sl, axis=0)

        out_path = os.path.join(OUTPUT_DIR, "sequences_" + split_name + ".npz")
        np.savez_compressed(out_path, X=X, Y=Y, regime=R, site=S)

        regime_names = {v: k for k, v in REGIME_MAP.items()}
        unique, cnts = np.unique(R, return_counts=True)
        regime_str = ", ".join(
            regime_names.get(int(u), str(u)) + ": " + str(int(c))
            for u, c in zip(unique, cnts)
        )

        print("  " + split_name.ljust(8) + " -> sequences_" + split_name + ".npz")
        print("    X shape : " + str(X.shape) + "  (days x hours x features)")
        print("    Y shape : " + str(Y.shape) + "  (days x horizon hours)")
        print("    Regimes : " + regime_str)
        print("")

        counts[split_name] = len(X)

    return counts


def main():
    print("GenerativeAURORA - Step 3: Build Sequences")
    print("Output dir : " + OUTPUT_DIR)
    print("")
    print("Approach   : one sequence per day per site (daytime hours only)")
    print("Lookback   : " + str(LOOKBACK) + " hours (padded)")
    print("Horizon    : " + str(HORIZON)  + " hours (padded)")
    print("Train years: " + str(TRAIN_YEARS))
    print("Val        : 2022 Jan-Jun")
    print("Test        : 2022 Jul-Dec")
    print("")

    df = load_data(PARQUET_FILE)

    scalers = compute_scalers(df)
    scaler_path = os.path.join(OUTPUT_DIR, "scaler_params.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scalers, f, indent=2)
    print("Scaler saved to " + scaler_path)
    print("")

    print("Building daily sequences per site...")
    all_data = {}
    for site_name in SITE_MAP.keys():
        site_df = df[df["SITE_NAME"] == site_name].copy()
        if len(site_df) == 0:
            print("  WARNING: No data for " + site_name)
            continue
        all_data[site_name] = build_sequences_for_site(site_df, site_name, scalers)

    counts = split_and_save(all_data)

    summary = {
        "lookback_hours":       LOOKBACK,
        "horizon_hours":        HORIZON,
        "n_condition_features": len([f for f in CONDITION_FEATURES if f in df.columns]),
        "condition_features":   CONDITION_FEATURES,
        "target_feature":       TARGET_FEATURE,
        "regime_map":           REGIME_MAP,
        "site_map":             SITE_MAP,
        "train_years":          TRAIN_YEARS,
        "n_train":              counts.get("train", 0),
        "n_val":                counts.get("val", 0),
        "n_test":               counts.get("test", 0),
    }
    summary_path = os.path.join(OUTPUT_DIR, "sequence_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to " + summary_path)

    print("")
    print("=== Step 3 complete ===")
    print("Train : " + str(counts.get("train", 0)) + " day-sequences")
    print("Val   : " + str(counts.get("val",   0)) + " day-sequences")
    print("Test  : " + str(counts.get("test",  0)) + " day-sequences")
    print("Total : " + str(sum(counts.values()))   + " day-sequences")
    print("")
    print("Next: run step4_build_model.py to define the diffusion model architecture.")


if __name__ == "__main__":
    main()