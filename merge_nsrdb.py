"""
GenerativeAURORA - NSRDB Merge and Inspect
Run this after all 25 CSV files are downloaded.

What this does:
    1. Loads all 25 CSV files from nsrdb_data/
    2. Standardizes column names to uppercase (matches Gemini download script)
    3. Re-computes clearness index and sky regime if missing
    4. Filters to daytime hours only (GHI > 0 and solar zenith < 90)
    5. Saves merged clean dataset as nsrdb_merged.parquet
    6. Prints summary stats per site

Run:
    python merge_nsrdb.py
"""

import os
import glob
import json
import pandas as pd

DATA_DIR      = "nsrdb_data"
OUTPUT_PARQUET = "nsrdb_merged.parquet"
SUMMARY_FILE   = "nsrdb_summary.json"

SITES = [
    "Phoenix_AZ",
    "LosAngeles_CA",
    "Denver_CO",
    "Miami_FL",
    "Seattle_WA",
]

YEARS = [2018, 2019, 2020, 2021, 2022]


def load_all_files():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print("Found " + str(len(files)) + " CSV files in " + DATA_DIR + "/")
    print("")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8")

            # Standardize all column names to uppercase and stripped
            df.columns = df.columns.astype(str).str.strip().str.upper()

            fname = os.path.basename(f)
            print("  Loaded " + fname.ljust(30) + " -> " + str(len(df)) + " rows, " + str(len(df.columns)) + " cols")
            dfs.append(df)

        except Exception as e:
            print("  ERROR loading " + f + ": " + str(e))

    if not dfs:
        print("No files loaded. Check that nsrdb_data/ folder exists and contains CSV files.")
        return None

    merged = pd.concat(dfs, ignore_index=True)
    print("")
    print("Total rows before filtering: " + str(len(merged)))
    return merged


def ensure_regime_columns(df):
    """Re-compute clearness index and sky regime if not already present."""
    if "CLEARNESS_INDEX" not in df.columns:
        if "GHI" in df.columns and "CLEARSKY GHI" in df.columns:
            df["CLEARNESS_INDEX"] = (
                df["GHI"] / df["CLEARSKY GHI"].replace(0, float("nan"))
            ).clip(0, 1).fillna(0)
        else:
            print("  WARNING: Cannot compute clearness index. GHI or CLEARSKY GHI missing.")
            df["CLEARNESS_INDEX"] = 0.0

    if "SKY_REGIME" not in df.columns:
        df["SKY_REGIME"] = "Overcast"
        df.loc[df["CLEARNESS_INDEX"] > 0.10, "SKY_REGIME"] = "Cloudy"
        df.loc[df["CLEARNESS_INDEX"] > 0.35, "SKY_REGIME"] = "Clear"

    return df


def filter_daytime(df):
    """
    Keep only daytime hours.
    Daytime = GHI > 0 AND solar zenith angle < 90 degrees.
    Nighttime rows are useless for solar generation modeling.
    """
    before = len(df)

    mask = (df["GHI"] > 0)
    if "SOLAR ZENITH ANGLE" in df.columns:
        mask = mask & (df["SOLAR ZENITH ANGLE"] < 90)

    df_day = df[mask].copy()
    after = len(df_day)
    removed = before - after
    pct = round(100.0 * removed / before, 1) if before > 0 else 0
    print("Daytime filter: kept " + str(after) + " rows, removed " + str(removed) + " nighttime rows (" + str(pct) + "%)")
    return df_day


def build_datetime_index(df):
    """
    Build a proper datetime column from Year/Month/Day/Hour columns.
    NSRDB uses local time by default.
    """
    time_cols = ["YEAR", "MONTH", "DAY", "HOUR"]
    if all(c in df.columns for c in time_cols):
        df["DATETIME"] = pd.to_datetime(
            df["YEAR"].astype(str) + "-"
            + df["MONTH"].astype(str).str.zfill(2) + "-"
            + df["DAY"].astype(str).str.zfill(2) + " "
            + df["HOUR"].astype(str).str.zfill(2) + ":00:00"
        )
        print("Datetime column built from Year/Month/Day/Hour.")
    else:
        missing = [c for c in time_cols if c not in df.columns]
        print("WARNING: Cannot build datetime. Missing columns: " + str(missing))
    return df


def print_summary(df):
    print("")
    print("=== Columns in merged dataset ===")
    print(str(list(df.columns)))

    print("")
    print("=== Row count per site ===")
    if "SITE_NAME" in df.columns:
        print(df.groupby("SITE_NAME").size().to_string())
    elif "SITE" in df.columns:
        print(df.groupby("SITE").size().to_string())
    else:
        print("No SITE_NAME or SITE column found.")

    print("")
    print("=== GHI stats per site (daytime only, W/m2) ===")
    site_col = "SITE_NAME" if "SITE_NAME" in df.columns else "SITE"
    if site_col in df.columns and "GHI" in df.columns:
        stats = df.groupby(site_col)["GHI"].describe().round(1)
        print(stats.to_string())

    print("")
    print("=== Sky regime distribution per site ===")
    if site_col in df.columns and "SKY_REGIME" in df.columns:
        regime_table = df.groupby([site_col, "SKY_REGIME"]).size().unstack(fill_value=0)
        print(regime_table.to_string())

    print("")
    print("=== Clearness index stats per site ===")
    if site_col in df.columns and "CLEARNESS_INDEX" in df.columns:
        ci_stats = df.groupby(site_col)["CLEARNESS_INDEX"].describe().round(3)
        print(ci_stats.to_string())


def save_summary(df):
    site_col = "SITE_NAME" if "SITE_NAME" in df.columns else "SITE"
    summary = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "sites": {},
    }
    if site_col in df.columns:
        for site, grp in df.groupby(site_col):
            regime_counts = {}
            if "SKY_REGIME" in grp.columns:
                regime_counts = grp["SKY_REGIME"].value_counts().to_dict()
            summary["sites"][site] = {
                "rows": len(grp),
                "ghi_mean": round(float(grp["GHI"].mean()), 2) if "GHI" in grp.columns else None,
                "ghi_max": round(float(grp["GHI"].max()), 2) if "GHI" in grp.columns else None,
                "clearness_mean": round(float(grp["CLEARNESS_INDEX"].mean()), 3) if "CLEARNESS_INDEX" in grp.columns else None,
                "sky_regime": regime_counts,
            }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to " + SUMMARY_FILE)


def main():
    print("GenerativeAURORA - NSRDB Merge Script")
    print("")

    # Step 1: Load all files
    df = load_all_files()
    if df is None:
        return

    # Step 2: Ensure regime columns exist
    df = ensure_regime_columns(df)

    # Step 3: Build datetime index
    df = build_datetime_index(df)

    # Step 4: Filter to daytime only
    df = filter_daytime(df)

    # Step 5: Print summary
    print_summary(df)

    # Step 6: Save merged parquet
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print("")
    print("Merged dataset saved to: " + OUTPUT_PARQUET)
    print("Total rows (daytime): " + str(len(df)))
    print("Total columns       : " + str(len(df.columns)))

    # Step 7: Save JSON summary
    save_summary(df)

    print("")
    print("Done. Next step: run eda_nsrdb.py to visualize distributions.")


if __name__ == "__main__":
    main()