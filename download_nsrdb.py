"""
GenerativeAURORA - NSRDB Data Downloader
Fixed version: correct domain (nrel.gov), checkpointing, no emoji.

BEFORE RUNNING:
    1. pip install requests pandas pyarrow
    2. Replace YOUR_API_KEY and YOUR_EMAIL below
    3. Run: python download_nsrdb.py
    4. If power cuts: just re-run, already-downloaded files are skipped automatically.
"""

import os
import time
import json
import requests
import pandas as pd
from io import StringIO

# -----------------------------------------------------------------------
# CONFIG - edit these two lines only
# -----------------------------------------------------------------------
API_KEY = "pQ4eEm4XCrFHOWiMFy8ZtuZETH5SkjBAcxLbbagD"   # <-- replace with your NEW key
EMAIL   = "rhaditkurnia@gmail.com"      # <-- your email address
# -----------------------------------------------------------------------

SITES = [
    {"name": "Phoenix_AZ",    "lat": 33.4484,  "lon": -112.0740, "climate": "hot_desert"},
    {"name": "LosAngeles_CA", "lat": 34.0522,  "lon": -118.2437, "climate": "mediterranean"},
    {"name": "Denver_CO",     "lat": 39.7392,  "lon": -104.9903, "climate": "semi_arid"},
    {"name": "Miami_FL",      "lat": 25.7617,  "lon":  -80.1918, "climate": "subtropical"},
    {"name": "Seattle_WA",    "lat": 47.6062,  "lon": -122.3321, "climate": "oceanic"},
]

YEARS = [2018, 2019, 2020, 2021, 2022]

ATTRIBUTES = ",".join([
    "ghi",
    "dhi",
    "dni",
    "wind_speed",
    "air_temperature",
    "relative_humidity",
    "surface_pressure",
    "cloud_type",
    "solar_zenith_angle",
    "clearsky_ghi",
])

OUTPUT_DIR      = "nsrdb_data"
CHECKPOINT_FILE = "download_checkpoint.json"

# FIXED: nrel.gov (not nlr.gov)
BASE_URL = (
    "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"
    "?wkt=POINT({lon}+{lat})"
    "&names={year}"
    "&leap_day=true"
    "&interval=60"
    "&utc=false"
    "&full_name=Researcher"
    "&email={email}"
    "&affiliation=YUNTECH"
    "&mailing_list=false"
    "&reason=Academic+Research"
    "&attributes={attributes}"
    "&api_key={api_key}"
)


def load_checkpoint():
    """Load set of already-completed (site, year) pairs from checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        completed = set(tuple(x) for x in data.get("completed", []))
        print("Checkpoint loaded. Already completed: " + str(len(completed)) + " files.")
        return completed
    return set()


def save_checkpoint(completed):
    """Save completed (site, year) pairs to checkpoint file."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"completed": [list(x) for x in completed]}, f, indent=2)


def download_site_year(site, year):
    fname = os.path.join(OUTPUT_DIR, site["name"] + "_" + str(year) + ".csv")

    url = BASE_URL.format(
        lon=site["lon"],
        lat=site["lat"],
        year=year,
        email=EMAIL,
        attributes=ATTRIBUTES,
        api_key=API_KEY,
    )

    print("  Downloading " + site["name"] + " " + str(year) + "...")

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        csv_content = resp.text

        # NSRDB CSV structure:
        # Row 0: site metadata (lat, lon, timezone, elevation, etc.)
        # Row 1: units for each column
        # Row 2: column headers
        # Row 3+: actual data
        df = pd.read_csv(StringIO(csv_content), skiprows=2)

        # Add site identification columns
        df["site"]    = site["name"]
        df["climate"] = site["climate"]
        df["lat"]     = site["lat"]
        df["lon"]     = site["lon"]

        # Compute clearness index: GHI / Clearsky GHI, clipped to [0, 1]
        if "GHI" in df.columns and "Clearsky GHI" in df.columns:
            df["clearness_index"] = (
                df["GHI"] / df["Clearsky GHI"].replace(0, float("nan"))
            ).clip(0, 1).fillna(0)

            # Sky regime labels (same thresholds as Papers 8 and 9)
            df["sky_regime"] = "Overcast"
            df.loc[df["clearness_index"] > 0.10, "sky_regime"] = "Cloudy"
            df.loc[df["clearness_index"] > 0.35, "sky_regime"] = "Clear"
        else:
            print("  WARNING: GHI or Clearsky GHI column not found.")
            print("  Columns available: " + str(list(df.columns)))

        df.to_csv(fname, index=False, encoding="utf-8")
        print("  Saved " + fname + "  (" + str(len(df)) + " rows)")
        return True

    except requests.exceptions.HTTPError as e:
        print("  HTTP ERROR " + site["name"] + " " + str(year) + ": " + str(e))
        print("  URL attempted: " + url)
        return False
    except Exception as e:
        print("  ERROR " + site["name"] + " " + str(year) + ": " + str(e))
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total     = len(SITES) * len(YEARS)
    completed = load_checkpoint()

    print("")
    print("GenerativeAURORA NSRDB Downloader")
    print("Sites  : " + str(len(SITES)))
    print("Years  : " + str(YEARS))
    print("Total  : " + str(total) + " files")
    print("Pending: " + str(total - len(completed)) + " files")
    print("")

    success_count = 0
    fail_count    = 0
    skip_count    = len(completed)

    for site in SITES:
        print("")
        print("--- " + site["name"] + " (" + site["climate"] + ") ---")

        for year in YEARS:
            key = (site["name"], year)

            # CHECKPOINT: skip if already done
            if key in completed:
                print("  [skip] " + site["name"] + " " + str(year) + " already completed")
                continue

            ok = download_site_year(site, year)

            if ok:
                completed.add(key)
                save_checkpoint(completed)   # save after every successful file
                success_count += 1
            else:
                fail_count += 1

            time.sleep(1.1)   # NREL rate limit: max 1 request per second

    print("")
    print("=== Download complete ===")
    print("Success : " + str(success_count))
    print("Skipped : " + str(skip_count))
    print("Failed  : " + str(fail_count))
    print("")

    if fail_count > 0:
        print("Some files failed. Re-run the script to retry failed files.")
        print("Successfully completed files will be skipped automatically.")
    else:
        print("All files downloaded. Run merge_nsrdb.py next.")


if __name__ == "__main__":
    main()