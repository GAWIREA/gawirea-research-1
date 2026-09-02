import os
import time
import json
import requests
import pandas as pd
from io import StringIO

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
API_KEY = "pQ4eEm4XCrFHOWiMFy8ZtuZETH5SkjBAcxLbbagD" 
EMAIL   = "rhaditkurnia@gmail.com"
# -----------------------------------------------------------------------

SITES = [
    {"name": "Phoenix_AZ",    "lat": 33.4484,  "lon": -112.0740, "climate": "hot_desert"},
    {"name": "LosAngeles_CA", "lat": 34.0522,  "lon": -118.2437, "climate": "mediterranean"},
    {"name": "Denver_CO",     "lat": 39.7392,  "lon": -104.9903, "climate": "semi_arid"},
    {"name": "Miami_FL",      "lat": 25.7617,  "lon":  -80.1918, "climate": "subtropical"},
    {"name": "Seattle_WA",    "lat": 47.6062,  "lon": -122.3321, "climate": "oceanic"},
]

YEARS = [2018, 2019, 2020, 2021, 2022]
ATTRIBUTES = "ghi,dhi,dni,wind_speed,air_temperature,relative_humidity,surface_pressure,cloud_type,solar_zenith_angle,clearsky_ghi"

OUTPUT_DIR      = "nsrdb_data"
CHECKPOINT_FILE = "download_checkpoint.json"

# FIXED 1: Use the new 2026 domain (nlr.gov)
# FIXED 2: Use the versioned endpoint (psm3-2-2-download.csv)
# FIXED: Using the new 2026 PSM v4 endpoint for US data
BASE_URL = (
    "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv"
    "?wkt=POINT({lon}%20{lat})"
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
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
            return set(tuple(x) for x in data.get("completed", []))
        except: return set()
    return set()

def save_checkpoint(completed):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"completed": [list(x) for x in completed]}, f, indent=2)

def download_site_year(site, year):
    fname = os.path.join(OUTPUT_DIR, f"{site['name']}_{year}.csv")
    url = BASE_URL.format(lon=site["lon"], lat=site["lat"], year=year, email=EMAIL, attributes=ATTRIBUTES, api_key=API_KEY)

    print(f"   Downloading {site['name']} {year}...")
    try:
        resp = requests.get(url, timeout=120)
        
        if resp.status_code != 200:
            # This will now print the EXACT reason NREL is rejecting the call
            print(f"   API ERROR {resp.status_code}: {resp.text[:200]}")
            return False

        df = pd.read_csv(StringIO(resp.text), skiprows=2)
        df.columns = df.columns.astype(str).str.strip().str.upper()

        # Add site metadata
        df["SITE_NAME"] = site["name"]
        df["CLIMATE"]   = site["climate"]
        
        # Clearness Index Logic
        if "GHI" in df.columns and "CLEARSKY GHI" in df.columns:
            df["CLEARNESS_INDEX"] = (df["GHI"] / df["CLEARSKY GHI"].replace(0, float("nan"))).clip(0, 1).fillna(0)
            df["SKY_REGIME"] = "Overcast"
            df.loc[df["CLEARNESS_INDEX"] > 0.10, "SKY_REGIME"] = "Cloudy"
            df.loc[df["CLEARNESS_INDEX"] > 0.35, "SKY_REGIME"] = "Clear"

        df.to_csv(fname, index=False, encoding="utf-8")
        print(f"   Saved {fname} ({len(df)} rows)")
        return True
    except Exception as e:
        print(f"   SCRIPT ERROR: {str(e)}")
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    completed = load_checkpoint()
    
    print("\nGenerativeAURORA NSRDB Downloader - Paper 10 Fix")
    print(f"Status: Using developer.nlr.gov | Target: PSM v3.2.2")

    for site in SITES:
        print(f"\n--- Region: {site['name']} ---")
        for year in YEARS:
            key = (site["name"], year)
            if key in completed:
                print(f"   [skip] {year} done")
                continue

            if download_site_year(site, year):
                completed.add(key)
                save_checkpoint(completed)
            
            time.sleep(1.2) # Safety delay for rate limits

    print("\nProcess finished.")

if __name__ == "__main__":
    main()