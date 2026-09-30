"""
================================================================================
GenerativeAURORA - NSRDB Data Downloader
================================================================================

PURPOSE
-------
This script downloads hourly solar and meteorological data from the NREL
National Solar Radiation Database (NSRDB) for the five representative study
locations used in the GenerativeAURORA project.

The downloaded data are used as the raw dataset for the subsequent stages of
the GenerativeAURORA pipeline.

DATASET CONFIGURATION
---------------------
Site Locations:
For this case, we used five representative locations:
A. Phoenix AZ     -> Hot desert
B. Los Angeles CA -> Mediterranian
C. Denver CO      -> Semi Arid
D. Miami FL       -> Subtropical
E. Seattle WA     -> Oceanic

Study Period:
5 years (2018-2022)

Temporal Resolution:
60 minutes (hourly)

NSRDB Variables:
1. Global Horizontal Irradiance (ghi)
2. Diffuse Horizontal Irradiance (dhi)
3. Direct Normal Irradiance (dni)
4. Wind Speed
5. Air Temperature
6. Relative Humidity
7. Surface Pressure
8. Cloud Type
9. Solar Zenith Angle
10. Clearsky GHI

Additional variables are calculated locally after the NSRDB data are
downloaded:
1. Clearness Index
2. Sky Regime

DERIVED VARIABLES
-----------------
Clearness Index is calculated as:

    Clearness Index = GHI / Clearsky GHI

The resulting value is clipped to the range [0, 1].

Each observation is then assigned to one of three sky regimes:
    Clearness Index <= 0.10        -> Overcast
    0.10 < Clearness Index <= 0.35 -> Cloudy
    Clearness Index > 0.35         -> Clear

MAIN FUNCTIONS
--------------
The script is organized into four main functions.
1. load_checkpoint()
2. save_checkpoint()
3. download_site_year()
4. main()

CHECKPOINT / RESUME MECHANISM
-----------------------------
The script uses:

    download_checkpoint.json

to record successfully downloaded site-year combinations.

If the download process is interrupted because of:
    - power failure,
    - internet connection problems,
    - computer restart, or
    - an API request failure,

the script can simply be run again.

Previously completed files will be skipped automatically, while failed
downloads will be attempted again.

BEFORE RUNNING
--------------
1. Install the required packages:

       pip install requests pandas

2. Provide a valid NREL API key and email address in the CONFIG section
   below.

   IMPORTANT:
   Use an environment variable or another secure credential
   management method for production/research sharing.

3. Run:

       python download_nsrdb.py

4. After all 25 files have been downloaded successfully, continue with:

       python merge_nsrdb.py
       
EXPECTED FINAL MESSAGE
----------------------
If all downloads are successful, the script should report:

    === Download complete ===
    Success : 25
    Skipped : 0
    Failed  : 0

If some downloads fail, re-run the script. Successfully downloaded files
will be skipped automatically, and only incomplete site-year combinations
will be retried.
"""

import os
import time
import json
import requests
import pandas as pd
from io import StringIO

# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================
# Update ONLY the API key and email address below before running the script.
#
# IMPORTANT:
# - Never share your API key publicly.
# - Do not commit the key to GitHub.
# - If a key has already been exposed publicly, revoke it and generate a new one.

API_KEY = "YOUR_API_KEY"
EMAIL = "YOUR_EMAIL"

# =============================================================================
# 2. STUDY SITES
# =============================================================================
# Five representative locations are used to capture different climate regimes.

SITES = [
    {"name": "Phoenix_AZ",    "lat": 33.4484,  "lon": -112.0740, "climate": "hot_desert"},
    {"name": "LosAngeles_CA", "lat": 34.0522,  "lon": -118.2437, "climate": "mediterranean"},
    {"name": "Denver_CO",     "lat": 39.7392,  "lon": -104.9903, "climate": "semi_arid"},
    {"name": "Miami_FL",      "lat": 25.7617,  "lon":  -80.1918, "climate": "subtropical"},
    {"name": "Seattle_WA",    "lat": 47.6062,  "lon": -122.3321, "climate": "oceanic"},
]

# =============================================================================
# 3. DOWNLOAD PERIOD
# =============================================================================
# Five years of hourly data are downloaded for each site.
# Total expected files = 5 sites x 5 years = 25 files.

YEARS = [2018, 2019, 2020, 2021, 2022]

# =============================================================================
# 4. NSRDB VARIABLES
# =============================================================================
# Variables requested from the NREL NSRDB API.

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

# =============================================================================
# 5. OUTPUT AND CHECKPOINT SETTINGS
# =============================================================================

OUTPUT_DIR      = "nsrdb_data"
CHECKPOINT_FILE = "download_checkpoint.json"

# =============================================================================
# 6. NREL NSRDB API POINT
# =============================================================================

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
    """
    Load previously completed site-year combinations.

    Returns
    -------
    set
        A set of tuples in the form:
            {("Phoenix_AZ", 2018), ("Denver_CO", 2019), ...}

        Returns an empty set if no checkpoint file exists.
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        completed = set(tuple(x) for x in data.get("completed", []))
        print("Checkpoint loaded. Already completed: " + str(len(completed)) + " files.")
        return completed
    return set()


def save_checkpoint(completed):
    """
    Save successfully completed site-year combinations.

    Parameters
    ----------
    completed : set
        Set of (site_name, year) tuples representing completed downloads.
    """
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"completed": [list(x) for x in completed]}, f, indent=2)


def download_site_year(site, year):
    """
    Download one year of NSRDB data for a single study location.

    Parameters
    ----------
    site : dict
        Study-site information containing:
            - name
            - latitude
            - longitude
            - climate classification

    year : int
        Calendar year to download.

    Returns
    -------
    bool
        True if the download and file saving are successful;
        False if an HTTP or other error occurs.

    Processing performed
    --------------------
    1. Construct the NSRDB API request URL.
    2. Request the hourly NSRDB data.
    3. Parse the returned CSV.
    4. Add site-level metadata.
    5. Calculate the clearness index.
    6. Assign the sky regime.
    7. Save the processed data as a CSV file.
    """
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
    """
    Execute the complete NSRDB download workflow.

    The workflow:
        1. Creates the output directory.
        2. Loads the download checkpoint.
        3. Identifies completed and pending site-year combinations.
        4. Downloads each pending dataset.
        5. Saves the checkpoint after every successful download.
        6. Waits between API requests to respect the NREL rate limit.
        7. Prints a final summary of successful, skipped, and failed downloads.
    """
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
