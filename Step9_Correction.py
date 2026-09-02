# GenerativeAURORA - Step 9: Number Corrections Summary
# Fixed: removed Unicode characters for Windows cp950 compatibility
#
# Run:
#   python step9_corrections.py

import os
import json
import sys

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = "C:/Users/user/Paper_10/Dataset"
EVAL_DIR    = os.path.join(DATASET_DIR, "evaluation")

print("=" * 65)
print("GenerativeAURORA - PAPER NUMBER CORRECTIONS")
print("From actual step6 + step7 terminal output")
print("=" * 65)

print("")
print("--- SECTION III-C (RCCC Calibration Quantiles) ---")
print("OLD (paper)         --> NEW (actual step7 output)")
print("Clear   q = 12.35   --> q = 11.71  W/m2")
print("Cloudy  q = 21.44   --> q = 20.30  W/m2")
print("Overcast q = 64.81  --> q = 65.49  W/m2")

print("")
print("--- TABLE II (Overall Test Metrics - Gen.AURORA+RCCC row) ---")
print("CRPS        : 20.70   (unchanged)")
print("Pb p10      : 7.55    (was 7.60)")
print("Pb p50      : 13.97   (was 13.94)")
print("Coverage 90%: 96.12%  (was 96.1%)")
print("Width       : 151.07  (was 151.6)")
print("MAE         : 27.93   (was 27.88)")

print("")
print("--- TABLE III (Regime Metrics - Gen.AURORA+RCCC) ---")
print("Regime     N    CRPS   Pb_p50  Cov90   Cov80   Width   MAE")
print("Clear    866   20.95   14.10   96.22%  88.04%  154.16  28.21")
print("Cloudy    49   16.94   11.78   95.15%  71.94%  101.40  23.55")
print("Overcast   5   24.50   13.51   88.75%  37.50%  101.79  27.02")

print("")
print("--- TABLE IV (Before vs After RCCC) ---")
print("Regime     N    Cov_before  Width_bef  Cov_after  Width_aft  Delta_pp")
print("Clear    866    93.92%      136.19     96.22%     154.16     +2.30")
print("Cloudy    49    83.42%       74.36     95.15%     101.40    +11.73")
print("Overcast   5    47.50%       19.79     88.75%     101.79    +41.25")
print("")
print("IMPORTANT NOTE:")
print("  Overcast before was 42.5% in paper --> ACTUAL is 47.5%")
print("  Delta was +46.25pp in paper        --> ACTUAL is +41.25pp")

print("")
print("--- TABLE V (Per-Site Metrics) ---")
print("Site             N    CRPS   Cov90   Cov80   Width   MAE")
print("Phoenix AZ     184   17.83   96.33%  87.36%  136.87  23.78")
print("Los Angeles CA 184   15.43   96.81%  88.89%  119.46  20.45")
print("Denver CO      184   25.56   95.24%  85.46%  171.65  34.82")
print("Miami FL       184   27.20   95.82%  88.15%  197.07  36.14")
print("Seattle WA     184   17.76   96.40%  84.68%  130.28  24.57")

print("")
print("--- ABSTRACT + SECTION V-B (Key claim corrections) ---")
print("OLD: Overcast from 42.5% to 88.75% (+46.25 pp)")
print("NEW: Overcast from 47.5% to 88.75% (+41.25 pp)")
print("")
print("OLD: Cloudy from 83.2% to 95.0% (+11.86 pp)")
print("NEW: Cloudy from 83.4% to 95.2% (+11.73 pp)")
print("")
print("OLD: Clear from 93.7% to 96.2% (+2.52 pp)")
print("NEW: Clear from 93.9% to 96.2% (+2.30 pp)")

print("")
print("--- TABLE VI (Ablation - A1 row) ---")
print("CRPS         : 20.70  (unchanged)")
print("Cov Overcast : 88.75% (unchanged)")
print("MAE          : 27.93  (was 27.88 or 27.98)")

print("")
print("--- SECTION IV-C (Training Details) ---")
print("Best epoch   : 181    (unchanged)")
print("Val loss     : 0.0256 (unchanged)")
print("Parameters   : 5.09M  (confirmed: 5,095,329)")

print("")
print("--- SECTION III-B (MLP Fusion) ---")
print("OLD: 3-Layer MLP Fusion")
print("NEW: 2-Layer MLP Fusion  (256->128 SiLU->64)")
print("     This must be fixed in paper text AND figures")

print("")
print("=" * 65)
print("FULL ACTION LIST - CORRECTIONS NEEDED IN PAPER")
print("=" * 65)
print("")

corrections = [
    ("Abstract",     "Overcast +46.25pp",       "Overcast +41.25pp"),
    ("Abstract",     "Overcast from 42.5%",      "Overcast from 47.5%"),
    ("Abstract",     "Cloudy +11.86pp",           "Cloudy +11.73pp"),
    ("Contribution3","42.5% before RCCC",         "47.5% before RCCC"),
    ("Contribution3","+46.25pp",                  "+41.25pp"),
    ("Sec III-B",    "3-Layer MLP Fusion",        "2-Layer MLP Fusion"),
    ("Sec III-C",    "Clear q=12.35 W/m2",        "Clear q=11.71 W/m2"),
    ("Sec III-C",    "Cloudy q=21.44 W/m2",       "Cloudy q=20.30 W/m2"),
    ("Sec III-C",    "Overcast q=64.81 W/m2",     "Overcast q=65.49 W/m2"),
    ("Sec III-C",    "5.3x larger than Clear",    "5.6x larger than Clear"),
    ("Table II",     "Pb p10: 7.60",              "Pb p10: 7.55"),
    ("Table II",     "Pb p50: 13.94",             "Pb p50: 13.97"),
    ("Table II",     "Width: 151.6",              "Width: 151.07"),
    ("Table II",     "MAE: 27.88",                "MAE: 27.93"),
    ("Table III",    "Cov 90%: 88.8%",            "Cov 90%: 88.75%"),
    ("Table III",    "MAE: 28.11",                "MAE: 28.21"),
    ("Table IV",     "Clear before 93.72%",       "Clear before 93.92%"),
    ("Table IV",     "Clear delta +2.52pp",       "Clear delta +2.30pp"),
    ("Table IV",     "Cloudy before 83.16%",      "Cloudy before 83.42%"),
    ("Table IV",     "Cloudy after 95.03%",       "Cloudy after 95.15%"),
    ("Table IV",     "Cloudy delta +11.86pp",     "Cloudy delta +11.73pp"),
    ("Table IV",     "Overcast before 42.5%",     "Overcast before 47.5%"),
    ("Table IV",     "Overcast delta +46.25pp",   "Overcast delta +41.25pp"),
    ("Table VI A1",  "MAE: 27.88 or 27.98",       "MAE: 27.93"),
    ("Table VI A2",  "Overcast 42.5% or 45.0%",   "Overcast 47.5%"),
    ("Sec V-B",      "42.5% deficit of 47.5pp",   "47.5% deficit of 42.5pp"),
    ("Sec V-B",      "q=64.81 is 5.3x Clear",     "q=65.49 is 5.6x Clear"),
    ("Conclusion",   "+46.25pp Overcast",          "+41.25pp Overcast"),
    ("Conclusion",   "from 42.5% to 88.75%",      "from 47.5% to 88.75%"),
]

print(f"{'#':<4} {'Location':<18} {'OLD (wrong)':<35} {'NEW (correct)'}")
print("-" * 90)
for i, (loc, old, new) in enumerate(corrections, 1):
    print(f"{i:<4} {loc:<18} {old:<35} {new}")

print("")
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"Total corrections needed : {len(corrections)}")
print("Numbers unchanged        : CRPS 20.70, Coverage 96.1%, 5.09M params")
print("Most critical fix        : Overcast 42.5% -> 47.5% (affects abstract)")
print("=" * 65)

# -----------------------------------------------------------------------
# Try to load JSON for verification
# -----------------------------------------------------------------------
metrics_path = os.path.join(EVAL_DIR, "scenario_metrics.json")
rccc_path    = os.path.join(EVAL_DIR, "rccc_results.csv")

print("")
print("--- Verification from saved files ---")
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        sm = json.load(f)
    print("scenario_metrics.json found:")
    for k, v in sm.items():
        print("  " + str(k) + " : " + str(v))
else:
    print("scenario_metrics.json not found - using terminal output values above")

if os.path.exists(rccc_path):
    print("rccc_results.csv found - cross-checking coverage values...")
    import csv
    with open(rccc_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("group") == "regime":
                print("  " + row.get("name","") +
                      "  before=" + str(row.get("coverage_before","")) +
                      "  after="  + str(row.get("coverage_after","")) +
                      "  delta="  + str(row.get("delta_coverage","")))
else:
    print("rccc_results.csv not found - using terminal output values above")

print("")
print("Done. Fix all items in the ACTION LIST above before submission.")