"""
Fixed scratch.py - Test script for running complete analysis

This version will work after you replace CSWBM.py with the fixed version.
"""

import os
import sys

# Make sure we can import from C-SWBM-main directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "C-SWBM-main"))

import complete_analysis as ca

print("=" * 70)
print("RUNNING COMPLETE ANALYSIS")
print("=" * 70)

# Define your data files
# Since we're running FROM C-SWBM-main directory, paths are relative
files = [
    "Data/Data_swbm_Germany_new.csv",
    "Data/Data_swbm_Spain_new.csv",
    "Data/Data_swbm_Sweden_new.csv",
]

# Check if files exist
print("\nChecking files...")
for f in files:
    if os.path.exists(f):
        print(f"  ✓ Found: {f}")
    else:
        print(f"  ✗ Missing: {f}")
        print(f"    (Current dir: {os.getcwd()})")

print("\n" + "=" * 70)

# Run complete analysis
# This does EVERYTHING:
# - Loads data
# - Runs model
# - Calculates metrics
# - Creates all plots
# - Saves CSVs
all_results, summary = ca.run_complete_analysis(
    filepaths=files,
    output_dir="complete_outputs",
    exp_runoff=2.0,  # Runoff exponent
    exp_et=0.5,  # ET exponent
    beta=0.8,  # Priestley-Taylor coefficient
    whc=420.0,  # Water holding capacity [mm]
    create_plots=True,  # Generate visualizations
)

# Also create cross-site comparison plot
if len(all_results) > 1:
    print("\nCreating cross-site comparison...")
    ca.quick_comparison_plot(all_results, output_dir="complete_outputs")

print("\n" + "=" * 70)
print("ALL DONE!")
print("=" * 70)
print("\nCheck the 'complete_outputs/' directory for:")
print("  - CSV files with results")
print("  - PNG plots for each site")
print("  - summary_metrics.csv with all statistics")
print("  - sites_comparison.png comparing all sites")
print("\n")

# Print summary statistics
if not summary.empty:
    print("QUICK SUMMARY:")
    print("-" * 70)
    for idx, row in summary.iterrows():
        print(f"\n{row['site']}:")
        if "sm_nse" in row:
            print(f"  Soil Moisture NSE: {row['sm_nse']:.3f}")
        if "ro_nse" in row:
            print(f"  Runoff NSE:        {row['ro_nse']:.3f}")
        if "et_nse" in row:
            print(f"  ET NSE:            {row['et_nse']:.3f}")
