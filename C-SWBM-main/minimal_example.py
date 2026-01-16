"""
MINIMAL WORKING EXAMPLE
Simple Water Balance Model

This script shows the simplest way to run the model and get results.
"""

from CSWBM import SimpleWaterBalanceModel
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Initialize the model with parameters
# =============================================================================
print("Step 1: Initializing model...")

model = SimpleWaterBalanceModel(
    exp_runoff=2.0,    # Runoff exponent (1-5, higher = more threshold)
    exp_et=0.8,        # ET exponent (0.1-2, higher = stronger drying effect)
    beta=0.6,          # Priestley-Taylor coefficient (0.5-1.2)
    whc=420.0,         # Water holding capacity in mm (50-300)
    delta=0.4,
    )

print("✓ Model initialized")


# =============================================================================
# STEP 2: Load your data
# =============================================================================
print("\nStep 2: Loading data...")

# Load data from CSV file
# Expected columns: time, latitude, longitude, snr_[MJ/m2], tp_[mm], 
#                   ro_[m], sm_[m3/m3], le_[W/m2], t2m_[K]
data = model.load_data('Data/Data_swbm_Germany_new.csv')

print(f"✓ Loaded {len(data)} days of data")
print(f"  Date range: {data['time'].min()} to {data['time'].max()}")
print(f"  Columns: {list(data.columns)}")


# =============================================================================
# STEP 3: Run the model
# =============================================================================
print("\nStep 3: Running model...")

results = model.run()

print("✓ Model run complete!")
print(f"  Generated {results['length']} days of output")


# =============================================================================
# STEP 4: Look at the results
# =============================================================================
print("\nStep 4: Examining results...")

# Results is a dictionary with:
# - soilmoisture: soil moisture time series [mm]
# - runoff: runoff time series [mm/day]
# - evapotranspiration: ET time series [mm/day]
# - parameters: exp_runoff, exp_et, beta, whc

print(f"\nSoil Moisture:")
print(f"  Mean: {results['soilmoisture'].mean():.1f} mm")
print(f"  Min:  {results['soilmoisture'].min():.1f} mm")
print(f"  Max:  {results['soilmoisture'].max():.1f} mm")

print(f"\nRunoff:")
print(f"  Total: {results['runoff'].sum():.1f} mm")
print(f"  Mean:  {results['runoff'].mean():.1f} mm/day")

print(f"\nEvapotranspiration:")
print(f"  Total: {results['evapotranspiration'].sum():.1f} mm")
print(f"  Mean:  {results['evapotranspiration'].mean():.1f} mm/day")


# =============================================================================
# STEP 5: Create a simple plot
# =============================================================================
print("\nStep 5: Creating plot...")

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Plot 1: Soil Moisture
axes[0].plot(data['time'], results['soilmoisture'], label='Modeled', color='darkgreen')
if 'sm' in data.columns:
    axes[0].plot(data['time'], data['sm'], label='Observed', color='green', alpha=0.5)
axes[0].axhline(y=results['whc'], color='red', linestyle='--', label=f'WHC = {results["whc"]:.0f} mm')
axes[0].set_ylabel('Soil Moisture (mm)')
axes[0].set_title('Soil Moisture')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Runoff
axes[1].plot(data['time'], results['runoff'], label='Modeled', color='darkblue')
if 'ro' in data.columns:
    axes[1].plot(data['time'], data['ro'], label='Observed', color='blue', alpha=0.5)
axes[1].set_ylabel('Runoff (mm/day)')
axes[1].set_title('Runoff')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Evapotranspiration
axes[2].plot(data['time'], results['evapotranspiration'], label='Modeled', color='darkred')
if 'le' in data.columns:
    axes[2].plot(data['time'], data['le'], label='Observed', color='red', alpha=0.5)
axes[2].set_ylabel('ET (mm/day)')
axes[2].set_xlabel('Date')
axes[2].set_title('Evapotranspiration')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('minimal_output.png', dpi=150)
print("✓ Plot saved as 'minimal_output.png'")


# =============================================================================
# STEP 6: Save results to CSV (optional)
# =============================================================================
print("\nStep 6: Saving results...")

# Create output dataframe
output_df = pd.DataFrame({
    'time': data['time'],
    'precipitation': data['tp'],
    'radiation': data['snr'],
    'soil_moisture_modeled': results['soilmoisture'],
    'runoff_modeled': results['runoff'],
    'et_modeled': results['evapotranspiration']
})

# Add observations if available
if 'sm' in data.columns:
    output_df['soil_moisture_observed'] = data['sm']
if 'ro' in data.columns:
    output_df['runoff_observed'] = data['ro']
if 'le' in data.columns:
    output_df['et_observed'] = data['le']

output_df.to_csv('minimal_results.csv', index=False)
print("✓ Results saved as 'minimal_results.csv'")

print("\n" + "="*70)
print("DONE! Check your directory for:")
print("  - minimal_output.png (plot)")
print("  - minimal_results.csv (detailed results)")
print("="*70)

plt.show()
