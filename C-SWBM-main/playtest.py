import pandas as pd
import csv

test_data = pd.read_csv("Data/Data_swbm_Germany_new.csv")
test_data[0:10:]

test_data[0:10].to_csv('test_data.csv',header=True)


"""
Example usage of Simple Water Balance Model with your test data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CSWBM import SimpleWaterBalanceModel

# ===== BASIC USAGE =====

# 1. Initialize the model with parameters
model = SimpleWaterBalanceModel(
    exp_runoff=2.0,      # Runoff exponent (controls runoff sensitivity to soil moisture)
    exp_et=0.5,          # ET exponent (controls ET sensitivity to soil moisture)
    beta=0.8,            # Beta parameter for ET calculation
    whc=150.0,           # Water holding capacity in mm
    use_snow=False       # Set to True if you want snow modeling (requires melting parameter)
)

# 2. Load your data
# The load_data method will automatically preprocess your CSV file
data = model.load_data('test_data.csv')

print("Loaded data columns:", data.columns.tolist())
print(f"Number of days: {len(data)}")
print(f"Date range: {data['time'].min()} to {data['time'].max()}")
print("\nFirst few rows of preprocessed data:")
print(data.head())

# 3. Run the model
results = model.run(data=data)

# 4. Access results
print("\n===== MODEL RESULTS =====")
print(f"Soil moisture shape: {results['soilmoisture'].shape}")
print(f"Runoff shape: {results['runoff'].shape}")
print(f"ET shape: {results['evapotranspiration'].shape}")
print(f"Snow shape: {results['snow'].shape}")

print("\nSummary statistics:")
print(f"Mean soil moisture: {np.nanmean(results['soilmoisture']):.2f} mm")
print(f"Total runoff: {np.nansum(results['runoff']):.2f} mm")
print(f"Total ET: {np.nansum(results['evapotranspiration']):.2f} mm")

# ===== EXAMPLE: QUICK VISUALIZATION =====

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Soil moisture
axes[0].plot(data['time'], results['soilmoisture'], label='Modeled', color='darkgreen')
if 'sm' in data.columns:
    axes[0].plot(data['time'], data['sm'], label='Observed', color='lightgreen', alpha=0.7)
axes[0].axhline(y=results['whc'], color='red', linestyle='--', label=f'WHC = {results["whc"]:.0f} mm')
axes[0].set_ylabel('Soil Moisture (mm)')
axes[0].set_title('Soil Moisture')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Runoff
axes[1].plot(data['time'], results['runoff'], label='Modeled', color='darkblue')
if 'ro' in data.columns:
    axes[1].plot(data['time'], data['ro'], label='Observed', color='lightblue', alpha=0.7)
axes[1].set_ylabel('Runoff (mm)')
axes[1].set_title('Runoff')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Evapotranspiration
axes[2].plot(data['time'], results['evapotranspiration'], label='Modeled', color='purple')
if 'le' in data.columns:
    axes[2].plot(data['time'], data['le'], label='Observed', color='pink', alpha=0.7)
axes[2].set_ylabel('ET (mm)')
axes[2].set_xlabel('Date')
axes[2].set_title('Evapotranspiration')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'model_results.png'")

# ===== ALTERNATIVE: WITH SNOW MODULE =====

# If you want to include snow processes, you need temperature data
# and must provide the melting parameter

model_with_snow = SimpleWaterBalanceModel(
    exp_runoff=2.0,
    exp_et=0.5,
    beta=0.8,
    whc=150.0,
    melting=3.0,        # Snow melting rate parameter
    use_snow=True       # Enable snow module
)

# Add temperature column to data if not already present
if 'temp' not in data.columns and 't2m_[K]' in pd.read_csv('test_data.csv').columns:
    raw_data = pd.read_csv('test_data.csv')
    data['temp'] = raw_data['t2m_[K]'] - 273.15  # Convert Kelvin to Celsius

results_with_snow = model_with_snow.run(data=data)

print("\n===== RESULTS WITH SNOW MODULE =====")
print(f"Total snow accumulation: {np.nansum(results_with_snow['snow']):.2f} mm")
print(f"Max snow depth: {np.nanmax(results_with_snow['snow']):.2f} mm")

# ===== CALCULATE PERFORMANCE METRICS =====

def calculate_metrics(observed, modeled):
    """Calculate performance metrics"""
    mask = ~(np.isnan(observed) | np.isnan(modeled))
    obs = observed[mask]
    mod = modeled[mask]
    
    if len(obs) == 0:
        return None
    
    rmse = np.sqrt(np.mean((obs - mod)**2))
    mae = np.mean(np.abs(obs - mod))
    corr = np.corrcoef(obs, mod)[0, 1]
    bias = np.mean(mod - obs)
    nse = 1 - (np.sum((obs - mod)**2) / np.sum((obs - np.mean(obs))**2))
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': corr,
        'Bias': bias,
        'NSE': nse
    }

print("\n===== PERFORMANCE METRICS =====")

if 'sm' in data.columns:
    sm_metrics = calculate_metrics(data['sm'].values, results['soilmoisture'])
    if sm_metrics:
        print("\nSoil Moisture:")
        for key, value in sm_metrics.items():
            print(f"  {key}: {value:.4f}")

if 'ro' in data.columns:
    ro_metrics = calculate_metrics(data['ro'].values, results['runoff'])
    if ro_metrics:
        print("\nRunoff:")
        for key, value in ro_metrics.items():
            print(f"  {key}: {value:.4f}")

if 'le' in data.columns:
    et_metrics = calculate_metrics(data['le'].values, results['evapotranspiration'])
    if et_metrics:
        print("\nEvapotranspiration:")
        for key, value in et_metrics.items():
            print(f"  {key}: {value:.4f}")

# ===== EXPORT RESULTS TO CSV =====

# Create a results dataframe
results_df = pd.DataFrame({
    'time': data['time'],
    'observed_sm': data['sm'] if 'sm' in data.columns else np.nan,
    'modeled_sm': results['soilmoisture'],
    'observed_ro': data['ro'] if 'ro' in data.columns else np.nan,
    'modeled_ro': results['runoff'],
    'observed_et': data['le'] if 'le' in data.columns else np.nan,
    'modeled_et': results['evapotranspiration'],
    'snow': results['snow'],
    'precipitation': data['tp'],
    'radiation': data['snr']
})

results_df.to_csv('model_output.csv', index=False)
print("\nResults exported to 'model_output.csv'")

print("\n===== DONE =====")