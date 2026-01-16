"""
Complete calibration and validation script for Simple Water Balance Model
Follows the assignment structure:
- Step 1: Run 243 parameter combinations on 2008-2013 (calibration)
- Step 2: Select best parameters based on sum of correlations
- Step 3: Validate on 2014-2018
- Step 4: Generate performance table
"""

import numpy as np
import pandas as pd
from itertools import product
from CSWBM import SimpleWaterBalanceModel
import os
from datetime import datetime
import json


def calculate_metrics(obs, mod):
    """
    Calculate performance metrics between observed and modeled data.
    
    Parameters
    ----------
    obs : array
        Observed values
    mod : array
        Modeled values
    
    Returns
    -------
    dict : Performance metrics (correlation, RMSE, NSE, bias)
    """
    # Remove NaN values
    mask = ~(np.isnan(obs) | np.isnan(mod))
    obs_clean = obs[mask]
    mod_clean = mod[mask]
    
    if len(obs_clean) < 10:  # Need minimum data points
        return {
            'correlation': np.nan,
            'rmse': np.nan,
            'nse': np.nan,
            'bias': np.nan,
            'n_points': len(obs_clean)
        }
    
    # Calculate metrics
    correlation = np.corrcoef(obs_clean, mod_clean)[0, 1]
    rmse = np.sqrt(np.mean((obs_clean - mod_clean)**2))
    nse = 1 - (np.sum((obs_clean - mod_clean)**2) / 
               np.sum((obs_clean - np.mean(obs_clean))**2))
    bias = np.mean(mod_clean - obs_clean)
    
    return {
        'correlation': correlation,
        'rmse': rmse,
        'nse': nse,
        'bias': bias,
        'n_points': len(obs_clean)
    }


def run_single_simulation(data, whc, exp_runoff, exp_et, beta, delta):
    """
    Run model with single parameter combination.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    whc : float
        Water holding capacity (cs in assignment)
    exp_runoff : float
        Runoff exponent (α in assignment)
    exp_et : float
        ET exponent (γ in assignment)
    beta : float
        Beta parameter (β in assignment)
    delta : float
        Fast runoff fraction (new parameter)
    
    Returns
    -------
    results : dict
        Model results
    metrics : dict
        Performance metrics for each variable
    """
    try:
        # Initialize model
        model = SimpleWaterBalanceModel(
            exp_runoff=exp_runoff,
            exp_et=exp_et,
            beta=beta,
            whc=whc,
            delta=delta
        )
        
        # Run model
        results = model.run(data=data)
        
        # Calculate metrics for each variable
        metrics = {}
        
        # Soil moisture
        if 'sm' in data.columns:
            sm_metrics = calculate_metrics(data['sm'].values, results['soilmoisture'])
            metrics['sm'] = sm_metrics
        
        # Runoff
        if 'ro' in data.columns:
            ro_metrics = calculate_metrics(data['ro'].values, results['runoff'])
            metrics['ro'] = ro_metrics
        
        # Evapotranspiration
        if 'le' in data.columns:
            et_metrics = calculate_metrics(data['le'].values, results['evapotranspiration'])
            metrics['et'] = et_metrics
        
        # Calculate sum of correlations (for ranking)
        sum_corr = 0
        n_vars = 0
        for var in ['sm', 'ro', 'et']:
            if var in metrics and not np.isnan(metrics[var]['correlation']):
                sum_corr += metrics[var]['correlation']
                n_vars += 1
        
        metrics['sum_correlation'] = sum_corr
        metrics['n_variables'] = n_vars
        
        return results, metrics
    
    except Exception as e:
        print(f"    Error in simulation: {e}")
        return None, None


def calibrate_site(filepath, site_name, output_dir='calibration_results'):
    """
    Step 1 & 2: Calibrate model on 2008-2013 data.
    
    Parameters
    ----------
    filepath : str
        Path to data file
    site_name : str
        Name of site (e.g., 'Germany', 'Spain', 'Sweden')
    output_dir : str
        Directory for outputs
    
    Returns
    -------
    best_params : dict
        Best parameter combination
    best_metrics : dict
        Performance metrics for best parameters
    all_results : pd.DataFrame
        Results for all parameter combinations
    """
    print(f"\n{'='*70}")
    print(f"CALIBRATION: {site_name}")
    print(f"{'='*70}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {filepath}...")
    model_temp = SimpleWaterBalanceModel(2.0, 0.5, 0.8, 150.0, 0.3)
    data_full = model_temp.load_data(filepath)
    
    # Filter for calibration period (2008-2013)
    data = data_full[(data_full['time'] >= '2008-01-01') & 
                     (data_full['time'] <= '2013-12-31')].reset_index(drop=True)
    
    print(f"Calibration period: {data['time'].min()} to {data['time'].max()}")
    print(f"Number of days: {len(data)}")
    
    # Define parameter ranges (as specified in assignment)
    param_grid = {
        'whc': [210.0, 420.0, 840.0],              # cs in assignment
        'exp_runoff': [2.0, 4.0, 8.0],             # α in assignment
        'exp_et': [0.2, 0.5, 0.8],                 # γ in assignment
        'beta': [0.4, 0.6, 0.8],                   # β in assignment
        'delta': [0.2, 0.4, 0.8]                   # New parameter (fast runoff fraction)
    }
    
    print(f"\nParameter ranges:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    n_combinations = len(combinations)
    print(f"\nParameter Combinations: {combinations}") 
    print(f"\nTotal parameter combinations: {n_combinations}")
    print(f"Expected: 3^5 = 243")
    assert n_combinations == 243, f"Expected 243 combinations, got {n_combinations}"
    
    # Run all combinations
    print(f"\nRunning {n_combinations} simulations...")
    results_list = []
    
    for idx, combo in enumerate(combinations, 1):
        if idx % 50 == 0 or idx == 1:
            print(f"  Progress: {idx}/{n_combinations} ({100*idx/n_combinations:.1f}%)")
        
        # Unpack parameters
        whc, exp_runoff, exp_et, beta, delta = combo
        
        # Run simulation
        results, metrics = run_single_simulation(
            data, whc, exp_runoff, exp_et, beta, delta
        )
        
        if metrics is not None:
            # Store results
            result_row = {
                'whc': whc,
                'exp_runoff': exp_runoff,
                'exp_et': exp_et,
                'beta': beta,
                'delta': delta,
                'sum_correlation': metrics['sum_correlation'],
                'n_variables': metrics['n_variables']
            }
            
            # Add individual variable metrics
            for var in ['sm', 'ro', 'et']:
                if var in metrics:
                    result_row[f'{var}_corr'] = metrics[var]['correlation']
                    result_row[f'{var}_rmse'] = metrics[var]['rmse']
                    result_row[f'{var}_nse'] = metrics[var]['nse']
                    result_row[f'{var}_bias'] = metrics[var]['bias']
            
            results_list.append(result_row)
    
    # Create results dataframe
    all_results = pd.DataFrame(results_list)
    
    # Save all results
    all_results_path = os.path.join(output_dir, f'{site_name}_all_calibrations.csv')
    all_results.to_csv(all_results_path, index=False)
    print(f"\n✓ Saved all calibration results to: {all_results_path}")
    
    # Find best parameters (highest sum of correlations)
    best_idx = all_results['sum_correlation'].idxmax()
    best_row = all_results.iloc[best_idx]
    
    best_params = {
        'whc': best_row['whc'],
        'exp_runoff': best_row['exp_runoff'],
        'exp_et': best_row['exp_et'],
        'beta': best_row['beta'],
        'delta': best_row['delta']
    }
    
    best_metrics = {
        'sum_correlation': best_row['sum_correlation'],
        'sm_corr': best_row.get('sm_corr', np.nan),
        'ro_corr': best_row.get('ro_corr', np.nan),
        'et_corr': best_row.get('et_corr', np.nan),
        'sm_nse': best_row.get('sm_nse', np.nan),
        'ro_nse': best_row.get('ro_nse', np.nan),
        'et_nse': best_row.get('et_nse', np.nan)
    }
    
    print(f"\n{'='*70}")
    print(f"BEST PARAMETERS FOR {site_name} (2008-2013):")
    print(f"{'='*70}")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\nCalibration Performance:")
    print(f"  Sum of correlations: {best_metrics['sum_correlation']:.4f}")
    print(f"  Soil Moisture corr:  {best_metrics['sm_corr']:.4f}")
    print(f"  Runoff corr:         {best_metrics['ro_corr']:.4f}")
    print(f"  ET corr:             {best_metrics['et_corr']:.4f}")
    
    return best_params, best_metrics, all_results


def validate_site(filepath, site_name, best_params, output_dir='calibration_results'):
    """
    Step 3: Validate model on 2014-2018 data using best parameters.
    
    Parameters
    ----------
    filepath : str
        Path to data file
    site_name : str
        Name of site
    best_params : dict
        Best parameter combination from calibration
    output_dir : str
        Directory for outputs
    
    Returns
    -------
    validation_metrics : dict
        Performance metrics on validation period
    """
    print(f"\n{'='*70}")
    print(f"VALIDATION: {site_name}")
    print(f"{'='*70}")
    
    # Load data
    model_temp = SimpleWaterBalanceModel(2.0, 0.5, 0.8, 150.0, 0.3)
    data_full = model_temp.load_data(filepath)
    
    # Filter for validation period (2014-2018)
    data = data_full[(data_full['time'] >= '2014-01-01') & 
                     (data_full['time'] <= '2018-12-31')].reset_index(drop=True)
    
    print(f"Validation period: {data['time'].min()} to {data['time'].max()}")
    print(f"Number of days: {len(data)}")
    
    print(f"\nUsing best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Run model with best parameters
    results, metrics = run_single_simulation(
        data,
        whc=best_params['whc'],
        exp_runoff=best_params['exp_runoff'],
        exp_et=best_params['exp_et'],
        beta=best_params['beta'],
        delta=best_params['delta']
    )
    
    if metrics is None:
        print("ERROR: Validation failed!")
        return None
    
    # Extract validation metrics
    validation_metrics = {
        'sum_correlation': metrics['sum_correlation'],
        'sm_corr': metrics['sm']['correlation'] if 'sm' in metrics else np.nan,
        'ro_corr': metrics['ro']['correlation'] if 'ro' in metrics else np.nan,
        'et_corr': metrics['et']['correlation'] if 'et' in metrics else np.nan,
        'sm_nse': metrics['sm']['nse'] if 'sm' in metrics else np.nan,
        'ro_nse': metrics['ro']['nse'] if 'ro' in metrics else np.nan,
        'et_nse': metrics['et']['nse'] if 'et' in metrics else np.nan,
        'sm_rmse': metrics['sm']['rmse'] if 'sm' in metrics else np.nan,
        'ro_rmse': metrics['ro']['rmse'] if 'ro' in metrics else np.nan,
        'et_rmse': metrics['et']['rmse'] if 'et' in metrics else np.nan
    }
    
    print(f"\nValidation Performance (2014-2018):")
    print(f"  Sum of correlations: {validation_metrics['sum_correlation']:.4f}")
    print(f"  Soil Moisture corr:  {validation_metrics['sm_corr']:.4f}")
    print(f"  Runoff corr:         {validation_metrics['ro_corr']:.4f}")
    print(f"  ET corr:             {validation_metrics['et_corr']:.4f}")
    
    # Save detailed validation results
    results_df = pd.DataFrame({
        'time': data['time'],
        'precipitation': data['tp'],  # NEW: Add precipitation
        'radiation': data['snr'],      # NEW: Add radiation
        'observed_sm': data['sm'] if 'sm' in data.columns else np.nan,
        'modeled_sm': results['soilmoisture'],
        'observed_ro': data['ro'] if 'ro' in data.columns else np.nan,
        'modeled_ro': results['runoff'],
        'observed_et': data['le'] if 'le' in data.columns else np.nan,
        'modeled_et': results['evapotranspiration'],
        'gw_storage': results['gw_storage'],
        'baseflow': results['baseflow'],
        'fast_runoff': results['fast_runoff']
    })
    
    validation_path = os.path.join(output_dir, f'{site_name}_validation_timeseries.csv')
    results_df.to_csv(validation_path, index=False)
    print(f"✓ Saved validation timeseries to: {validation_path}")
    
    return validation_metrics


def generate_performance_table(sites_results, output_dir='calibration_results'):
    """
    Step 4: Generate final performance table.
    
    Parameters
    ----------
    sites_results : dict
        Dictionary with site names as keys and results as values
    output_dir : str
        Directory for outputs
    """
    print(f"\n{'='*70}")
    print(f"GENERATING PERFORMANCE TABLE")
    print(f"{'='*70}")
    
    # Create performance table
    table_rows = []
    
    for site_name, results in sites_results.items():
        row = {
            'Site': site_name,
            # Best parameters
            'whc': results['best_params']['whc'],
            'exp_runoff': results['best_params']['exp_runoff'],
            'exp_et': results['best_params']['exp_et'],
            'beta': results['best_params']['beta'],
            'delta': results['best_params']['delta'],
            # Calibration performance
            'calib_sum_corr': results['calib_metrics']['sum_correlation'],
            'calib_sm_corr': results['calib_metrics']['sm_corr'],
            'calib_ro_corr': results['calib_metrics']['ro_corr'],
            'calib_et_corr': results['calib_metrics']['et_corr'],
            # Validation performance
            'valid_sum_corr': results['valid_metrics']['sum_correlation'],
            'valid_sm_corr': results['valid_metrics']['sm_corr'],
            'valid_ro_corr': results['valid_metrics']['ro_corr'],
            'valid_et_corr': results['valid_metrics']['et_corr'],
            'valid_sm_nse': results['valid_metrics']['sm_nse'],
            'valid_ro_nse': results['valid_metrics']['ro_nse'],
            'valid_et_nse': results['valid_metrics']['et_nse']
        }
        table_rows.append(row)
    
    performance_table = pd.DataFrame(table_rows)
    
    # Save table
    table_path = os.path.join(output_dir, 'MODEL_PERFORMANCE_TABLE.csv')
    performance_table.to_csv(table_path, index=False)
    
    print(f"\n✓ Saved performance table to: {table_path}")
    print(f"\n{performance_table.to_string(index=False)}")
    
    # Also save as formatted text
    text_path = os.path.join(output_dir, 'MODEL_PERFORMANCE_TABLE.txt')
    with open(text_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("MODEL PERFORMANCE TABLE\n")
        f.write("="*100 + "\n\n")
        f.write(performance_table.to_string(index=False))
        f.write("\n\n" + "="*100 + "\n")
    
    print(f"✓ Saved formatted table to: {text_path}")
    
    return performance_table


def run_complete_calibration_validation(file_dict, output_dir='calibration_results'):
    """
    Run complete calibration and validation workflow for all sites.
    
    Parameters
    ----------
    file_dict : dict
        Dictionary with site names as keys and file paths as values
        Example: {'Germany': 'Data/Data_swbm_Germany_new.csv', ...}
    output_dir : str
        Directory for outputs
    
    Returns
    -------
    sites_results : dict
        Complete results for all sites
    performance_table : pd.DataFrame
        Final performance table
    """
    print("="*70)
    print("COMPLETE CALIBRATION AND VALIDATION WORKFLOW")
    print("="*70)
    print(f"\nSites to process: {list(file_dict.keys())}")
    print(f"Output directory: {output_dir}")
    
    sites_results = {}
    
    for site_name, filepath in file_dict.items():
        print(f"\n\n{'#'*70}")
        print(f"# PROCESSING SITE: {site_name}")
        print(f"{'#'*70}")
        
        # Step 1 & 2: Calibration
        best_params, calib_metrics, all_calibs = calibrate_site(
            filepath, site_name, output_dir
        )
        
        # Step 3: Validation
        valid_metrics = validate_site(
            filepath, site_name, best_params, output_dir
        )
        
        # Store results
        sites_results[site_name] = {
            'best_params': best_params,
            'calib_metrics': calib_metrics,
            'valid_metrics': valid_metrics,
            'all_calibrations': all_calibs
        }
    
    # Step 4: Generate performance table
    performance_table = generate_performance_table(sites_results, output_dir)
    
    # Save summary JSON
    summary = {}
    for site_name, results in sites_results.items():
        summary[site_name] = {
            'best_parameters': results['best_params'],
            'calibration_performance': {
                k: float(v) if not np.isnan(v) else None 
                for k, v in results['calib_metrics'].items()
            },
            'validation_performance': {
                k: float(v) if not np.isnan(v) else None 
                for k, v in results['valid_metrics'].items()
            }
        }
    
    json_path = os.path.join(output_dir, 'calibration_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print("WORKFLOW COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - MODEL_PERFORMANCE_TABLE.csv (final results)")
    print(f"  - MODEL_PERFORMANCE_TABLE.txt (formatted)")
    print(f"  - calibration_summary.json (JSON format)")
    print(f"  - [Site]_all_calibrations.csv (all 243 combinations)")
    print(f"  - [Site]_validation_timeseries.csv (validation data)")
    
    return sites_results, performance_table


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    
    # Define your data files
    files = {
        'Germany': 'Data/Data_swbm_Germany_new.csv',
        'Spain': 'Data/Data_swbm_Spain_new.csv',
        'Sweden': 'Data/Data_swbm_Sweden_new.csv'
    }
    
    # Check if files exist
    print("Checking data files...")
    all_exist = True
    for site, filepath in files.items():
        if os.path.exists(filepath):
            print(f"  ✓ {site}: {filepath}")
        else:
            print(f"  ✗ {site}: {filepath} NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\nERROR: Some data files are missing!")
        print("Please update the file paths in the script.")
        exit(1)
    
    # Run complete workflow
    print("\n" + "="*70)
    print("STARTING CALIBRATION AND VALIDATION")
    print("="*70)
    print("\nThis will:")
    print("  1. Run 243 parameter combinations per site (calibration 2008-2013)")
    print("  2. Select best parameters for each site")
    print("  3. Validate on 2014-2018")
    print("  4. Generate performance table")
    print("\nEstimated time: ~5-15 minutes depending on your system")
    
    input("\nPress Enter to continue...")
    
    start_time = datetime.now()
    
    sites_results, performance_table = run_complete_calibration_validation(
        files,
        output_dir='calibration_results'
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"{'='*70}")
    print("\n✓ All done! Check 'calibration_results/' directory for outputs.")
