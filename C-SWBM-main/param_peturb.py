import itertools
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from CSWBM import SimpleWaterBalanceModel, prepro

# Import your model (assuming it's in the same directory or installed)
# If the model is in a different file, you might need:
# from your_module import SimpleWaterBalanceModel, prepro

def create_parameter_combinations():
    """
    Create all combinations of parameter values.
    
    Returns
    -------
    list of dict
        List of dictionaries with all parameter combinations
    """
    # Define parameter ranges from your specification
    param_ranges = {
        'cs': [210, 420, 840],          # Soil water holding capacity (mm)
        'alpha': [2, 4, 8],             # Runoff function shape (α)
        'gamma': [0.2, 0.5, 0.8],       # ET function shape (γ)
        'beta': [0.4, 0.6, 0.8]         # Maximum of ET function (β)
    }
    
    # Generate all combinations
    param_names = list(param_ranges.keys())
    value_combinations = list(itertools.product(*param_ranges.values()))
    
    # Create list of parameter dictionaries
    combinations = []
    for values in value_combinations:
        param_dict = dict(zip(param_names, values))
        combinations.append(param_dict)
    
    print(f"Created {len(combinations)} parameter combinations")
    return combinations

def run_parameter_sweep(data_path, output_dir="parameter_sweep_results"):
    """
    Run the model with all parameter combinations.
    
    Parameters
    ----------
    data_path : str
        Path to input CSV data file
    output_dir : str, optional
        Directory to save results (default: "parameter_sweep_results")
        
    Returns
    -------
    dict
        Dictionary containing all results and metadata
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data once
    print(f"Loading data from: {data_path}")
    raw_data = pd.read_csv(data_path)
    data = prepro(raw_data)
    
    # Get forcing data
    precip = data['tp'].values.copy()
    rad = data['snr'].values.copy()
    
    # Create parameter combinations
    param_combinations = create_parameter_combinations()
    
    # Initialize results storage
    results = {
        'metadata': {
            'data_file': data_path,
            'num_combinations': len(param_combinations),
            'run_datetime': datetime.now().isoformat(),
            'parameter_ranges': {
                'cs': [210, 420, 840],
                'alpha': [2, 4, 8],
                'gamma': [0.2, 0.5, 0.8],
                'beta': [0.4, 0.6, 0.8]
            }
        },
        'results': []
    }
    
    # Run model for each parameter combination
    for i, params in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"Running simulation {i+1}/{len(param_combinations)}")
        print(f"Parameters: cs={params['cs']}, α={params['alpha']}, "
              f"γ={params['gamma']}, β={params['beta']}")
        
        try:
            # Initialize model with current parameters
            # Note: Your model uses parameter names: exp_runoff, exp_et, beta, whc
            model = SimpleWaterBalanceModel(
                exp_runoff=params['alpha'],  # α is runoff exponent
                exp_et=params['gamma'],      # γ is ET exponent
                beta=params['beta'],         # β is Priestley-Taylor coefficient
                whc=params['cs']             # cs is water holding capacity
            )
            
            # Run simulation
            start_time = datetime.now()
            model_results = model.run(precip=precip, rad=rad)
            run_time = (datetime.now() - start_time).total_seconds()
            
            # Store results with metadata
            result_entry = {
                'parameters': params,
                'run_time_seconds': run_time,
                'model_outputs': {
                    'soilmoisture': model_results['soilmoisture'].tolist(),
                    'runoff': model_results['runoff'].tolist(),
                    'evapotranspiration': model_results['evapotranspiration'].tolist()
                },
                'summary_statistics': {
                    'mean_soilmoisture': float(np.nanmean(model_results['soilmoisture'])),
                    'mean_runoff': float(np.nanmean(model_results['runoff'])),
                    'mean_et': float(np.nanmean(model_results['evapotranspiration'])),
                    'total_runoff': float(np.nansum(model_results['runoff'])),
                    'total_et': float(np.nansum(model_results['evapotranspiration'])),
                    'soilmoisture_range': [
                        float(np.nanmin(model_results['soilmoisture'])),
                        float(np.nanmax(model_results['soilmoisture']))
                    ]
                }
            }
            
            results['results'].append(result_entry)
            print(f"✓ Completed in {run_time:.2f} seconds")
            
        except Exception as e:
            print(f"✗ Error with parameters {params}: {str(e)}")
            # Store error information
            results['results'].append({
                'parameters': params,
                'error': str(e),
                'run_time_seconds': None,
                'model_outputs': None,
                'summary_statistics': None
            })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"parameter_sweep_{timestamp}.json")
    
    # Save full results (might be large)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Saved full results to: {output_file}")
    
    # Also create a summary CSV for easier analysis
    create_summary_csv(results, output_dir, timestamp)
    
    return results

def create_summary_csv(results, output_dir, timestamp):
    """
    Create a CSV summary of the parameter sweep results.
    
    Parameters
    ----------
    results : dict
        Full results dictionary
    output_dir : str
        Output directory path
    timestamp : str
        Timestamp for filename
    """
    summary_data = []
    
    for result in results['results']:
        if result.get('error'):
            # Skip error cases
            continue
            
        row = {
            'cs': result['parameters']['cs'],
            'alpha': result['parameters']['alpha'],
            'gamma': result['parameters']['gamma'],
            'beta': result['parameters']['beta'],
            'run_time_seconds': result['run_time_seconds'],
            'mean_soilmoisture': result['summary_statistics']['mean_soilmoisture'],
            'mean_runoff': result['summary_statistics']['mean_runoff'],
            'mean_et': result['summary_statistics']['mean_et'],
            'total_runoff': result['summary_statistics']['total_runoff'],
            'total_et': result['summary_statistics']['total_et'],
            'soilmoisture_min': result['summary_statistics']['soilmoisture_range'][0],
            'soilmoisture_max': result['summary_statistics']['soilmoisture_range'][1]
        }
        summary_data.append(row)
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        csv_file = os.path.join(output_dir, f"parameter_sweep_summary_{timestamp}.csv")
        df_summary.to_csv(csv_file, index=False)
        print(f"Saved summary CSV to: {csv_file}")
        
        # Print a quick summary
        print("\nParameter Sensitivity Summary:")
        print("-" * 50)
        for param in ['cs', 'alpha', 'gamma', 'beta']:
            if param in df_summary.columns:
                grouped = df_summary.groupby(param)['mean_runoff'].mean()
                print(f"{param}: {grouped.to_dict()}")
    else:
        print("No successful simulations to summarize")

def analyze_sensitivity(results):
    """
    Analyze parameter sensitivity from results.
    
    Parameters
    ----------
    results : dict
        Results dictionary from run_parameter_sweep()
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sensitivity analysis
    """
    # Convert results to DataFrame for analysis
    data = []
    for result in results['results']:
        if result.get('error') or not result.get('summary_statistics'):
            continue
        
        row = result['parameters'].copy()
        row.update(result['summary_statistics'])
        data.append(row)
    
    if not data:
        print("No valid results for sensitivity analysis")
        return None
    
    df = pd.DataFrame(data)
    
    # Calculate sensitivity metrics
    print("\nParameter Sensitivity Analysis:")
    print("=" * 60)
    
    for output_var in ['mean_runoff', 'mean_et', 'mean_soilmoisture']:
        if output_var not in df.columns:
            continue
            
        print(f"\nSensitivity of {output_var}:")
        print("-" * 40)
        
        for param in ['cs', 'alpha', 'gamma', 'beta']:
            if param in df.columns:
                # Calculate range of output for each parameter value
                sensitivity = df.groupby(param)[output_var].agg(['mean', 'std', 'min', 'max'])
                print(f"\n{param}:")
                print(sensitivity.round(3))
    
    return df

def run_single_comparison(data_path, baseline_params=None):
    """
    Run a comparison between baseline and perturbed parameters.
    
    Parameters
    ----------
    data_path : str
        Path to input data
    baseline_params : dict, optional
        Baseline parameters. If None, uses default values.
        
    Returns
    -------
    dict
        Comparison results
    """
    if baseline_params is None:
        baseline_params = {
            'cs': 420,      # Mid-range value
            'alpha': 4,     # Mid-range value
            'gamma': 0.5,   # Mid-range value
            'beta': 0.6     # Mid-range value
        }
    
    # Load data
    raw_data = pd.read_csv(data_path)
    data = prepro(raw_data)
    precip = data['tp'].values
    rad = data['snr'].values
    
    # Run baseline
    print("Running baseline simulation...")
    baseline_model = SimpleWaterBalanceModel(
        exp_runoff=baseline_params['alpha'],
        exp_et=baseline_params['gamma'],
        beta=baseline_params['beta'],
        whc=baseline_params['cs']
    )
    baseline_results = baseline_model.run(precip=precip, rad=rad)
    
    # Create comparison plots for each parameter
    comparison_results = {
        'baseline': {
            'parameters': baseline_params,
            'results': baseline_results
        },
        'perturbations': {}
    }
    
    # Test each parameter individually
    perturbations = [
        ('cs', 210, 840),
        ('alpha', 2, 8),
        ('gamma', 0.2, 0.8),
        ('beta', 0.4, 0.8)
    ]
    
    for param_name, low_val, high_val in perturbations:
        print(f"\nTesting {param_name} perturbations...")
        
        for value in [low_val, high_val]:
            # Create perturbed parameters
            perturbed_params = baseline_params.copy()
            perturbed_params[param_name] = value
            
            # Run model
            model = SimpleWaterBalanceModel(
                exp_runoff=perturbed_params['alpha'],
                exp_et=perturbed_params['gamma'],
                beta=perturbed_params['beta'],
                whc=perturbed_params['cs']
            )
            
            results = model.run(precip=precip, rad=rad)
            
            # Store results
            key = f"{param_name}_{value}"
            comparison_results['perturbations'][key] = {
                'parameters': perturbed_params,
                'results': results
            }
            
            # Calculate difference from baseline
            diff_runoff = results['runoff'] - baseline_results['runoff']
            diff_et = results['evapotranspiration'] - baseline_results['evapotranspiration']
            
            print(f"  {key}: ΔRunoff={np.nanmean(diff_runoff):.3f}, ΔET={np.nanmean(diff_et):.3f}")
    
    return comparison_results

# Example usage
if __name__ == "__main__":
    # Example 1: Full parameter sweep (81 combinations)
    print("EXAMPLE 1: FULL PARAMETER SWEEP")
    print("=" * 60)
    
    # Replace with your actual data file path
    data_file = "your_data.csv"
    
    # Run the full parameter sweep
    # results = run_parameter_sweep(data_file)
    
    # Analyze sensitivity
    # sensitivity_df = analyze_sensitivity(results)
    
    # Example 2: Quick comparison (fewer runs)
    print("\n\nEXAMPLE 2: SINGLE COMPARISON")
    print("=" * 60)
    
    comparison = run_single_comparison(data_file)
    
    # You could add plotting functions here to visualize results
    # plot_comparison(comparison)
    
    print("\nScript complete! Check the 'parameter_sweep_results' directory for outputs.")
