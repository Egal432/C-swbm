"""
Parameter sensitivity analysis for the Simple Water Balance Model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from CSWBM import SimpleWaterBalanceModel

sns.set_style("whitegrid")


def calculate_metrics(obs, mod):
    """Calculate performance metrics"""
    mask = ~(np.isnan(obs) | np.isnan(mod))
    obs = obs[mask]
    mod = mod[mask]
    
    if len(obs) == 0:
        return {'rmse': np.nan, 'corr': np.nan, 'nse': np.nan, 'bias': np.nan}
    
    rmse = np.sqrt(np.mean((obs - mod)**2))
    corr = np.corrcoef(obs, mod)[0, 1] if len(obs) > 1 else np.nan
    nse = 1 - (np.sum((obs - mod)**2) / np.sum((obs - np.mean(obs))**2))
    bias = np.mean(mod - obs)
    
    return {'rmse': rmse, 'corr': corr, 'nse': nse, 'bias': bias}


def sensitivity_single_parameter(data, param_name, param_values, base_params):
    """
    Test sensitivity to a single parameter
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    param_name : str
        Name of parameter to vary
    param_values : array
        Values to test
    base_params : dict
        Base parameter values
    
    Returns
    -------
    results_df : pd.DataFrame
        Results for each parameter value
    """
    results_list = []
    
    for value in param_values:
        # Update parameter
        params = base_params.copy()
        params[param_name] = value
        
        # Run model
        model = SimpleWaterBalanceModel(**params)
        results = model.run(data=data)
        
        # Calculate metrics for each variable
        metrics = {}
        if 'sm' in data.columns:
            sm_metrics = calculate_metrics(data['sm'].values, results['soilmoisture'])
            metrics['sm_rmse'] = sm_metrics['rmse']
            metrics['sm_corr'] = sm_metrics['corr']
            metrics['sm_nse'] = sm_metrics['nse']
        
        if 'ro' in data.columns:
            ro_metrics = calculate_metrics(data['ro'].values, results['runoff'])
            metrics['ro_rmse'] = ro_metrics['rmse']
            metrics['ro_corr'] = ro_metrics['corr']
            metrics['ro_nse'] = ro_metrics['nse']
        
        if 'le' in data.columns:
            et_metrics = calculate_metrics(data['le'].values, results['evapotranspiration'])
            metrics['et_rmse'] = et_metrics['rmse']
            metrics['et_corr'] = et_metrics['corr']
            metrics['et_nse'] = et_metrics['nse']
        
        metrics[param_name] = value
        results_list.append(metrics)
    
    return pd.DataFrame(results_list)


def plot_sensitivity(sensitivity_results, param_name):
    """
    Plot sensitivity analysis results
    
    Parameters
    ----------
    sensitivity_results : dict
        Dictionary with parameter names as keys and DataFrames as values
    param_name : str
        Parameter name to plot
    """
    df = sensitivity_results[param_name]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Sensitivity Analysis: {param_name}', fontsize=16, fontweight='bold')
    
    variables = ['sm', 'ro', 'et']
    metrics = ['rmse', 'corr', 'nse']
    var_names = ['Soil Moisture', 'Runoff', 'ET']
    metric_names = ['RMSE', 'Correlation', 'NSE']
    
    for i, (var, var_name) in enumerate(zip(variables, var_names)):
        for j, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[j, i]
            col_name = f'{var}_{metric}'
            
            if col_name in df.columns:
                ax.plot(df[param_name], df[col_name], 'o-', linewidth=2, markersize=8)
                ax.set_xlabel(param_name)
                ax.set_ylabel(metric_name)
                ax.set_title(f'{var_name} - {metric_name}')
                ax.grid(True, alpha=0.3)
                
                # Highlight best value for NSE and Corr (higher is better)
                # and RMSE (lower is better)
                if metric in ['nse', 'corr']:
                    best_idx = df[col_name].idxmax()
                    if not pd.isna(best_idx):
                        ax.axvline(x=df[param_name].iloc[best_idx], 
                                  color='red', linestyle='--', alpha=0.5)
                elif metric == 'rmse':
                    best_idx = df[col_name].idxmin()
                    if not pd.isna(best_idx):
                        ax.axvline(x=df[param_name].iloc[best_idx], 
                                  color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def run_full_sensitivity(data_file, param_ranges=None, base_params=None):
    """
    Run full sensitivity analysis
    
    Parameters
    ----------
    data_file : str
        Path to data file
    param_ranges : dict, optional
        Dictionary with parameter names and ranges to test
    base_params : dict, optional
        Base parameter values
    """
    # Default parameter ranges
    if param_ranges is None:
        param_ranges = {
            'exp_runoff': np.linspace(0.5, 5.0, 10),
            'exp_et': np.linspace(0.1, 2.0, 10),
            'beta': np.linspace(0.3, 1.2, 10),
            'whc': np.linspace(50, 300, 10)
        }
    
    # Default base parameters
    if base_params is None:
        base_params = {
            'exp_runoff': 2.0,
            'exp_et': 0.5,
            'beta': 0.8,
            'whc': 150.0,
            'use_snow': False
        }
    
    # Load data
    print("Loading data...")
    model = SimpleWaterBalanceModel(**base_params)
    data = model.load_data(data_file)
    print(f"Loaded {len(data)} days of data")
    
    # Run sensitivity analysis for each parameter
    sensitivity_results = {}
    
    for param_name, param_values in param_ranges.items():
        print(f"\nAnalyzing sensitivity to {param_name}...")
        print(f"  Testing {len(param_values)} values from {param_values.min():.2f} to {param_values.max():.2f}")
        
        results_df = sensitivity_single_parameter(data, param_name, param_values, base_params)
        sensitivity_results[param_name] = results_df
        
        # Print summary
        print(f"  Results summary:")
        if 'sm_nse' in results_df.columns:
            best_sm = results_df.loc[results_df['sm_nse'].idxmax()]
            print(f"    Best SM NSE: {best_sm['sm_nse']:.4f} at {param_name}={best_sm[param_name]:.4f}")
        if 'ro_nse' in results_df.columns:
            best_ro = results_df.loc[results_df['ro_nse'].idxmax()]
            print(f"    Best RO NSE: {best_ro['ro_nse']:.4f} at {param_name}={best_ro[param_name]:.4f}")
        if 'et_nse' in results_df.columns:
            best_et = results_df.loc[results_df['et_nse'].idxmax()]
            print(f"    Best ET NSE: {best_et['et_nse']:.4f} at {param_name}={best_et[param_name]:.4f}")
    
    # Generate plots
    print("\nGenerating sensitivity plots...")
    for param_name in param_ranges.keys():
        fig = plot_sensitivity(sensitivity_results, param_name)
        filename = f'sensitivity_{param_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    # Save results to CSV
    print("\nSaving results to CSV...")
    for param_name, df in sensitivity_results.items():
        filename = f'sensitivity_{param_name}.csv'
        df.to_csv(filename, index=False)
        print(f"  Saved: {filename}")
    
    return sensitivity_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sensitivity analysis for water balance model')
    parser.add_argument('data_file', type=str, help='Path to input CSV data file')
    parser.add_argument('--no_show', action='store_true', help='Do not display plots')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SENSITIVITY ANALYSIS - SIMPLE WATER BALANCE MODEL")
    print("="*70)
    
    results = run_full_sensitivity(args.data_file)
    
    if not args.no_show:
        print("\nDisplaying plots... (close windows to exit)")
        plt.show()
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70 + "\n")