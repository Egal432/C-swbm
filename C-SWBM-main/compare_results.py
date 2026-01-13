"""
Visualization and comparison script for calibration results
Creates plots showing parameter sensitivity and model performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def plot_parameter_sensitivity(all_calibrations, site_name, output_dir='calibration_results'):
    """
    Create plots showing how each parameter affects performance.
    
    Parameters
    ----------
    all_calibrations : pd.DataFrame
        Results from all parameter combinations
    site_name : str
        Name of the site
    output_dir : str
        Output directory
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f'Parameter Sensitivity Analysis - {site_name}', 
                 fontsize=16, fontweight='bold')
    
    params = ['whc', 'exp_runoff', 'exp_et', 'beta', 'alpha']
    variables = ['sm_corr', 'ro_corr', 'et_corr']
    var_names = ['Soil Moisture', 'Runoff', 'ET']
    
    for i, var in enumerate(variables):
        for j, param in enumerate(params):
            ax = axes[i, j]
            
            # Group by parameter value and calculate mean correlation
            grouped = all_calibrations.groupby(param)[var].agg(['mean', 'std', 'count'])
            
            # Plot
            x = grouped.index.values
            y_mean = grouped['mean'].values
            y_std = grouped['std'].values
            
            ax.errorbar(x, y_mean, yerr=y_std, marker='o', markersize=8, 
                       capsize=5, linewidth=2, label='Mean ± Std')
            
            # Highlight best value
            best_idx = np.argmax(y_mean)
            ax.scatter([x[best_idx]], [y_mean[best_idx]], 
                      color='red', s=200, marker='*', 
                      zorder=5, label='Best')
            
            ax.set_xlabel(param, fontsize=10)
            ax.set_ylabel(f'{var_names[i]} Correlation', fontsize=10)
            ax.set_title(f'{var_names[i]} vs {param}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{site_name}_parameter_sensitivity.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_correlation_heatmap(all_calibrations, site_name, output_dir='calibration_results'):
    """
    Create heatmap showing correlation between parameters and performance.
    
    Parameters
    ----------
    all_calibrations : pd.DataFrame
        Results from all parameter combinations
    site_name : str
        Name of the site
    output_dir : str
        Output directory
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Parameter vs Performance Correlation - {site_name}', 
                 fontsize=16, fontweight='bold')
    
    params = ['whc', 'exp_runoff', 'exp_et', 'beta', 'alpha']
    variables = [('sm_corr', 'Soil Moisture'), 
                 ('ro_corr', 'Runoff'), 
                 ('et_corr', 'ET')]
    
    for idx, (var, var_name) in enumerate(variables):
        ax = axes[idx]
        
        # Create pivot table for heatmap
        # Show how combinations affect performance
        corr_data = all_calibrations[params + [var]].corr()
        
        sns.heatmap(corr_data.loc[params, [var]], 
                   annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0, vmin=-1, vmax=1, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        ax.set_title(f'{var_name} Correlation', fontsize=12)
        ax.set_ylabel('Parameters', fontsize=10)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{site_name}_parameter_correlation_heatmap.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_validation_timeseries(validation_df, site_name, output_dir='calibration_results'):
    """
    Plot validation period time series.
    
    Parameters
    ----------
    validation_df : pd.DataFrame
        Validation time series data
    site_name : str
        Name of the site
    output_dir : str
        Output directory
    """
    fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
    fig.suptitle(f'Validation Period (2014-2018) - {site_name}', 
                 fontsize=16, fontweight='bold')
    
    time = pd.to_datetime(validation_df['time'])
    
    # NEW: Plot 0: Precipitation (inverted, at top)
    ax = axes[0]
    if 'precipitation' in validation_df.columns:
        ax.bar(time, validation_df['precipitation'], 
               color='blue', alpha=0.6, width=1.0, label='Precipitation')
        ax.set_ylabel('Precipitation (mm/day)', fontsize=11)
        ax.set_title('Precipitation Input', fontsize=12)
        ax.invert_yaxis()  # Invert so precipitation "falls" from top
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # Soil Moisture
    ax = axes[1]
    ax.plot(time, validation_df['observed_sm'], 
            label='Observed', color='green', linewidth=1.5, alpha=0.7)
    ax.plot(time, validation_df['modeled_sm'], 
            label='Modeled', color='darkgreen', linewidth=1.5, linestyle='--')
    ax.set_ylabel('Soil Moisture (mm)', fontsize=11)
    ax.set_title('Soil Moisture', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Runoff (with components)
    ax = axes[2]
    ax.plot(time, validation_df['observed_ro'], 
            label='Observed Total', color='blue', linewidth=2, alpha=0.7)
    ax.plot(time, validation_df['modeled_ro'], 
            label='Modeled Total', color='darkblue', linewidth=2, linestyle='--')
    if 'fast_runoff' in validation_df.columns:
        ax.plot(time, validation_df['fast_runoff'], 
                label='Fast Runoff', color='lightblue', linewidth=1, alpha=0.6)
    if 'baseflow' in validation_df.columns:
        ax.plot(time, validation_df['baseflow'], 
                label='Baseflow', color='navy', linewidth=1, alpha=0.6, linestyle=':')
    ax.set_ylabel('Runoff (mm/day)', fontsize=11)
    ax.set_title('Runoff Components', fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # ET
    ax = axes[3]
    ax.plot(time, validation_df['observed_et'], 
            label='Observed', color='purple', linewidth=1.5, alpha=0.7)
    ax.plot(time, validation_df['modeled_et'], 
            label='Modeled', color='indigo', linewidth=1.5, linestyle='--')
    ax.set_ylabel('ET (mm/day)', fontsize=11)
    ax.set_title('Evapotranspiration', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Groundwater Storage
    ax = axes[4]
    if 'gw_storage' in validation_df.columns:
        ax.plot(time, validation_df['gw_storage'], 
                label='Groundwater Storage', color='brown', linewidth=1.5)
        ax.set_ylabel('Storage (mm)', fontsize=11)
        ax.set_title('Groundwater Storage', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    ax.set_xlabel('Date', fontsize=11)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{site_name}_validation_timeseries.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_scatter_validation(validation_df, site_name, output_dir='calibration_results'):
    """
    Create scatter plots for validation period.
    
    Parameters
    ----------
    validation_df : pd.DataFrame
        Validation time series data
    site_name : str
        Name of the site
    output_dir : str
        Output directory
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Observed vs Modeled (Validation 2014-2018) - {site_name}', 
                 fontsize=16, fontweight='bold')
    
    variables = [
        ('observed_sm', 'modeled_sm', 'Soil Moisture (mm)', 'green'),
        ('observed_ro', 'modeled_ro', 'Runoff (mm/day)', 'blue'),
        ('observed_et', 'modeled_et', 'ET (mm/day)', 'purple')
    ]
    
    for idx, (obs_col, mod_col, label, color) in enumerate(variables):
        ax = axes[idx]
        
        obs = validation_df[obs_col].values
        mod = validation_df[mod_col].values
        
        # Remove NaN
        mask = ~(np.isnan(obs) | np.isnan(mod))
        obs_clean = obs[mask]
        mod_clean = mod[mask]
        
        # Scatter plot
        ax.scatter(obs_clean, mod_clean, alpha=0.5, s=10, color=color)
        
        # 1:1 line
        min_val = min(obs_clean.min(), mod_clean.min())
        max_val = max(obs_clean.max(), mod_clean.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='1:1 line')
        
        # Calculate metrics
        corr = np.corrcoef(obs_clean, mod_clean)[0, 1]
        rmse = np.sqrt(np.mean((obs_clean - mod_clean)**2))
        nse = 1 - (np.sum((obs_clean - mod_clean)**2) / 
                   np.sum((obs_clean - np.mean(obs_clean))**2))
        bias = np.mean(mod_clean - obs_clean)
        
        # Add statistics
        stats_text = f'R = {corr:.3f}\nRMSE = {rmse:.2f}\nNSE = {nse:.3f}\nBias = {bias:.2f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel(f'Observed {label}', fontsize=10)
        ax.set_ylabel(f'Modeled {label}', fontsize=10)
        ax.set_title(label.split('(')[0].strip(), fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{site_name}_validation_scatter.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_top_combinations(all_calibrations, site_name, n_top=10, output_dir='calibration_results'):
    """
    Plot top N parameter combinations and their performance.
    
    Parameters
    ----------
    all_calibrations : pd.DataFrame
        All calibration results
    site_name : str
        Name of the site
    n_top : int
        Number of top combinations to show
    output_dir : str
        Output directory
    """
    # Get top N combinations
    top_results = all_calibrations.nlargest(n_top, 'sum_correlation')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Top {n_top} Parameter Combinations - {site_name}', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Parameter values
    ax = axes[0]
    params = ['whc', 'exp_runoff', 'exp_et', 'beta', 'alpha']
    x = np.arange(n_top)
    width = 0.15
    
    for i, param in enumerate(params):
        # Normalize to 0-1 for visualization
        values = top_results[param].values
        values_norm = (values - all_calibrations[param].min()) / \
                     (all_calibrations[param].max() - all_calibrations[param].min())
        ax.bar(x + i*width, values_norm, width, label=param)
    
    ax.set_xlabel('Rank', fontsize=11)
    ax.set_ylabel('Normalized Parameter Value (0-1)', fontsize=11)
    ax.set_title('Parameter Values (normalized)', fontsize=12)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f'#{i+1}' for i in range(n_top)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Correlations
    ax = axes[1]
    x = np.arange(n_top)
    width = 0.25
    
    sm_corr = top_results['sm_corr'].values
    ro_corr = top_results['ro_corr'].values
    et_corr = top_results['et_corr'].values
    
    ax.bar(x - width, sm_corr, width, label='Soil Moisture', color='green', alpha=0.7)
    ax.bar(x, ro_corr, width, label='Runoff', color='blue', alpha=0.7)
    ax.bar(x + width, et_corr, width, label='ET', color='purple', alpha=0.7)
    
    # Add sum as line
    ax2 = ax.twinx()
    sum_corr = top_results['sum_correlation'].values
    ax2.plot(x, sum_corr, 'ro-', linewidth=2, markersize=8, label='Sum')
    ax2.set_ylabel('Sum of Correlations', fontsize=11, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_xlabel('Rank', fontsize=11)
    ax.set_ylabel('Individual Correlations', fontsize=11)
    ax.set_title('Performance Metrics', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'#{i+1}' for i in range(n_top)])
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{site_name}_top_combinations.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_cross_site_comparison(performance_table, output_dir='calibration_results'):
    """
    Compare performance across all sites.
    
    Parameters
    ----------
    performance_table : pd.DataFrame
        Performance table with all sites
    output_dir : str
        Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-Site Comparison', fontsize=16, fontweight='bold')
    
    sites = performance_table['Site'].values
    x = np.arange(len(sites))
    width = 0.35
    
    # Plot 1: Calibration vs Validation Correlations
    ax = axes[0, 0]
    ax.bar(x - width/2, performance_table['calib_sum_corr'], width, 
          label='Calibration', alpha=0.7, color='steelblue')
    ax.bar(x + width/2, performance_table['valid_sum_corr'], width, 
          label='Validation', alpha=0.7, color='coral')
    ax.set_ylabel('Sum of Correlations', fontsize=11)
    ax.set_title('Overall Performance', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Individual Variable Correlations (Validation)
    ax = axes[0, 1]
    width = 0.25
    ax.bar(x - width, performance_table['valid_sm_corr'], width, 
          label='Soil Moisture', alpha=0.7, color='green')
    ax.bar(x, performance_table['valid_ro_corr'], width, 
          label='Runoff', alpha=0.7, color='blue')
    ax.bar(x + width, performance_table['valid_et_corr'], width, 
          label='ET', alpha=0.7, color='purple')
    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_title('Validation: Individual Variables', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Best Parameters Comparison
    ax = axes[1, 0]
    params_to_plot = ['whc', 'exp_runoff', 'exp_et', 'beta', 'alpha']
    
    # Normalize parameters for comparison
    param_data = {}
    for param in params_to_plot:
        values = performance_table[param].values
        # Normalize to 0-1
        param_data[param] = (values - values.min()) / (values.max() - values.min() + 1e-10)
    
    x_pos = np.arange(len(sites))
    for i, param in enumerate(params_to_plot):
        ax.plot(x_pos, param_data[param], 'o-', label=param, linewidth=2, markersize=8)
    
    ax.set_ylabel('Normalized Parameter Value', fontsize=11)
    ax.set_title('Best Parameters (normalized)', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sites)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: NSE Comparison
    ax = axes[1, 1]
    width = 0.25
    ax.bar(x - width, performance_table['valid_sm_nse'], width, 
          label='Soil Moisture', alpha=0.7, color='green')
    ax.bar(x, performance_table['valid_ro_nse'], width, 
          label='Runoff', alpha=0.7, color='blue')
    ax.bar(x + width, performance_table['valid_et_nse'], width, 
          label='ET', alpha=0.7, color='purple')
    ax.set_ylabel('NSE', fontsize=11)
    ax.set_title('Validation: Nash-Sutcliffe Efficiency', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'cross_site_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def generate_all_visualizations(output_dir='calibration_results'):
    """
    Generate all visualization plots from saved calibration results.
    
    Parameters
    ----------
    output_dir : str
        Directory containing calibration results
    """
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    # Load performance table
    perf_table_path = os.path.join(output_dir, 'MODEL_PERFORMANCE_TABLE.csv')
    if not os.path.exists(perf_table_path):
        print(f"ERROR: Performance table not found at {perf_table_path}")
        return
    
    performance_table = pd.read_csv(perf_table_path)
    print(f"\nLoaded performance table with {len(performance_table)} sites")
    
    # Process each site
    for idx, row in performance_table.iterrows():
        site_name = row['Site']
        print(f"\n{site_name}:")
        
        # Load all calibrations
        calib_path = os.path.join(output_dir, f'{site_name}_all_calibrations.csv')
        if os.path.exists(calib_path):
            all_calibrations = pd.read_csv(calib_path)
            plot_parameter_sensitivity(all_calibrations, site_name, output_dir)
            plot_correlation_heatmap(all_calibrations, site_name, output_dir)
            plot_top_combinations(all_calibrations, site_name, n_top=10, output_dir=output_dir)
        
        # Load validation timeseries
        valid_path = os.path.join(output_dir, f'{site_name}_validation_timeseries.csv')
        if os.path.exists(valid_path):
            validation_df = pd.read_csv(valid_path)
            plot_validation_timeseries(validation_df, site_name, output_dir)
            plot_scatter_validation(validation_df, site_name, output_dir)
    
    # Cross-site comparison
    print(f"\nCross-site comparison:")
    plot_cross_site_comparison(performance_table, output_dir)
    
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll plots saved to: {output_dir}/")


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate visualizations from calibration results'
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='calibration_results',
        help='Directory containing calibration results (default: calibration_results)'
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Directory not found: {args.input_dir}")
        print("Please run the calibration script first.")
        exit(1)
    
    # Generate all visualizations
    generate_all_visualizations(args.input_dir)
    
    print("\n✓ Done! All visualizations created.")
