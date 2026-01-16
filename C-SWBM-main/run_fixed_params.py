"""
Simple script to run the water balance model with fixed parameters on a dataset
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add C-SWBM-main to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'C-SWBM-main'))

from CSWBM import SimpleWaterBalanceModel


def calculate_metrics(obs, mod):
    """Calculate performance metrics"""
    mask = ~(np.isnan(obs) | np.isnan(mod))
    obs_clean = obs[mask]
    mod_clean = mod[mask]
    
    if len(obs_clean) < 10:
        return None
    
    return {
        'correlation': np.corrcoef(obs_clean, mod_clean)[0, 1],
        'rmse': np.sqrt(np.mean((obs_clean - mod_clean)**2)),
        'mae': np.mean(np.abs(obs_clean - mod_clean)),
        'nse': 1 - (np.sum((obs_clean - mod_clean)**2) / 
                   np.sum((obs_clean - np.mean(obs_clean))**2)),
        'bias': np.mean(mod_clean - obs_clean),
        'n_points': len(obs_clean)
    }


def run_model_fixed_params(data_file, whc=420.0, exp_runoff=2.0, exp_et=0.5, 
                          beta=0.8, delta=0.3, start_date=None, end_date=None,
                          output_prefix=None, create_plots=True):
    """
    Run model with fixed parameters on a dataset.
    
    Parameters
    ----------
    data_file : str
        Path to CSV data file
    whc : float
        Water holding capacity (mm)
    exp_runoff : float
        Runoff exponent (α in assignment)
    exp_et : float
        ET exponent (γ in assignment)
    beta : float
        Beta parameter (β in assignment)
    delta : float
        Fast runoff fraction (new parameter)
    start_date : str, optional
        Start date for analysis (YYYY-MM-DD)
    end_date : str, optional
        End date for analysis (YYYY-MM-DD)
    output_prefix : str, optional
        Prefix for output files
    create_plots : bool
        Whether to create plots
    
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with all results
    metrics : dict
        Performance metrics
    """
    print("="*70)
    print("SIMPLE WATER BALANCE MODEL - SINGLE RUN")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {data_file}")
    model = SimpleWaterBalanceModel(
        exp_runoff=exp_runoff,
        exp_et=exp_et,
        beta=beta,
        whc=whc,
        delta=delta
    )
    
    data = model.load_data(data_file)
    print(f"Loaded {len(data)} days of data")
    print(f"Full date range: {data['time'].min()} to {data['time'].max()}")
    
    # Filter by date if specified
    if start_date or end_date:
        mask = pd.Series([True] * len(data))
        if start_date:
            mask &= data['time'] >= pd.to_datetime(start_date)
        if end_date:
            mask &= data['time'] <= pd.to_datetime(end_date)
        data = data[mask].reset_index(drop=True)
        print(f"Filtered to: {data['time'].min()} to {data['time'].max()}")
        print(f"Analysis period: {len(data)} days")
    
    # Print parameters
    print(f"\nModel Parameters:")
    print(f"  WHC (cs):           {whc} mm")
    print(f"  exp_runoff (α):     {exp_runoff}")
    print(f"  exp_et (γ):         {exp_et}")
    print(f"  beta (β):           {beta}")
    print(f"  delta (fast frac):  {delta}")
    print(f"  k_gw (fixed):       {model.k_gw}")
    
    # Run model
    print("\nRunning model...")
    results = model.run(data=data)
    print("✓ Model run complete")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'time': data['time'],
        'precipitation': data['tp'],
        'radiation': data['snr'],
        'modeled_sm': results['soilmoisture'],
        'modeled_ro': results['runoff'],
        'modeled_et': results['evapotranspiration'],
        'gw_storage': results['gw_storage'],
        'baseflow': results['baseflow'],
        'fast_runoff': results['fast_runoff']
    })
    
    # Add observations if available
    if 'sm' in data.columns:
        results_df['observed_sm'] = data['sm']
    if 'ro' in data.columns:
        results_df['observed_ro'] = data['ro']
    if 'le' in data.columns:
        results_df['observed_et'] = data['le']
    
    # Calculate metrics
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    metrics = {}
    
    for var_name, obs_col, mod_col in [
        ('Soil Moisture', 'observed_sm', 'modeled_sm'),
        ('Runoff', 'observed_ro', 'modeled_ro'),
        ('ET', 'observed_et', 'modeled_et')
    ]:
        if obs_col in results_df.columns:
            var_metrics = calculate_metrics(
                results_df[obs_col].values,
                results_df[mod_col].values
            )
            
            if var_metrics:
                metrics[var_name.lower().replace(' ', '_')] = var_metrics
                print(f"\n{var_name}:")
                print(f"  Correlation: {var_metrics['correlation']:.4f}")
                print(f"  RMSE:        {var_metrics['rmse']:.4f} mm")
                print(f"  MAE:         {var_metrics['mae']:.4f} mm")
                print(f"  NSE:         {var_metrics['nse']:.4f}")
                print(f"  Bias:        {var_metrics['bias']:.4f} mm")
                print(f"  N points:    {var_metrics['n_points']}")
    
    # Calculate sum of correlations
    if metrics:
        sum_corr = sum(m['correlation'] for m in metrics.values())
        print(f"\nSum of Correlations: {sum_corr:.4f}")
        metrics['sum_correlation'] = sum_corr
    
    # Save results
    if output_prefix:
        csv_file = f"{output_prefix}_results.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"\n✓ Results saved to: {csv_file}")
        
        # Save metrics
        metrics_file = f"{output_prefix}_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MODEL PARAMETERS\n")
            f.write("="*70 + "\n")
            f.write(f"WHC:        {whc}\n")
            f.write(f"exp_runoff: {exp_runoff}\n")
            f.write(f"exp_et:     {exp_et}\n")
            f.write(f"beta:       {beta}\n")
            f.write(f"delta:      {delta}\n")
            f.write(f"k_gw:       {model.k_gw}\n")
            f.write("\n" + "="*70 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("="*70 + "\n")
            for var_name, var_metrics in metrics.items():
                if var_name != 'sum_correlation':
                    f.write(f"\n{var_name.replace('_', ' ').title()}:\n")
                    for metric, value in var_metrics.items():
                        f.write(f"  {metric}: {value}\n")
            if 'sum_correlation' in metrics:
                f.write(f"\nSum of Correlations: {metrics['sum_correlation']}\n")
        print(f"✓ Metrics saved to: {metrics_file}")
    
    # Create plots
    if create_plots and output_prefix:
        print("\nGenerating plots...")
        create_model_plots(results_df, output_prefix)
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    
    return results_df, metrics


def create_model_plots(results_df, output_prefix):
    """Create visualization plots"""
    import seaborn as sns
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
    fig.suptitle('Model Results', fontsize=16, fontweight='bold')
    
    time = pd.to_datetime(results_df['time'])
    
    # Plot 1: Precipitation (inverted)
    ax = axes[0]
    ax.bar(time, results_df['precipitation'], color='blue', alpha=0.6, width=1.0)
    ax.set_ylabel('Precipitation (mm/day)', fontsize=11)
    ax.set_title('Precipitation', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Soil Moisture
    ax = axes[1]
    if 'observed_sm' in results_df.columns:
        ax.plot(time, results_df['observed_sm'], label='Observed', 
                color='green', linewidth=1.5, alpha=0.7)
    ax.plot(time, results_df['modeled_sm'], label='Modeled', 
            color='darkgreen', linewidth=1.5, linestyle='--')
    ax.set_ylabel('Soil Moisture (mm)', fontsize=11)
    ax.set_title('Soil Moisture', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Runoff with components
    ax = axes[2]
    if 'observed_ro' in results_df.columns:
        ax.plot(time, results_df['observed_ro'], label='Observed Total', 
                color='blue', linewidth=2, alpha=0.7)
    ax.plot(time, results_df['modeled_ro'], label='Modeled Total', 
            color='darkblue', linewidth=2, linestyle='--')
    ax.plot(time, results_df['fast_runoff'], label='Fast Runoff', 
            color='lightblue', linewidth=1, alpha=0.6)
    ax.plot(time, results_df['baseflow'], label='Baseflow', 
            color='navy', linewidth=1, alpha=0.6, linestyle=':')
    ax.set_ylabel('Runoff (mm/day)', fontsize=11)
    ax.set_title('Runoff Components', fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: ET
    ax = axes[3]
    if 'observed_et' in results_df.columns:
        ax.plot(time, results_df['observed_et'], label='Observed', 
                color='purple', linewidth=1.5, alpha=0.7)
    ax.plot(time, results_df['modeled_et'], label='Modeled', 
            color='indigo', linewidth=1.5, linestyle='--')
    ax.set_ylabel('ET (mm/day)', fontsize=11)
    ax.set_title('Evapotranspiration', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Groundwater Storage
    ax = axes[4]
    ax.plot(time, results_df['gw_storage'], label='Groundwater Storage', 
            color='brown', linewidth=1.5)
    ax.set_ylabel('Storage (mm)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_title('Groundwater Storage', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = f"{output_prefix}_timeseries.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved timeseries plot: {plot_file}")
    
    # Scatter plots if observations available
    if 'observed_sm' in results_df.columns or 'observed_ro' in results_df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Observed vs Modeled', fontsize=16, fontweight='bold')
        
        variables = [
            ('observed_sm', 'modeled_sm', 'Soil Moisture (mm)', 'green'),
            ('observed_ro', 'modeled_ro', 'Runoff (mm/day)', 'blue'),
            ('observed_et', 'modeled_et', 'ET (mm/day)', 'purple')
        ]
        
        for idx, (obs_col, mod_col, label, color) in enumerate(variables):
            ax = axes[idx]
            
            if obs_col in results_df.columns:
                obs = results_df[obs_col].values
                mod = results_df[mod_col].values
                
                mask = ~(np.isnan(obs) | np.isnan(mod))
                obs_clean = obs[mask]
                mod_clean = mod[mask]
                
                ax.scatter(obs_clean, mod_clean, alpha=0.5, s=10, color=color)
                
                min_val = min(obs_clean.min(), mod_clean.min())
                max_val = max(obs_clean.max(), mod_clean.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                
                corr = np.corrcoef(obs_clean, mod_clean)[0, 1]
                rmse = np.sqrt(np.mean((obs_clean - mod_clean)**2))
                
                stats_text = f'R = {corr:.3f}\nRMSE = {rmse:.2f}'
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                ax.set_xlabel(f'Observed {label}', fontsize=10)
                ax.set_ylabel(f'Modeled {label}', fontsize=10)
                ax.set_title(label.split('(')[0].strip(), fontsize=11)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        scatter_file = f"{output_prefix}_scatter.png"
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved scatter plots: {scatter_file}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run Simple Water Balance Model with fixed parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required argument
    parser.add_argument('data_file', type=str, 
                       help='Path to input CSV data file')
    
    # Model parameters
    parser.add_argument('--whc', type=float, default=420.0,
                       help='Water holding capacity (mm)')
    parser.add_argument('--exp_runoff', type=float, default=2.0,
                       help='Runoff exponent (α)')
    parser.add_argument('--exp_et', type=float, default=0.5,
                       help='ET exponent (γ)')
    parser.add_argument('--beta', type=float, default=0.8,
                       help='Beta parameter (β)')
    parser.add_argument('--delta', type=float, default=0.3,
                       help='Fast runoff fraction (0-1)')
    
    # Date filtering
    parser.add_argument('--start_date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    
    # Output options
    parser.add_argument('--output', type=str, default='model_output',
                       help='Output file prefix')
    parser.add_argument('--no_plots', action='store_true',
                       help='Do not create plots')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.data_file):
        print(f"ERROR: File not found: {args.data_file}")
        return
    
    # Run model
    results_df, metrics = run_model_fixed_params(
        data_file=args.data_file,
        whc=args.whc,
        exp_runoff=args.exp_runoff,
        exp_et=args.exp_et,
        beta=args.beta,
        delta=args.delta,
        start_date=args.start_date,
        end_date=args.end_date,
        output_prefix=args.output,
        create_plots=not args.no_plots
    )


if __name__ == "__main__":
    main()
