"""
Complete analysis script for Simple Water Balance Model
Runs model on multiple files with visualization and sensitivity analysis
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CSWBM import SimpleWaterBalanceModel
from visualize_results import (plot_monthly_comparison,
                               plot_scatter_comparison, plot_timeseries,
                               print_statistics)


def run_complete_analysis(filepaths, output_dir='complete_outputs', 
                         exp_runoff=2.0, exp_et=0.5, beta=0.8, whc=150.0,
                         create_plots=True):
    """
    Complete analysis pipeline for multiple datasets.
    
    Parameters
    ----------
    filepaths : list of str or str
        Single filepath or list of filepaths
    output_dir : str
        Directory for outputs
    exp_runoff, exp_et, beta, whc : float
        Model parameters
    create_plots : bool
        Generate visualization plots
        
    Returns
    -------
    all_results : dict
        Results for each file
    summary : pd.DataFrame
        Summary metrics table
    """
    
    # Handle single file input
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    summary_list = []
    
    print("="*70)
    print("SIMPLE WATER BALANCE MODEL - COMPLETE ANALYSIS")
    print("="*70)
    print(f"\nModel Parameters:")
    print(f"  exp_runoff: {exp_runoff}")
    print(f"  exp_et:     {exp_et}")
    print(f"  beta:       {beta}")
    print(f"  WHC:        {whc} mm")
    print(f"\nProcessing {len(filepaths)} file(s)...")
    print("="*70 + "\n")
    
    for idx, filepath in enumerate(filepaths, 1):
        filename = os.path.basename(filepath)
        site_name = filename.replace('.csv', '').replace('Data_swbm_', '').replace('_new', '')
        
        print(f"[{idx}/{len(filepaths)}] Processing: {filename}")
        print("-"*70)
        
        try:
            # Initialize model
            model = SimpleWaterBalanceModel(
                exp_runoff=exp_runoff,
                exp_et=exp_et,
                beta=beta,
                whc=whc
            )
            
            # Load data
            print("  Loading data...")
            data = model.load_data(filepath)
            print(f"    ✓ Loaded {len(data)} days of data")
            print(f"    Date range: {data['time'].min()} to {data['time'].max()}")
            
            # Run model
            print("  Running model...")
            results = model.run()
            print(f"    ✓ Model completed")
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'time': data['time'],
                'precipitation': data['tp'],
                'radiation': data['snr'],
                'modeled_sm': results['soilmoisture'],
                'modeled_ro': results['runoff'],
                'modeled_et': results['evapotranspiration'],
            })
            
            # Add observations if available
            if 'sm' in data.columns:
                results_df['observed_sm'] = data['sm']
            if 'ro' in data.columns:
                results_df['observed_ro'] = data['ro']
            if 'le' in data.columns:
                results_df['observed_et'] = data['le']
            if 'lat' in data.columns:
                results_df['latitude'] = data['lat'].iloc[0]
            if 'long' in data.columns:
                results_df['longitude'] = data['long'].iloc[0]
            
            # Save detailed results
            csv_path = os.path.join(output_dir, f"{site_name}_results.csv")
            results_df.to_csv(csv_path, index=False)
            print(f"    ✓ Saved results to: {csv_path}")
            
            # Calculate metrics
            metrics = {}
            for var, obs_col, mod_col in [
                ('sm', 'observed_sm', 'modeled_sm'),
                ('ro', 'observed_ro', 'modeled_ro'),
                ('et', 'observed_et', 'modeled_et')
            ]:
                if obs_col in results_df.columns:
                    obs = results_df[obs_col].values
                    mod = results_df[mod_col].values
                    mask = ~(np.isnan(obs) | np.isnan(mod))
                    
                    if mask.sum() > 0:
                        obs_valid = obs[mask]
                        mod_valid = mod[mask]
                        
                        metrics[var] = {
                            'rmse': np.sqrt(np.mean((obs_valid - mod_valid)**2)),
                            'mae': np.mean(np.abs(obs_valid - mod_valid)),
                            'r': np.corrcoef(obs_valid, mod_valid)[0, 1],
                            'bias': np.mean(mod_valid - obs_valid),
                            'nse': 1 - (np.sum((obs_valid - mod_valid)**2) / 
                                       np.sum((obs_valid - np.mean(obs_valid))**2))
                        }
            
            # Print metrics
            if metrics:
                print("  Performance Metrics:")
                for var, var_metrics in metrics.items():
                    print(f"    {var.upper()}: NSE={var_metrics['nse']:.3f}, "
                          f"R={var_metrics['r']:.3f}, RMSE={var_metrics['rmse']:.3f}")
            
            # Create plots
            if create_plots:
                print("  Generating plots...")
                
                # Time series plot
                try:
                    fig1 = plot_timeseries(data, results)
                    plot_path1 = os.path.join(output_dir, f"{site_name}_timeseries.png")
                    plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
                    plt.close(fig1)
                    print(f"    ✓ Saved: {site_name}_timeseries.png")
                except Exception as e:
                    print(f"    ✗ Timeseries plot error: {e}")
                
                # Scatter comparison
                try:
                    fig2 = plot_scatter_comparison(data, results)
                    plot_path2 = os.path.join(output_dir, f"{site_name}_scatter.png")
                    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                    print(f"    ✓ Saved: {site_name}_scatter.png")
                except Exception as e:
                    print(f"    ✗ Scatter plot error: {e}")
                
                # Monthly comparison
                try:
                    fig3 = plot_monthly_comparison(data, results)
                    plot_path3 = os.path.join(output_dir, f"{site_name}_monthly.png")
                    plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
                    plt.close(fig3)
                    print(f"    ✓ Saved: {site_name}_monthly.png")
                except Exception as e:
                    print(f"    ✗ Monthly plot error: {e}")
            
            # Store results
            all_results[site_name] = {
                'data': data,
                'results': results,
                'results_df': results_df,
                'metrics': metrics
            }
            
            # Add to summary
            summary_row = {
                'site': site_name,
                'file': filename,
                'n_days': len(data),
                'start_date': str(data['time'].min()),
                'end_date': str(data['time'].max())
            }
            
            for var, var_metrics in metrics.items():
                for metric_name, value in var_metrics.items():
                    summary_row[f"{var}_{metric_name}"] = value
            
            summary_list.append(summary_row)
            
            print(f"  ✓ Completed: {site_name}\n")
            
        except Exception as e:
            print(f"  ✗ ERROR processing {filename}: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Create summary dataframe
    if summary_list:
        summary = pd.DataFrame(summary_list)
        summary_path = os.path.join(output_dir, 'summary_metrics.csv')
        summary.to_csv(summary_path, index=False)
        
        print("="*70)
        print("SUMMARY OF ALL SITES")
        print("="*70)
        print(summary.to_string(index=False))
        print(f"\n✓ Summary saved to: {summary_path}")
    else:
        summary = pd.DataFrame()
        print("No results generated.")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")
    
    return all_results, summary


def quick_comparison_plot(all_results, output_dir='complete_outputs'):
    """
    Create comparison plots across all sites.
    
    Parameters
    ----------
    all_results : dict
        Results from run_complete_analysis
    output_dir : str
        Output directory
    """
    if not all_results:
        print("No results to plot.")
        return
    
    print("Creating cross-site comparison plots...")
    
    # Get site names
    sites = list(all_results.keys())
    n_sites = len(sites)
    
    # Create comparison figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_sites))
    
    for idx, site in enumerate(sites):
        results_df = all_results[site]['results_df']
        metrics = all_results[site]['metrics']
        
        # Soil Moisture
        if 'observed_sm' in results_df.columns:
            nse_sm = metrics.get('sm', {}).get('nse', np.nan)
            axes[0].plot(results_df['time'], results_df['observed_sm'], 
                        alpha=0.3, color=colors[idx])
            axes[0].plot(results_df['time'], results_df['modeled_sm'], 
                        label=f"{site} (NSE={nse_sm:.2f})", 
                        color=colors[idx], linewidth=1.5)
        
        # Runoff
        if 'observed_ro' in results_df.columns:
            nse_ro = metrics.get('ro', {}).get('nse', np.nan)
            axes[1].plot(results_df['time'], results_df['observed_ro'], 
                        alpha=0.3, color=colors[idx])
            axes[1].plot(results_df['time'], results_df['modeled_ro'], 
                        label=f"{site} (NSE={nse_ro:.2f})", 
                        color=colors[idx], linewidth=1.5)
        
        # ET
        if 'observed_et' in results_df.columns:
            nse_et = metrics.get('et', {}).get('nse', np.nan)
            axes[2].plot(results_df['time'], results_df['observed_et'], 
                        alpha=0.3, color=colors[idx])
            axes[2].plot(results_df['time'], results_df['modeled_et'], 
                        label=f"{site} (NSE={nse_et:.2f})", 
                        color=colors[idx], linewidth=1.5)
    
    axes[0].set_ylabel('Soil Moisture (mm)')
    axes[0].set_title('Soil Moisture Comparison Across Sites')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Runoff (mm)')
    axes[1].set_title('Runoff Comparison Across Sites')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel('ET (mm)')
    axes[2].set_xlabel('Date')
    axes[2].set_title('Evapotranspiration Comparison Across Sites')
    axes[2].legend(loc='best', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'sites_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plot: {comparison_path}")


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    
    # Define your files
    files = [
        'Data/Data_swbm_Germany_new.csv',
        'Data/Data_swbm_Spain_new.csv', 
        'Data/Data_swbm_Sweden_new.csv'
    ]
    
    # Or use a single file for testing
    # files = 'test_data.csv'
    
    # Run complete analysis
    all_results, summary = run_complete_analysis(
        filepaths=files,
        output_dir='complete_outputs',
        exp_runoff=2.0,
        exp_et=0.5,
        beta=0.8,
        whc=150.0,
        create_plots=True
    )
    
    # Create cross-site comparison
    if len(all_results) > 1:
        quick_comparison_plot(all_results, output_dir='complete_outputs')
    
    print("\n✓ All analyses complete!")
    print(f"  Results saved in: complete_outputs/")
    print(f"  Files generated:")
    print(f"    - [site]_results.csv (detailed results)")
    print(f"    - [site]_timeseries.png")
    print(f"    - [site]_scatter.png")
    print(f"    - [site]_monthly.png")
    print(f"    - summary_metrics.csv")
    if len(all_results) > 1:
        print(f"    - sites_comparison.png")
