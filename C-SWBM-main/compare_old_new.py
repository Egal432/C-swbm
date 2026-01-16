"""
Compare old model (without groundwater) vs new model (with groundwater)
Shows the improvement gained by adding the groundwater storage component
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from CSWBM import SimpleWaterBalanceModel
import os
from itertools import product

sns.set_style("whitegrid")


class OldSimpleWaterBalanceModel:
    """
    Original model WITHOUT groundwater storage (for comparison).
    This is the model before adding the delta parameter.
    """
    
    def __init__(self, exp_runoff, exp_et, beta, whc):
        self.exp_runoff = exp_runoff
        self.exp_et = exp_et
        self.beta = beta
        self.whc = whc
        self.k_gw = 0.05  # Not used, but kept for compatibility
        
        self.soilm = None
        self.runoff = None
        self.et = None
        self.length = None
        self.data = None
        
    def load_data(self, filepath):
        """Load and preprocess ERA5 data from CSV file."""
        from CSWBM import prepro
        raw_data = pd.read_csv(filepath)
        self.data = prepro(raw_data)
        return self.data
    
    def spinup(self, precip, rad, n_years=5):
        """Spin up the model to equilibrium."""
        n_days = min(n_years * 365, len(precip))
        length = len(precip)
        
        soilm = np.full(length, np.nan)
        et = np.full(length, np.nan)
        et_corr = np.full(length, np.nan)
        infiltration = np.full(length, np.nan)
        infiltration_corr = np.full(length, np.nan)
        
        soilm[0] = 0.9 * self.whc
        
        for i in range(1, n_days):
            et[i-1] = rad[i-1] * self.beta * min(1.0, (soilm[i-1] / self.whc) ** self.exp_et)
            et_corr[i-1] = (rad[i-1] * self.beta * 
                           min(max(0.0, self.whc - soilm[i-1]), self.exp_et / self.whc) * 
                           (soilm[i-1] / self.whc) ** (self.exp_et - 1.0))
            infiltration[i-1] = (1.0 - min(1.0, (soilm[i-1] / self.whc) ** self.exp_runoff)) * precip[i-1]
            infiltration_corr[i-1] = ((-1.0) * 
                                     min(max(0.0, self.whc - soilm[i-1]), self.exp_runoff / self.whc) * 
                                     (soilm[i-1] / self.whc) ** (self.exp_runoff - 1.0) * 
                                     precip[i-1])
            et[i-1] = min(et[i-1], soilm[i-1] - 5.0)
            soilm[i] = soilm[i-1] + ((infiltration[i-1] - et[i-1]) / 
                                     (1.0 + et_corr[i-1] - infiltration_corr[i-1]))
        
        dec31_indices = []
        for year in range(n_years):
            idx = (year + 1) * 365 - 1
            if idx < n_days and not np.isnan(soilm[idx]):
                dec31_indices.append(idx)
        
        if dec31_indices:
            return np.mean([soilm[idx] for idx in dec31_indices])
        else:
            return 0.9 * self.whc
    
    def run(self, data=None, precip=None, rad=None):
        """Run the OLD water balance model (without groundwater)."""
        if data is not None:
            self.data = data
            precip = data['tp'].values.copy()
            rad = data['snr'].values.copy()
        
        if precip is None or rad is None:
            raise ValueError("Must provide either data or (precip, rad)")
        
        precip = precip.copy()
        rad = rad.copy()
        
        self.length = len(precip)
        
        # Initialize arrays
        self.soilm = np.full(self.length, np.nan)
        self.runoff = np.full(self.length, np.nan)
        self.et = np.full(self.length, np.nan)
        et_corr = np.full(self.length, np.nan)
        infiltration = np.full(self.length, np.nan)
        infiltration_corr = np.full(self.length, np.nan)
        
        # Spin up
        self.soilm[0] = self.spinup(precip, rad)
        
        # Main model run (OLD VERSION - direct runoff, no groundwater)
        for i in range(1, self.length):
            self.et[i-1] = rad[i-1] * self.beta * min(1.0, (self.soilm[i-1] / self.whc) ** self.exp_et)
            et_corr[i-1] = (rad[i-1] * self.beta * 
                           min(max(0.0, self.whc - self.soilm[i-1]), self.exp_et / self.whc) * 
                           (self.soilm[i-1] / self.whc) ** (self.exp_et - 1.0))
            infiltration[i-1] = (1.0 - min(1.0, (self.soilm[i-1] / self.whc) ** self.exp_runoff)) * precip[i-1]
            infiltration_corr[i-1] = ((-1.0) * 
                                     min(max(0.0, self.whc - self.soilm[i-1]), self.exp_runoff / self.whc) * 
                                     (self.soilm[i-1] / self.whc) ** (self.exp_runoff - 1.0) * 
                                     precip[i-1])
            self.et[i-1] = min(self.et[i-1], self.soilm[i-1] - 5.0)
            self.soilm[i] = self.soilm[i-1] + ((infiltration[i-1] - self.et[i-1]) / 
                                               (1.0 + et_corr[i-1] - infiltration_corr[i-1]))
            
            # OLD: All excess becomes immediate runoff
            self.runoff[i-1] = (min(1.0, (self.soilm[i-1] / self.whc) ** self.exp_runoff)) * precip[i-1]
            
            self.et[i-1] = self.et[i-1] + (self.soilm[i] - self.soilm[i-1]) * et_corr[i-1]
        
        return self.get_results()
    
    def get_results(self):
        """Return model results as a dictionary."""
        return {
            'soilmoisture': self.soilm,
            'runoff': self.runoff,
            'evapotranspiration': self.et,
            'exp_runoff': self.exp_runoff,
            'exp_et': self.exp_et,
            'beta': self.beta,
            'whc': self.whc,
            'length': self.length
        }


def calculate_metrics(obs, mod):
    """Calculate performance metrics."""
    mask = ~(np.isnan(obs) | np.isnan(mod))
    obs_clean = obs[mask]
    mod_clean = mod[mask]
    
    if len(obs_clean) < 10:
        return {'correlation': np.nan, 'rmse': np.nan, 'nse': np.nan, 'bias': np.nan}
    
    correlation = np.corrcoef(obs_clean, mod_clean)[0, 1]
    rmse = np.sqrt(np.mean((obs_clean - mod_clean)**2))
    nse = 1 - (np.sum((obs_clean - mod_clean)**2) / 
               np.sum((obs_clean - np.mean(obs_clean))**2))
    bias = np.mean(mod_clean - obs_clean)
    
    return {'correlation': correlation, 'rmse': rmse, 'nse': nse, 'bias': bias}


def compare_models_single_site(filepath, site_name, period='validation', 
                               output_dir='model_comparison'):
    """
    Compare old vs new model for a single site.
    
    Parameters
    ----------
    filepath : str
        Path to data file
    site_name : str
        Name of site
    period : str
        'calibration' (2008-2013) or 'validation' (2014-2018)
    output_dir : str
        Output directory
    
    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison results
    """
    print(f"\n{'='*70}")
    print(f"Comparing Models: {site_name} ({period.upper()})")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    model_temp = SimpleWaterBalanceModel(2.0, 0.5, 0.8, 150.0, 0.3)
    data_full = model_temp.load_data(filepath)
    
    # Filter by period
    if period == 'calibration':
        data = data_full[(data_full['time'] >= '2008-01-01') & 
                        (data_full['time'] <= '2013-12-31')].reset_index(drop=True)
    else:  # validation
        data = data_full[(data_full['time'] >= '2014-01-01') & 
                        (data_full['time'] <= '2018-12-31')].reset_index(drop=True)
    
    print(f"Period: {data['time'].min()} to {data['time'].max()}")
    print(f"Days: {len(data)}")
    
    # Define parameter grid (reduced for comparison - use typical values)
    param_grid = {
        'whc': [210.0, 420.0, 840.0],
        'exp_runoff': [2.0, 4.0, 8.0],
        'exp_et': [0.2, 0.5, 0.8],
        'beta': [0.4, 0.6, 0.8]
    }
    
    # For new model, test different delta values
    delta_values = [0.2, 0.4, 0.8]
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    base_combinations = list(product(*param_values))
    
    print(f"\nTesting {len(base_combinations)} base combinations × {len(delta_values)} delta values")
    print(f"Old model: {len(base_combinations)} runs (no delta)")
    print(f"New model: {len(base_combinations) * len(delta_values)} runs (with delta)")
    
    results_list = []
    
    # Run OLD model (without groundwater)
    print("\nRunning OLD model (no groundwater)...")
    for idx, combo in enumerate(base_combinations, 1):
        if idx % 20 == 0 or idx == 1:
            print(f"  OLD model progress: {idx}/{len(base_combinations)}")
        
        whc, exp_runoff, exp_et, beta = combo
        
        try:
            old_model = OldSimpleWaterBalanceModel(exp_runoff, exp_et, beta, whc)
            old_results = old_model.run(data=data)
            
            # Calculate metrics
            old_metrics = {}
            if 'sm' in data.columns:
                old_metrics['sm'] = calculate_metrics(data['sm'].values, old_results['soilmoisture'])
            if 'ro' in data.columns:
                old_metrics['ro'] = calculate_metrics(data['ro'].values, old_results['runoff'])
            if 'le' in data.columns:
                old_metrics['et'] = calculate_metrics(data['le'].values, old_results['evapotranspiration'])
            
            sum_corr = sum(m['correlation'] for m in old_metrics.values() if not np.isnan(m['correlation']))
            
            results_list.append({
                'model': 'OLD',
                'whc': whc,
                'exp_runoff': exp_runoff,
                'exp_et': exp_et,
                'beta': beta,
                'delta': np.nan,
                'sum_correlation': sum_corr,
                'sm_corr': old_metrics.get('sm', {}).get('correlation', np.nan),
                'ro_corr': old_metrics.get('ro', {}).get('correlation', np.nan),
                'et_corr': old_metrics.get('et', {}).get('correlation', np.nan),
                'ro_rmse': old_metrics.get('ro', {}).get('rmse', np.nan),
                'ro_nse': old_metrics.get('ro', {}).get('nse', np.nan)
            })
        except Exception as e:
            print(f"    Error: {e}")
    
    # Run NEW model (with groundwater)
    print("\nRunning NEW model (with groundwater)...")
    count = 0
    for combo in base_combinations:
        for delta in delta_values:
            count += 1
            if count % 50 == 0 or count == 1:
                print(f"  NEW model progress: {count}/{len(base_combinations) * len(delta_values)}")
            
            whc, exp_runoff, exp_et, beta = combo
            
            try:
                new_model = SimpleWaterBalanceModel(exp_runoff, exp_et, beta, whc, delta)
                new_results = new_model.run(data=data)
                
                # Calculate metrics
                new_metrics = {}
                if 'sm' in data.columns:
                    new_metrics['sm'] = calculate_metrics(data['sm'].values, new_results['soilmoisture'])
                if 'ro' in data.columns:
                    new_metrics['ro'] = calculate_metrics(data['ro'].values, new_results['runoff'])
                if 'le' in data.columns:
                    new_metrics['et'] = calculate_metrics(data['le'].values, new_results['evapotranspiration'])
                
                sum_corr = sum(m['correlation'] for m in new_metrics.values() if not np.isnan(m['correlation']))
                
                results_list.append({
                    'model': 'NEW',
                    'whc': whc,
                    'exp_runoff': exp_runoff,
                    'exp_et': exp_et,
                    'beta': beta,
                    'delta': delta,
                    'sum_correlation': sum_corr,
                    'sm_corr': new_metrics.get('sm', {}).get('correlation', np.nan),
                    'ro_corr': new_metrics.get('ro', {}).get('correlation', np.nan),
                    'et_corr': new_metrics.get('et', {}).get('correlation', np.nan),
                    'ro_rmse': new_metrics.get('ro', {}).get('rmse', np.nan),
                    'ro_nse': new_metrics.get('ro', {}).get('nse', np.nan)
                })
            except Exception as e:
                print(f"    Error: {e}")
    
    comparison_df = pd.DataFrame(results_list)
    
    # Save results
    csv_path = os.path.join(output_dir, f'{site_name}_{period}_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved comparison data to: {csv_path}")
    
    # Summary statistics
    old_best = comparison_df[comparison_df['model'] == 'OLD']['sum_correlation'].max()
    new_best = comparison_df[comparison_df['model'] == 'NEW']['sum_correlation'].max()
    
    print(f"\n{'='*70}")
    print(f"SUMMARY - {site_name} ({period})")
    print(f"{'='*70}")
    print(f"OLD model best sum correlation: {old_best:.4f}")
    print(f"NEW model best sum correlation: {new_best:.4f}")
    print(f"Improvement: {new_best - old_best:.4f} ({100*(new_best-old_best)/old_best:.1f}%)")
    
    # Runoff-specific improvement
    old_best_ro = comparison_df[comparison_df['model'] == 'OLD']['ro_corr'].max()
    new_best_ro = comparison_df[comparison_df['model'] == 'NEW']['ro_corr'].max()
    print(f"\nRunoff correlation:")
    print(f"  OLD: {old_best_ro:.4f}")
    print(f"  NEW: {new_best_ro:.4f}")
    print(f"  Improvement: {new_best_ro - old_best_ro:.4f}")
    
    return comparison_df


def plot_model_comparison(comparison_df, site_name, period, output_dir='model_comparison'):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'OLD vs NEW Model Comparison - {site_name} ({period.capitalize()})', 
                 fontsize=16, fontweight='bold')
    
    old_data = comparison_df[comparison_df['model'] == 'OLD']
    new_data = comparison_df[comparison_df['model'] == 'NEW']
    
    # Plot 1: Distribution of sum correlations
    ax = axes[0, 0]
    ax.hist(old_data['sum_correlation'].dropna(), bins=30, alpha=0.6, 
           label='OLD (no GW)', color='red', edgecolor='black')
    ax.hist(new_data['sum_correlation'].dropna(), bins=30, alpha=0.6, 
           label='NEW (with GW)', color='green', edgecolor='black')
    ax.axvline(old_data['sum_correlation'].max(), color='red', 
              linestyle='--', linewidth=2, label='OLD best')
    ax.axvline(new_data['sum_correlation'].max(), color='green', 
              linestyle='--', linewidth=2, label='NEW best')
    ax.set_xlabel('Sum of Correlations', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Overall Performance Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Runoff correlation comparison
    ax = axes[0, 1]
    ax.scatter(old_data['ro_corr'], old_data['sum_correlation'], 
              alpha=0.4, s=30, c='red', label='OLD')
    ax.scatter(new_data['ro_corr'], new_data['sum_correlation'], 
              alpha=0.4, s=30, c='green', label='NEW')
    ax.set_xlabel('Runoff Correlation', fontsize=11)
    ax.set_ylabel('Sum of Correlations', fontsize=11)
    ax.set_title('Runoff Performance', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Individual variable correlations
    ax = axes[1, 0]
    variables = ['sm_corr', 'ro_corr', 'et_corr']
    var_names = ['SM', 'RO', 'ET']
    x = np.arange(len(variables))
    width = 0.35
    
    old_means = [old_data[var].mean() for var in variables]
    new_means = [new_data[var].mean() for var in variables]
    old_max = [old_data[var].max() for var in variables]
    new_max = [new_data[var].max() for var in variables]
    
    ax.bar(x - width/2, old_means, width, alpha=0.7, color='red', 
          label='OLD (mean)', edgecolor='black')
    ax.bar(x + width/2, new_means, width, alpha=0.7, color='green', 
          label='NEW (mean)', edgecolor='black')
    ax.scatter(x - width/2, old_max, marker='*', s=200, c='darkred', 
              zorder=5, label='OLD (max)')
    ax.scatter(x + width/2, new_max, marker='*', s=200, c='darkgreen', 
              zorder=5, label='NEW (max)')
    
    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_title('Variable-specific Performance', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(var_names)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Runoff RMSE comparison
    ax = axes[1, 1]
    ax.hist(old_data['ro_rmse'].dropna(), bins=30, alpha=0.6, 
           label='OLD', color='red', edgecolor='black')
    ax.hist(new_data['ro_rmse'].dropna(), bins=30, alpha=0.6, 
           label='NEW', color='green', edgecolor='black')
    ax.axvline(old_data['ro_rmse'].min(), color='red', 
              linestyle='--', linewidth=2, label='OLD best')
    ax.axvline(new_data['ro_rmse'].min(), color='green', 
              linestyle='--', linewidth=2, label='NEW best')
    ax.set_xlabel('Runoff RMSE (mm/day)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Runoff Error Distribution (lower is better)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{site_name}_{period}_comparison_plot.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison plot: {filename}")


def compare_all_sites(file_dict, output_dir='model_comparison'):
    """
    Compare old vs new model for all sites.
    
    Parameters
    ----------
    file_dict : dict
        Dictionary with site names and file paths
    output_dir : str
        Output directory
    """
    print(f"\n{'#'*70}")
    print("# COMPARING OLD vs NEW MODEL FOR ALL SITES")
    print(f"{'#'*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    summary_results = []
    
    for site_name, filepath in file_dict.items():
        # Run comparison for validation period
        comparison_df = compare_models_single_site(
            filepath, site_name, period='validation', output_dir=output_dir
        )
        
        # Create plots
        plot_model_comparison(comparison_df, site_name, 'validation', output_dir)
        
        # Extract best results
        old_best = comparison_df[comparison_df['model'] == 'OLD'].nlargest(1, 'sum_correlation').iloc[0]
        new_best = comparison_df[comparison_df['model'] == 'NEW'].nlargest(1, 'sum_correlation').iloc[0]
        
        summary_results.append({
            'site': site_name,
            'old_sum_corr': old_best['sum_correlation'],
            'new_sum_corr': new_best['sum_correlation'],
            'improvement': new_best['sum_correlation'] - old_best['sum_correlation'],
            'improvement_pct': 100 * (new_best['sum_correlation'] - old_best['sum_correlation']) / old_best['sum_correlation'],
            'old_ro_corr': old_best['ro_corr'],
            'new_ro_corr': new_best['ro_corr'],
            'ro_improvement': new_best['ro_corr'] - old_best['ro_corr'],
            'best_delta': new_best['delta']
        })
    
    # Create summary table
    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(output_dir, 'OLD_vs_NEW_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*70}")
    print("OVERALL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))
    print(f"\n✓ Saved summary to: {summary_path}")
    
    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('OLD vs NEW Model: Summary Across All Sites', 
                 fontsize=16, fontweight='bold')
    
    sites = summary_df['site'].values
    x = np.arange(len(sites))
    width = 0.35
    
    # Plot 1: Sum of correlations
    ax = axes[0]
    ax.bar(x - width/2, summary_df['old_sum_corr'], width, 
          label='OLD (no GW)', alpha=0.7, color='red', edgecolor='black')
    ax.bar(x + width/2, summary_df['new_sum_corr'], width, 
          label='NEW (with GW)', alpha=0.7, color='green', edgecolor='black')
    ax.set_ylabel('Sum of Correlations', fontsize=11)
    ax.set_title('Overall Performance', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add improvement percentages as text
    for i, (old_val, new_val, imp_pct) in enumerate(zip(
        summary_df['old_sum_corr'], summary_df['new_sum_corr'], summary_df['improvement_pct']
    )):
        ax.text(i, max(old_val, new_val) + 0.05, f'+{imp_pct:.1f}%', 
               ha='center', fontsize=9, fontweight='bold', color='green')
    
    # Plot 2: Runoff correlation
    ax = axes[1]
    ax.bar(x - width/2, summary_df['old_ro_corr'], width, 
          label='OLD (no GW)', alpha=0.7, color='red', edgecolor='black')
    ax.bar(x + width/2, summary_df['new_ro_corr'], width, 
          label='NEW (with GW)', alpha=0.7, color='green', edgecolor='black')
    ax.set_ylabel('Runoff Correlation', fontsize=11)
    ax.set_title('Runoff Performance (Main Improvement)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add improvement as text
    for i, (old_val, new_val, imp) in enumerate(zip(
        summary_df['old_ro_corr'], summary_df['new_ro_corr'], summary_df['ro_improvement']
    )):
        ax.text(i, max(old_val, new_val) + 0.02, f'+{imp:.3f}', 
               ha='center', fontsize=9, fontweight='bold', color='green')
    
    plt.tight_layout()
    summary_plot = os.path.join(output_dir, 'OLD_vs_NEW_summary_plot.png')
    plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved summary plot: {summary_plot}")
    
    return summary_df


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    
    files = {
        'Germany': 'Data/Data_swbm_Germany_new.csv',
        'Spain': 'Data/Data_swbm_Spain_new.csv',
        'Sweden': 'Data/Data_swbm_Sweden_new.csv'
    }
    
    print("="*70)
    print("OLD vs NEW MODEL COMPARISON")
    print("="*70)
    print("\nThis script will:")
    print("  1. Run OLD model (without groundwater) on validation period")
    print("  2. Run NEW model (with groundwater) on validation period")
    print("  3. Compare performance metrics")
    print("  4. Generate comparison plots")
    print("\nEstimated time: ~10-20 minutes")
    
    input("\nPress Enter to continue...")
    
    summary = compare_all_sites(files, output_dir='model_comparison')
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print("\nCheck 'model_comparison/' directory for:")
    print("  - Comparison CSV files for each site")
    print("  - Comparison plots for each site")
    print("  - OLD_vs_NEW_summary.csv (overall results)")
    print("  - OLD_vs_NEW_summary_plot.png (visual summary)")
