"""
Visualization script for Simple Water Balance Model results
Compares model outputs with input data and observations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from CSWBM import SimpleWaterBalanceModel

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def plot_timeseries(data, results, start_date=None, end_date=None):
    """
    Plot time series of model inputs and outputs
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with 'time', 'tp', 'snr', 'ro', 'le', 'sm' columns
    results : dict
        Model results from SimpleWaterBalanceModel.get_results()
    start_date : str, optional
        Start date for plotting (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date for plotting (format: 'YYYY-MM-DD')
    """
    # Filter data by date range if provided
    if start_date or end_date:
        mask = pd.Series([True] * len(data))
        if start_date:
            mask &= data['time'] >= pd.to_datetime(start_date)
        if end_date:
            mask &= data['time'] <= pd.to_datetime(end_date)
        data = data[mask].reset_index(drop=True)
        
        # Filter results accordingly
        for key in ['soilmoisture', 'runoff', 'evapotranspiration']:
            if results[key] is not None:
                results[key] = results[key][mask.values]
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Plot 1: Precipitation and Radiation
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.bar(data['time'], data['tp'], alpha=0.6, color='blue', label='Precipitation')
    ax1_twin.plot(data['time'], data['snr'], color='orange', linewidth=1.5, label='Net Radiation')
    ax1.set_ylabel('Precipitation (mm)', color='blue')
    ax1_twin.set_ylabel('Net Radiation (mm)', color='orange')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='orange')
    ax1.set_title('Model Inputs: Precipitation and Net Radiation')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Soil Moisture
    ax2 = axes[1]
    if 'sm' in data.columns:
        ax2.plot(data['time'], data['sm'], label='Observed SM', color='green', linewidth=1.5, alpha=0.7)
    ax2.plot(data['time'], results['soilmoisture'], label='Modeled SM', 
             color='darkgreen', linewidth=1.5, linestyle='--')
    ax2.axhline(y=results['whc'], color='red', linestyle=':', label=f"WHC = {results['whc']:.1f} mm")
    ax2.set_ylabel('Soil Moisture (mm)')
    ax2.set_title('Soil Moisture: Observed vs Modeled')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Runoff
    ax3 = axes[2]
    if 'ro' in data.columns:
        ax3.plot(data['time'], data['ro'], label='Observed Runoff', 
                color='blue', linewidth=1.5, alpha=0.7)
    ax3.plot(data['time'], results['runoff'], label='Modeled Runoff', 
             color='darkblue', linewidth=1.5, linestyle='--')
    ax3.set_ylabel('Runoff (mm)')
    ax3.set_title('Runoff: Observed vs Modeled')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Evapotranspiration
    ax4 = axes[3]
    if 'le' in data.columns:
        ax4.plot(data['time'], data['le'], label='Observed ET', 
                color='purple', linewidth=1.5, alpha=0.7)
    ax4.plot(data['time'], results['evapotranspiration'], label='Modeled ET', 
             color='indigo', linewidth=1.5, linestyle='--')
    ax4.set_ylabel('Evapotranspiration (mm)')
    ax4.set_xlabel('Date')
    ax4.set_title('Evapotranspiration: Observed vs Modeled')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_scatter_comparison(data, results):
    """
    Create scatter plots comparing observed vs modeled values
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with observations
    results : dict
        Model results
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    variables = [
        ('sm', 'soilmoisture', 'Soil Moisture', 'green'),
        ('ro', 'runoff', 'Runoff', 'blue'),
        ('le', 'evapotranspiration', 'ET', 'purple')
    ]
    
    for ax, (obs_col, mod_key, title, color) in zip(axes, variables):
        if obs_col in data.columns:
            obs = data[obs_col].values
            mod = results[mod_key]
            
            # Remove NaN values
            mask = ~(np.isnan(obs) | np.isnan(mod))
            obs = obs[mask]
            mod = mod[mask]
            
            # Scatter plot
            ax.scatter(obs, mod, alpha=0.5, s=10, color=color)
            
            # 1:1 line
            min_val = min(obs.min(), mod.min())
            max_val = max(obs.max(), mod.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
            
            # Calculate statistics
            rmse = np.sqrt(np.mean((obs - mod)**2))
            mae = np.mean(np.abs(obs - mod))
            corr = np.corrcoef(obs, mod)[0, 1]
            bias = np.mean(mod - obs)
            
            # Add statistics to plot
            stats_text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR: {corr:.3f}\nBias: {bias:.2f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(f'Observed {title} (mm)')
            ax.set_ylabel(f'Modeled {title} (mm)')
            ax.set_title(f'{title}: Observed vs Modeled')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_monthly_comparison(data, results):
    """
    Create monthly aggregated comparison plots
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    results : dict
        Model results
    """
    # Create a dataframe with all data
    df = data.copy()
    df['modeled_sm'] = results['soilmoisture']
    df['modeled_ro'] = results['runoff']
    df['modeled_et'] = results['evapotranspiration']
    
    # Add month column
    df['month'] = df['time'].dt.to_period('M')
    
    # Aggregate by month
    monthly = df.groupby('month').agg({
        'tp': 'sum',
        'sm': 'mean',
        'modeled_sm': 'mean',
        'ro': 'sum',
        'modeled_ro': 'sum',
        'le': 'sum',
        'modeled_et': 'sum'
    }).reset_index()
    
    monthly['month_str'] = monthly['month'].astype(str)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Precipitation
    ax1 = axes[0, 0]
    ax1.bar(range(len(monthly)), monthly['tp'], color='blue', alpha=0.6)
    ax1.set_ylabel('Total Precipitation (mm)')
    ax1.set_title('Monthly Precipitation')
    ax1.set_xticks(range(0, len(monthly), max(1, len(monthly)//12)))
    ax1.set_xticklabels([monthly['month_str'].iloc[i] for i in range(0, len(monthly), max(1, len(monthly)//12))], 
                        rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Soil Moisture
    ax2 = axes[0, 1]
    if 'sm' in df.columns:
        ax2.plot(range(len(monthly)), monthly['sm'], 'o-', label='Observed', color='green', linewidth=2)
    ax2.plot(range(len(monthly)), monthly['modeled_sm'], 's-', label='Modeled', 
             color='darkgreen', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Mean Soil Moisture (mm)')
    ax2.set_title('Monthly Mean Soil Moisture')
    ax2.legend()
    ax2.set_xticks(range(0, len(monthly), max(1, len(monthly)//12)))
    ax2.set_xticklabels([monthly['month_str'].iloc[i] for i in range(0, len(monthly), max(1, len(monthly)//12))], 
                        rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Runoff
    ax3 = axes[1, 0]
    if 'ro' in df.columns:
        ax3.plot(range(len(monthly)), monthly['ro'], 'o-', label='Observed', color='blue', linewidth=2)
    ax3.plot(range(len(monthly)), monthly['modeled_ro'], 's-', label='Modeled', 
             color='darkblue', linewidth=2, alpha=0.7)
    ax3.set_ylabel('Total Runoff (mm)')
    ax3.set_title('Monthly Total Runoff')
    ax3.legend()
    ax3.set_xticks(range(0, len(monthly), max(1, len(monthly)//12)))
    ax3.set_xticklabels([monthly['month_str'].iloc[i] for i in range(0, len(monthly), max(1, len(monthly)//12))], 
                        rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Evapotranspiration
    ax4 = axes[1, 1]
    if 'le' in df.columns:
        ax4.plot(range(len(monthly)), monthly['le'], 'o-', label='Observed', color='purple', linewidth=2)
    ax4.plot(range(len(monthly)), monthly['modeled_et'], 's-', label='Modeled', 
             color='indigo', linewidth=2, alpha=0.7)
    ax4.set_ylabel('Total ET (mm)')
    ax4.set_title('Monthly Total Evapotranspiration')
    ax4.legend()
    ax4.set_xticks(range(0, len(monthly), max(1, len(monthly)//12)))
    ax4.set_xticklabels([monthly['month_str'].iloc[i] for i in range(0, len(monthly), max(1, len(monthly)//12))], 
                        rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_statistics(data, results):
    """
    Print statistical comparison between observed and modeled values
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    results : dict
        Model results
    """
    print("\n" + "="*60)
    print("MODEL PERFORMANCE STATISTICS")
    print("="*60)
    
    variables = [
        ('sm', 'soilmoisture', 'Soil Moisture'),
        ('ro', 'runoff', 'Runoff'),
        ('le', 'evapotranspiration', 'Evapotranspiration')
    ]
    
    for obs_col, mod_key, name in variables:
        if obs_col in data.columns:
            obs = data[obs_col].values
            mod = results[mod_key]
            
            # Remove NaN values
            mask = ~(np.isnan(obs) | np.isnan(mod))
            obs = obs[mask]
            mod = mod[mask]
            
            if len(obs) > 0:
                rmse = np.sqrt(np.mean((obs - mod)**2))
                mae = np.mean(np.abs(obs - mod))
                corr = np.corrcoef(obs, mod)[0, 1]
                bias = np.mean(mod - obs)
                nse = 1 - (np.sum((obs - mod)**2) / np.sum((obs - np.mean(obs))**2))
                
                print(f"\n{name}:")
                print(f"  RMSE:        {rmse:.4f} mm")
                print(f"  MAE:         {mae:.4f} mm")
                print(f"  Correlation: {corr:.4f}")
                print(f"  Bias:        {bias:.4f} mm")
                print(f"  NSE:         {nse:.4f}")
                print(f"  Obs mean:    {np.mean(obs):.4f} mm")
                print(f"  Mod mean:    {np.mean(mod):.4f} mm")
    
    print("\n" + "="*60)
    print("MODEL PARAMETERS:")
    print("="*60)
    print(f"  exp_runoff:  {results['exp_runoff']:.4f}")
    print(f"  exp_et:      {results['exp_et']:.4f}")
    print(f"  beta:        {results['beta']:.4f}")
    print(f"  WHC:         {results['whc']:.4f} mm")
    print(f"  melting:     {results['melting']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Loading data and running model...")
    
    # Initialize model with example parameters
    model = SimpleWaterBalanceModel(
        exp_runoff=2.0,
        exp_et=0.5,
        beta=0.8,
        whc=150.0,
    )
    
    # Load your data file
    data = model.load_data('your_data_file.csv')
    
    # Run model
    results = model.run()
    
    # Generate all plots
    print("\nGenerating visualizations...")
    
    # Time series plot
    fig1 = plot_timeseries(data, results)
    plt.savefig('timeseries_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: timeseries_comparison.png")
    
    # Scatter comparison
    fig2 = plot_scatter_comparison(data, results)
    plt.savefig('scatter_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: scatter_comparison.png")
    
    # Monthly comparison
    fig3 = plot_monthly_comparison(data, results)
    plt.savefig('monthly_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: monthly_comparison.png")
    
    # Print statistics
    print_statistics(data, results)
    
    # Show plots
    plt.show()