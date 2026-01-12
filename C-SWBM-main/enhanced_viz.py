"""
Enhanced visualization methods for Simple Water Balance Model
Adds zoomed-in views and better labeling with site names
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_single_year(data, results, year, site_name="Site"):
    """
    Plot detailed view of a single year with site name.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with 'time', 'tp', 'snr', 'ro', 'le', 'sm' columns
    results : dict
        Model results from SimpleWaterBalanceModel.get_results()
    year : int
        Year to plot (e.g., 2010)
    site_name : str
        Name of the site/country for plot titles
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Filter data for the specified year
    mask = data['time'].dt.year == year
    data_year = data[mask].reset_index(drop=True)
    
    if len(data_year) == 0:
        print(f"Warning: No data found for year {year}")
        return None
    
    # Filter results accordingly
    results_year = {
        'soilmoisture': results['soilmoisture'][mask.values],
        'runoff': results['runoff'][mask.values],
        'evapotranspiration': results['evapotranspiration'][mask.values],
        'whc': results['whc']
    }
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Main title with site and year
    fig.suptitle(f'{site_name} - Water Balance Model Results ({year})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Precipitation and Radiation
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    # Precipitation as bars
    ax1.bar(data_year['time'], data_year['tp'], alpha=0.6, 
            color='steelblue', label='Precipitation', width=0.8)
    
    # Radiation as line
    ax1_twin.plot(data_year['time'], data_year['snr'], 
                  color='orange', linewidth=2, label='Net Radiation')
    
    ax1.set_ylabel('Precipitation (mm/day)', color='steelblue', fontsize=11, fontweight='bold')
    ax1_twin.set_ylabel('Net Radiation (mm/day)', color='orange', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1_twin.tick_params(axis='y', labelcolor='orange')
    ax1.set_title('Meteorological Inputs', fontsize=12, fontweight='bold', pad=10)
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Soil Moisture
    ax2 = axes[1]
    
    # Modeled soil moisture
    ax2.plot(data_year['time'], results_year['soilmoisture'], 
             label='Modeled SM', color='darkgreen', linewidth=2.5, zorder=3)
    
    # Observed soil moisture (if available)
    if 'sm' in data_year.columns:
        ax2.plot(data_year['time'], data_year['sm'], 
                label='Observed SM', color='lightgreen', linewidth=2, 
                alpha=0.7, linestyle='--', zorder=2)
        
        # Calculate and show correlation
        mask_valid = ~(np.isnan(data_year['sm']) | np.isnan(results_year['soilmoisture']))
        if mask_valid.sum() > 0:
            corr = np.corrcoef(data_year['sm'][mask_valid], 
                             results_year['soilmoisture'][mask_valid])[0, 1]
            ax2.text(0.02, 0.98, f'R = {corr:.3f}', 
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # WHC line
    ax2.axhline(y=results_year['whc'], color='red', linestyle=':', 
                linewidth=2, label=f"WHC = {results_year['whc']:.1f} mm", zorder=1)
    
    ax2.set_ylabel('Soil Moisture (mm)', fontsize=11, fontweight='bold')
    ax2.set_title('Soil Moisture Dynamics', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Plot 3: Runoff
    ax3 = axes[2]
    
    # Modeled runoff
    ax3.fill_between(data_year['time'], 0, results_year['runoff'],
                     label='Modeled Runoff', color='darkblue', alpha=0.6)
    
    # Observed runoff (if available)
    if 'ro' in data_year.columns:
        ax3.plot(data_year['time'], data_year['ro'], 
                label='Observed Runoff', color='cyan', linewidth=2, 
                alpha=0.8, linestyle='--', marker='o', markersize=3)
        
        # Calculate total runoff
        total_obs = data_year['ro'].sum()
        total_mod = results_year['runoff'].sum()
        ax3.text(0.02, 0.98, f'Total Obs: {total_obs:.1f} mm\nTotal Mod: {total_mod:.1f} mm', 
                transform=ax3.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax3.set_ylabel('Runoff (mm/day)', fontsize=11, fontweight='bold')
    ax3.set_title('Runoff Generation', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    # Plot 4: Evapotranspiration
    ax4 = axes[3]
    
    # Modeled ET
    ax4.fill_between(data_year['time'], 0, results_year['evapotranspiration'],
                     label='Modeled ET', color='darkred', alpha=0.6)
    
    # Observed ET (if available)
    if 'le' in data_year.columns:
        ax4.plot(data_year['time'], data_year['le'], 
                label='Observed ET', color='salmon', linewidth=2, 
                alpha=0.8, linestyle='--', marker='o', markersize=3)
        
        # Calculate total ET
        total_obs = data_year['le'].sum()
        total_mod = results_year['evapotranspiration'].sum()
        ax4.text(0.02, 0.98, f'Total Obs: {total_obs:.1f} mm\nTotal Mod: {total_mod:.1f} mm', 
                transform=ax4.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
    
    ax4.set_ylabel('Evapotranspiration (mm/day)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax4.set_title('Evapotranspiration', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(loc='best', framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)
    
    # Format x-axis to show months nicely
    import matplotlib.dates as mdates
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    plt.tight_layout()
    return fig


def plot_date_range(data, results, start_date, end_date, site_name="Site"):
    """
    Plot detailed view of a custom date range.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    results : dict
        Model results
    start_date : str
        Start date ('YYYY-MM-DD')
    end_date : str
        End date ('YYYY-MM-DD')
    site_name : str
        Name of the site/country
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Convert dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Filter data
    mask = (data['time'] >= start) & (data['time'] <= end)
    data_range = data[mask].reset_index(drop=True)
    
    if len(data_range) == 0:
        print(f"Warning: No data found between {start_date} and {end_date}")
        return None
    
    # Filter results
    results_range = {
        'soilmoisture': results['soilmoisture'][mask.values],
        'runoff': results['runoff'][mask.values],
        'evapotranspiration': results['evapotranspiration'][mask.values],
        'whc': results['whc']
    }
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Main title
    fig.suptitle(f'{site_name} - Water Balance Model Results\n{start_date} to {end_date}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Use same plotting code as plot_single_year but with data_range
    # (Similar structure to above, but using data_range instead of data_year)
    
    # Plot 1: Precipitation and Radiation
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.bar(data_range['time'], data_range['tp'], alpha=0.6, 
            color='steelblue', label='Precipitation', width=0.8)
    ax1_twin.plot(data_range['time'], data_range['snr'], 
                  color='orange', linewidth=2, label='Net Radiation')
    ax1.set_ylabel('Precipitation (mm/day)', color='steelblue', fontweight='bold')
    ax1_twin.set_ylabel('Net Radiation (mm/day)', color='orange', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1_twin.tick_params(axis='y', labelcolor='orange')
    ax1.set_title('Meteorological Inputs', fontsize=12, fontweight='bold', pad=10)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Soil Moisture
    ax2 = axes[1]
    ax2.plot(data_range['time'], results_range['soilmoisture'], 
             label='Modeled SM', color='darkgreen', linewidth=2.5)
    if 'sm' in data_range.columns:
        ax2.plot(data_range['time'], data_range['sm'], 
                label='Observed SM', color='lightgreen', linewidth=2, alpha=0.7, linestyle='--')
    ax2.axhline(y=results_range['whc'], color='red', linestyle=':', 
                linewidth=2, label=f"WHC = {results_range['whc']:.1f} mm")
    ax2.set_ylabel('Soil Moisture (mm)', fontweight='bold')
    ax2.set_title('Soil Moisture Dynamics', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Runoff
    ax3 = axes[2]
    ax3.fill_between(data_range['time'], 0, results_range['runoff'],
                     label='Modeled Runoff', color='darkblue', alpha=0.6)
    if 'ro' in data_range.columns:
        ax3.plot(data_range['time'], data_range['ro'], 
                label='Observed Runoff', color='cyan', linewidth=2, alpha=0.8, linestyle='--')
    ax3.set_ylabel('Runoff (mm/day)', fontweight='bold')
    ax3.set_title('Runoff Generation', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Evapotranspiration
    ax4 = axes[3]
    ax4.fill_between(data_range['time'], 0, results_range['evapotranspiration'],
                     label='Modeled ET', color='darkred', alpha=0.6)
    if 'le' in data_range.columns:
        ax4.plot(data_range['time'], data_range['le'], 
                label='Observed ET', color='salmon', linewidth=2, alpha=0.8, linestyle='--')
    ax4.set_ylabel('Evapotranspiration (mm/day)', fontweight='bold')
    ax4.set_xlabel('Date', fontweight='bold')
    ax4.set_title('Evapotranspiration', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(loc='best', framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_seasonal_comparison(data, results, site_name="Site"):
    """
    Plot seasonal patterns (averaged by month) with site name.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    results : dict
        Model results
    site_name : str
        Name of the site/country
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Create dataframe with results
    df = data.copy()
    df['modeled_sm'] = results['soilmoisture']
    df['modeled_ro'] = results['runoff']
    df['modeled_et'] = results['evapotranspiration']
    df['month'] = df['time'].dt.month
    
    # Calculate monthly means
    monthly = df.groupby('month').agg({
        'tp': 'mean',
        'sm': 'mean',
        'modeled_sm': 'mean',
        'ro': 'mean',
        'modeled_ro': 'mean',
        'le': 'mean',
        'modeled_et': 'mean'
    })
    
    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{site_name} - Seasonal Patterns (Monthly Averages)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Precipitation
    ax1 = axes[0, 0]
    ax1.bar(range(1, 13), monthly['tp'], color='steelblue', alpha=0.7)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_names)
    ax1.set_ylabel('Mean Precipitation (mm/day)', fontweight='bold')
    ax1.set_title('Precipitation', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Soil Moisture
    ax2 = axes[0, 1]
    if 'sm' in df.columns:
        ax2.plot(range(1, 13), monthly['sm'], 'o-', label='Observed', 
                color='green', linewidth=2.5, markersize=8)
    ax2.plot(range(1, 13), monthly['modeled_sm'], 's--', label='Modeled', 
            color='darkgreen', linewidth=2.5, markersize=8, alpha=0.7)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_names)
    ax2.set_ylabel('Mean Soil Moisture (mm)', fontweight='bold')
    ax2.set_title('Soil Moisture', fontsize=12, fontweight='bold')
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Runoff
    ax3 = axes[1, 0]
    if 'ro' in df.columns:
        ax3.plot(range(1, 13), monthly['ro'], 'o-', label='Observed', 
                color='blue', linewidth=2.5, markersize=8)
    ax3.plot(range(1, 13), monthly['modeled_ro'], 's--', label='Modeled', 
            color='darkblue', linewidth=2.5, markersize=8, alpha=0.7)
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(month_names)
    ax3.set_ylabel('Mean Runoff (mm/day)', fontweight='bold')
    ax3.set_title('Runoff', fontsize=12, fontweight='bold')
    ax3.legend(framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Evapotranspiration
    ax4 = axes[1, 1]
    if 'le' in df.columns:
        ax4.plot(range(1, 13), monthly['le'], 'o-', label='Observed', 
                color='red', linewidth=2.5, markersize=8)
    ax4.plot(range(1, 13), monthly['modeled_et'], 's--', label='Modeled', 
            color='darkred', linewidth=2.5, markersize=8, alpha=0.7)
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(month_names)
    ax4.set_ylabel('Mean ET (mm/day)', fontweight='bold')
    ax4.set_title('Evapotranspiration', fontsize=12, fontweight='bold')
    ax4.legend(framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from CSWBM import SimpleWaterBalanceModel
    
    # Initialize and run model
    model = SimpleWaterBalanceModel(exp_runoff=2.0, exp_et=0.5, beta=0.8, whc=150.0)
    data = model.load_data('test_data.csv')
    results = model.run()
    
    # Extract site name from filename
    site_name = "Germany"  # or extract from filename
    
    # Plot single year - 2010
    fig1 = plot_single_year(data, results, year=2010, site_name=site_name)
    plt.savefig(f'{site_name}_2010_detailed.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {site_name}_2010_detailed.png")
    
    # Plot custom date range
    fig2 = plot_date_range(data, results, '2010-06-01', '2010-09-30', site_name=site_name)
    plt.savefig(f'{site_name}_summer2010.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {site_name}_summer2010.png")
    
    # Plot seasonal patterns
    fig3 = plot_seasonal_comparison(data, results, site_name=site_name)
    plt.savefig(f'{site_name}_seasonal.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {site_name}_seasonal.png")
    
    plt.show()
