"""
EXAMPLE: How to use the new yearly visualization features

This shows different ways to create zoomed-in, detailed plots
with country/site names prominently displayed.
"""

from CSWBM import SimpleWaterBalanceModel
from enhanced_viz import plot_single_year, plot_date_range, plot_seasonal_comparison
import matplotlib.pyplot as plt

# =============================================================================
# METHOD 1: Create yearly plots manually (most control)
# =============================================================================

print("="*70)
print("METHOD 1: Manual yearly plots")
print("="*70)

# Run model
model = SimpleWaterBalanceModel(exp_runoff=2.0, exp_et=0.5, beta=0.8, whc=150.0)
data = model.load_data('Data/Data_swbm_Germany_new.csv')
results = model.run()

# Plot specific year with country name
fig = plot_single_year(data, results, year=2010, site_name="Germany")
plt.savefig('Germany_2010_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Germany_2010_detailed.png")

# Plot different year
fig = plot_single_year(data, results, year=2015, site_name="Germany")
plt.savefig('Germany_2015_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Germany_2015_detailed.png")

plt.close('all')


# =============================================================================
# METHOD 2: Custom date range (e.g., just summer)
# =============================================================================

print("\n" + "="*70)
print("METHOD 2: Custom date range")
print("="*70)

# Plot just summer 2010
fig = plot_date_range(data, results, 
                     start_date='2010-06-01', 
                     end_date='2010-08-31',
                     site_name="Germany")
plt.savefig('Germany_summer2010.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Germany_summer2010.png")

# Plot winter 2010-2011
fig = plot_date_range(data, results,
                     start_date='2010-12-01',
                     end_date='2011-02-28',
                     site_name="Germany")
plt.savefig('Germany_winter2010.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Germany_winter2010.png")

plt.close('all')


# =============================================================================
# METHOD 3: Seasonal patterns
# =============================================================================

print("\n" + "="*70)
print("METHOD 3: Seasonal patterns")
print("="*70)

fig = plot_seasonal_comparison(data, results, site_name="Germany")
plt.savefig('Germany_seasonal.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Germany_seasonal.png")

plt.close('all')


# =============================================================================
# METHOD 4: Use enhanced complete_analysis (automatic for multiple files)
# =============================================================================

print("\n" + "="*70)
print("METHOD 4: Automatic yearly plots via complete_analysis")
print("="*70)

from enhanced_complete_analysis import run_complete_analysis

files = [
    'Data/Data_swbm_Germany_new.csv',
    'Data/Data_swbm_Spain_new.csv',
]

# This will automatically:
# - Extract country names from filenames
# - Create standard plots with country names in titles
# - Create detailed yearly plots for first complete year
# - Create seasonal pattern plots
all_results, summary = run_complete_analysis(
    filepaths=files,
    output_dir='yearly_outputs',
    whc=150.0,
    create_plots=True,
    create_yearly_plots=True,  # Enable yearly detail plots
    yearly_focus=2010          # Focus on 2010 (or None for auto-select)
)

print("\n✓ Check 'yearly_outputs/' folder for all plots!")


# =============================================================================
# METHOD 5: Multiple years for one site
# =============================================================================

print("\n" + "="*70)
print("METHOD 5: Multiple years comparison")
print("="*70)

# Run model for Germany
model = SimpleWaterBalanceModel(exp_runoff=2.0, exp_et=0.5, beta=0.8, whc=150.0)
data = model.load_data('Data/Data_swbm_Germany_new.csv')
results = model.run()

# Plot multiple years
years_to_plot = [2008, 2010, 2012, 2015]

for year in years_to_plot:
    fig = plot_single_year(data, results, year=year, site_name="Germany")
    if fig is not None:
        plt.savefig(f'Germany_{year}_detailed.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: Germany_{year}_detailed.png")
        plt.close(fig)


# =============================================================================
# WHAT YOU GET
# =============================================================================

print("\n" + "="*70)
print("OUTPUT FILES CREATED")
print("="*70)
print("""
Yearly detail plots (4 subplots):
  - Precipitation + Radiation (dual y-axis)
  - Soil Moisture (with WHC line and correlation)
  - Runoff (with totals)
  - Evapotranspiration (with totals)

Custom date range plots:
  - Same 4-panel layout for any date range

Seasonal plots:
  - Monthly averages across all years
  - Shows seasonal patterns clearly

All plots include:
  ✓ Country/site name in title
  ✓ Observed vs modeled data
  ✓ Performance metrics
  ✓ Clear legends and labels
  ✓ High-resolution output (300 dpi)
""")

print("="*70)
print("DONE!")
print("="*70)
