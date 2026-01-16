"""
Unified main script for Simple Water Balance Model analysis
Runs complete calibration, validation, comparison, and visualization workflow
"""

import os
import sys
from datetime import datetime

# Add C-SWBM-main to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'C-SWBM-main'))


def print_header(text, char='='):
    """Print formatted header"""
    print(f"\n{char*70}")
    print(f"{text.center(70)}")
    print(f"{char*70}\n")


def main():
    """Main execution function"""
    
    print_header("SIMPLE WATER BALANCE MODEL - COMPLETE ANALYSIS", '=')
    
    # Configuration
    data_files = {
        'Germany': 'Data/Data_swbm_Germany_new.csv',
        'Spain': 'Data/Data_swbm_Spain_new.csv',
        'Sweden': 'Data/Data_swbm_Sweden_new.csv'
    }
    
    # Output directories
    calibration_dir = 'calibration_results'
    comparison_dir = 'model_comparison'
    
    # Parameter grid - MODIFY HERE TO CHANGE ALPHA RANGE
    param_grid = {
        'whc': [210.0, 420.0, 840.0],
        'exp_runoff': [2.0, 4.0, 8.0],
        'exp_et': [0.2, 0.5, 0.8],
        'beta': [0.4, 0.6, 0.8],
        'alpha': [0.2, 0.3, 0.8]  # For Spain issues, try: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    }
    
    print("Configuration:")
    print(f"  Sites: {', '.join(data_files.keys())}")
    print(f"  Calibration period: 2008-2013")
    print(f"  Validation period: 2014-2018")
    print(f"  Parameter combinations: {3**5} = 243")
    print(f"\nParameter ranges:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Check if files exist
    print_header("Checking Data Files", '-')
    all_exist = True
    for site, filepath in data_files.items():
        if os.path.exists(filepath):
            print(f"  ✓ {site}: {filepath}")
        else:
            print(f"  ✗ {site}: {filepath} NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\nERROR: Some data files are missing!")
        print("Please update file paths or place data files in correct location.")
        return
    
    # Ask user what to run
    print_header("Select Analysis Steps", '-')
    print("What would you like to run?")
    print("  1. Full calibration and validation (NEW model with groundwater)")
    print("  2. Compare OLD vs NEW models")
    print("  3. Generate visualizations")
    print("  4. Run everything (recommended)")
    print("  5. Custom selection")
    
    choice = input("\nEnter choice (1-5) [default: 4]: ").strip() or "4"
    
    run_calibration = False
    run_comparison = False
    run_visualization = False
    
    if choice == "1":
        run_calibration = True
    elif choice == "2":
        run_comparison = True
    elif choice == "3":
        run_visualization = True
    elif choice == "4":
        run_calibration = True
        run_comparison = True
        run_visualization = True
    elif choice == "5":
        run_calibration = input("  Run calibration? (y/n) [y]: ").strip().lower() != 'n'
        run_comparison = input("  Run OLD vs NEW comparison? (y/n) [y]: ").strip().lower() != 'n'
        run_visualization = input("  Generate visualizations? (y/n) [y]: ").strip().lower() != 'n'
    else:
        print("Invalid choice. Running everything.")
        run_calibration = run_comparison = run_visualization = True
    
    # Confirm before starting
    print_header("Analysis Plan", '-')
    if run_calibration:
        print("  ✓ Full calibration and validation")
        print(f"    - Test {3**len(param_grid)} parameter combinations per site")
        print("    - Calibrate on 2008-2013")
        print("    - Validate on 2014-2018")
        print(f"    - Output: {calibration_dir}/")
    if run_comparison:
        print("  ✓ OLD vs NEW model comparison")
        print("    - Compare models with/without groundwater")
        print(f"    - Output: {comparison_dir}/")
    if run_visualization:
        print("  ✓ Generate all visualizations")
        print("    - Parameter sensitivity plots")
        print("    - Validation timeseries")
        print("    - Cross-site comparisons")
    
    if not (run_calibration or run_comparison or run_visualization):
        print("Nothing selected to run. Exiting.")
        return
    
    print("\nEstimated time:")
    time_est = 0
    if run_calibration:
        time_est += 10
    if run_comparison:
        time_est += 15
    if run_visualization:
        time_est += 2
    print(f"  ~{time_est} minutes (depending on system)")
    
    confirm = input("\nProceed? (y/n) [y]: ").strip().lower()
    if confirm == 'n':
        print("Cancelled.")
        return
    
    start_time = datetime.now()
    
    # ========== STEP 1: CALIBRATION ==========
    if run_calibration:
        print_header("STEP 1: CALIBRATION AND VALIDATION", '=')
        try:
            # Import here to avoid issues if module not found
            import calibration_script as cs
            
            sites_results, performance_table = cs.run_complete_calibration_validation(
                data_files,
                output_dir=calibration_dir
            )
            
            print_header("CALIBRATION COMPLETE", '-')
            print(f"Results saved to: {calibration_dir}/")
            
        except ImportError as e:
            print(f"ERROR: Could not import calibration_script.py")
            print(f"Make sure calibration_script.py is in the same directory as main.py")
            print(f"Error: {e}")
            return
        except Exception as e:
            print(f"ERROR during calibration: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # ========== STEP 2: COMPARISON ==========
    if run_comparison:
        print_header("STEP 2: OLD vs NEW MODEL COMPARISON", '=')
        try:
            import compare_old_new as con
            
            summary = con.compare_all_sites(data_files, output_dir=comparison_dir)
            
            print_header("COMPARISON COMPLETE", '-')
            print(f"Results saved to: {comparison_dir}/")
            
        except ImportError as e:
            print(f"ERROR: Could not import compare_old_new.py")
            print(f"Make sure compare_old_new.py is in the same directory as main.py")
            print(f"Error: {e}")
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== STEP 3: VISUALIZATION ==========
    if run_visualization:
        print_header("STEP 3: GENERATING VISUALIZATIONS", '=')
        try:
            import compare_results as cr
            
            # Visualize calibration results
            if os.path.exists(calibration_dir):
                print("Generating calibration visualizations...")
                cr.generate_all_visualizations(calibration_dir)
            else:
                print(f"Skipping: {calibration_dir}/ not found")
            
            print_header("VISUALIZATION COMPLETE", '-')
            
        except ImportError as e:
            print(f"ERROR: Could not import compare_results.py")
            print(f"Make sure compare_results.py is in the same directory as main.py")
            print(f"Error: {e}")
        except Exception as e:
            print(f"ERROR during visualization: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== SUMMARY ==========
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_header("ANALYSIS COMPLETE", '=')
    print(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    print("\n📁 Output Files Generated:")
    
    if run_calibration and os.path.exists(calibration_dir):
        print(f"\n{calibration_dir}/")
        print("  ├── MODEL_PERFORMANCE_TABLE.csv        ← Main results for report")
        print("  ├── MODEL_PERFORMANCE_TABLE.txt        ← Formatted version")
        print("  ├── calibration_summary.json")
        print("  ├── [Site]_all_calibrations.csv        ← All 243 combinations")
        print("  ├── [Site]_validation_timeseries.csv")
        if run_visualization:
            print("  ├── [Site]_timeseries.png              ← With precipitation!")
            print("  ├── [Site]_scatter.png")
            print("  ├── [Site]_parameter_sensitivity.png")
            print("  ├── [Site]_top_combinations.png")
            print("  └── cross_site_comparison.png")
    
    if run_comparison and os.path.exists(comparison_dir):
        print(f"\n{comparison_dir}/")
        print("  ├── OLD_vs_NEW_summary.csv             ← Shows improvement")
        print("  ├── OLD_vs_NEW_summary_plot.png")
        print("  ├── [Site]_validation_comparison.csv")
        print("  └── [Site]_validation_comparison_plot.png")
    
    print("\n🎯 Key Files to Check:")
    print(f"  1. {calibration_dir}/MODEL_PERFORMANCE_TABLE.csv")
    print(f"  2. {comparison_dir}/OLD_vs_NEW_summary.csv")
    print(f"  3. {calibration_dir}/Spain_validation_timeseries.png  ← Check Spain's performance")
    
    print("\n💡 Next Steps:")
    if run_calibration:
        print("  - Review MODEL_PERFORMANCE_TABLE.csv for best parameters")
        print("  - Check validation correlations for each site")
    if run_comparison:
        print("  - Compare OLD vs NEW performance in summary files")
        print("  - Analyze which sites benefited most from groundwater component")
    print("  - If Spain's runoff is poor, consider wider alpha range:")
    print("    → Edit this main.py, change alpha to [0.1, 0.2, ..., 0.8]")
    print("    → Re-run calibration for Spain only")
    
    print_header("DONE", '=')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
