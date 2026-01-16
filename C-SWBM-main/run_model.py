"""
Simple script to run the water balance model and generate visualizations
"""

import argparse
import matplotlib.pyplot as plt
from CSWBM import SimpleWaterBalanceModel
from visualize_results import (
    plot_timeseries, 
    plot_scatter_comparison, 
    plot_monthly_comparison,
    print_statistics
)


def main():
    parser = argparse.ArgumentParser(description='Run Simple Water Balance Model')
    
    # Required arguments
    parser.add_argument('data_file', type=str, help='Path to input CSV data file')
    
    # Model parameters
    parser.add_argument('--exp_runoff', type=float, default=2.0, 
                       help='Runoff exponent parameter (default: 2.0)')
    parser.add_argument('--exp_et', type=float, default=0.5,
                       help='ET exponent parameter (default: 0.5)')
    parser.add_argument('--beta', type=float, default=0.8,
                       help='Beta parameter for ET calculation (default: 0.8)')
    parser.add_argument('--whc', type=float, default=150.0,
                       help='Water holding capacity in mm (default: 150.0)')
    parser.add_argument('--delta', type=float, default=0.5,
                       help='runoff splitting')   
    # Plotting options
    parser.add_argument('--start_date', type=str, default=None,
                       help='Start date for plotting (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date for plotting (YYYY-MM-DD)')
    parser.add_argument('--output_prefix', type=str, default='model_output',
                       help='Prefix for output files (default: model_output)')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display plots (only save)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SIMPLE WATER BALANCE MODEL")
    print("="*70)
    print(f"\nInput file: {args.data_file}")
    print(f"\nModel parameters:")
    print(f"  exp_runoff: {args.exp_runoff}")
    print(f"  exp_et:     {args.exp_et}")
    print(f"  beta:       {args.beta}")
    print(f"  WHC:        {args.whc} mm")
    print("\n" + "="*70)
    
    # Initialize model
    model = SimpleWaterBalanceModel(
        exp_runoff=args.exp_runoff,
        exp_et=args.exp_et,
        beta=args.beta,
        whc=args.whc,
        delta=args.delta
        )
    
    # Load data
    print("\nLoading data...")
    try:
        data = model.load_data(args.data_file)
        print(f"Loaded {len(data)} days of data")
        print(f"Date range: {data['time'].min()} to {data['time'].max()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run model
    print("\nRunning model...")
    try:
        results = model.run()
        print("Model run completed successfully")
    except Exception as e:
        print(f"Error running model: {e}")
        return
    
    # Generate statistics
    print_statistics(data, results)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    try:
        # Time series plot
        print("  Creating time series plot...")
        fig1 = plot_timeseries(data, results, args.start_date, args.end_date)
        filename1 = f"{args.output_prefix}_timeseries.png"
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename1}")
        
        # Scatter comparison
        print("  Creating scatter comparison plot...")
        fig2 = plot_scatter_comparison(data, results)
        filename2 = f"{args.output_prefix}_scatter.png"
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename2}")
        
        # Monthly comparison
        print("  Creating monthly comparison plot...")
        fig3 = plot_monthly_comparison(data, results)
        filename3 = f"{args.output_prefix}_monthly.png"
        plt.savefig(filename3, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename3}")
        
        print("\nAll visualizations generated successfully!")
        
        # Show plots if requested
        if not args.no_show:
            print("\nDisplaying plots... (close windows to exit)")
            plt.show()
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        return
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
