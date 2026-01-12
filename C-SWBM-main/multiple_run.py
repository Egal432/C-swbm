"""
Compact function to run Simple Water Balance Model on multiple datasets
"""

import numpy as np
import pandas as pd
from CSWBM import SimpleWaterBalanceModel


def run_swbm(data, exp_runoff=2.0, exp_et=0.5, beta=0.8, whc=150.0):
    """
    Run Simple Water Balance Model on a dataset.

    Parameters
    ----------
    data : str or pd.DataFrame
        Either filepath to CSV or preprocessed DataFrame
    exp_runoff : float
        Runoff exponent parameter (default: 2.0)
    exp_et : float
        ET exponent parameter (default: 0.5)
    beta : float
        Beta parameter for ET (default: 0.8)
    whc : float
        Water holding capacity in mm (default: 150.0)

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with all inputs, outputs, and observations
    metrics : dict
        Performance metrics (if observations available)
    """
    # Initialize model
    model = SimpleWaterBalanceModel(
        exp_runoff=exp_runoff,
        exp_et=exp_et,
        beta=beta,
        whc=whc,
    )

    # Load data if filepath provided
    if isinstance(data, str):
        input_data = model.load_data(data)
    else:
        input_data = data.copy()
        model.data = input_data

    # Run model
    results = model.run()

    # Create output dataframe
    results_df = pd.DataFrame(
        {
            "time": input_data["time"].values,
            "precipitation": input_data["tp"].values,
            "radiation": input_data["snr"].values,
            "modeled_sm": results["soilmoisture"],
            "modeled_ro": results["runoff"],
            "modeled_et": results["evapotranspiration"],
        }
    )

    # Add observations if available
    if "sm" in input_data.columns:
        results_df["observed_sm"] = input_data["sm"].values
    if "ro" in input_data.columns:
        results_df["observed_ro"] = input_data["ro"].values
    if "le" in input_data.columns:
        results_df["observed_et"] = input_data["le"].values

    # Calculate metrics
    metrics = {}

    for var, obs_col, mod_col in [
        ("sm", "observed_sm", "modeled_sm"),
        ("ro", "observed_ro", "modeled_ro"),
        ("et", "observed_et", "modeled_et"),
    ]:
        if obs_col in results_df.columns:
            obs = results_df[obs_col].values
            mod = results_df[mod_col].values
            mask = ~(np.isnan(obs) | np.isnan(mod))

            if mask.sum() > 0:
                obs_valid = obs[mask]
                mod_valid = mod[mask]

                metrics[var] = {
                    "rmse": np.sqrt(np.mean((obs_valid - mod_valid) ** 2)),
                    "mae": np.mean(np.abs(obs_valid - mod_valid)),
                    "r": np.corrcoef(obs_valid, mod_valid)[0, 1],
                    "bias": np.mean(mod_valid - obs_valid),
                    "nse": 1
                    - (
                        np.sum((obs_valid - mod_valid) ** 2)
                        / np.sum((obs_valid - np.mean(obs_valid)) ** 2)
                    ),
                }

    return results_df, metrics


def run_swbm_multiple(filepaths, output_dir="outputs", **model_params):
    """
    Run SWBM on multiple datasets and save results.

    Parameters
    ----------
    filepaths : list of str
        List of file paths to process
    output_dir : str
        Directory to save outputs (default: 'outputs')
    **model_params : dict
        Model parameters (exp_runoff, exp_et, beta, whc, etc.)

    Returns
    -------
    all_results : dict
        Dictionary with filenames as keys and (results_df, metrics) as values
    summary : pd.DataFrame
        Summary table with metrics for all files
    """
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    summary_list = []

    print(f"Processing {len(filepaths)} files...")
    print(f"Parameters: {model_params}\n")

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        site_name = (
            filename.replace(".csv", "").replace("Data_swbm_", "").replace("_new", "")
        )

        print(f"Processing: {filename}")

        try:
            # Run model
            results_df, metrics = run_swbm(filepath, **model_params)

            print(f"  Generated {len(results_df)} rows of data")

            # Save results
            output_path = os.path.join(output_dir, f"{site_name}_results.csv")
            results_df.to_csv(output_path, index=False)

            # Store results
            all_results[site_name] = (results_df, metrics)

            # Add to summary
            summary_row = {
                "site": site_name,
                "file": filename,
                "n_days": len(results_df),
            }

            for var, var_metrics in metrics.items():
                for metric_name, value in var_metrics.items():
                    summary_row[f"{var}_{metric_name}"] = value

            summary_list.append(summary_row)

            # Print metrics
            if metrics:
                for var, var_metrics in metrics.items():
                    print(
                        f"  {var.upper()}: NSE={var_metrics['nse']:.3f}, "
                        f"R={var_metrics['r']:.3f}, RMSE={var_metrics['rmse']:.3f}"
                    )

            print(f"  ✓ Saved to {output_path}\n")

        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            import traceback

            traceback.print_exc()

    # Create summary dataframe
    if summary_list:
        summary = pd.DataFrame(summary_list)
        summary_path = os.path.join(output_dir, "summary_metrics.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to {summary_path}")
        print("\nSummary:")
        print(summary.to_string(index=False))
    else:
        summary = pd.DataFrame()
        print("No results generated.")

    return all_results, summary


if __name__ == "__main__":
    # Test with your files
    files = [
        "Data/Data_swbm_Germany_new.csv",
        "Data/Data_swbm_Spain_new.csv",
        "Data/Data_swbm_Sweden_new.csv",
    ]

    all_results, summary = run_swbm_multiple(
        files, output_dir="outputs", exp_runoff=2.0, exp_et=0.5, beta=0.8, whc=420.0
    )

