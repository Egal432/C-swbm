import complete_analysis as ca

files = [
    "Data/Data_swbm_Germany_new.csv",
    "Data/Data_swbm_Spain_new.csv",
    "Data/Data_swbm_Sweden_new.csv",
]

# This does EVERYTHING: runs model + creates all plots
all_results, summary = ca.run_complete_analysis(
    filepaths=files,
    output_dir="complete_outputs",
    whc=150.0,
    beta=0.8,
    create_plots=True,
)

# Also creates cross-site comparison
ca.quick_comparison_plot(all_results, output_dir="complete_outputs")
