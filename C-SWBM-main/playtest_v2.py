import multiple_run as mr
import pandas as pd
# Single file - simplest usage
results_df, metrics = mr.run_swbm('test_data.csv')

# Multiple files with same parameters
files = ['Data/Data_swbm_Germany_new.csv', 
         'Data/Data_swbm_Spain_new.csv', 
         'Data/Data_swbm_Sweden_new.csv']
all_results, summary = mr.run_swbm_multiple(files, whc=150.0, beta=0.8)