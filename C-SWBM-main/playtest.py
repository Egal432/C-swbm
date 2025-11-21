import pandas as pd
import csv

test_data = pd.read_csv("Data/Data_swbm_Germany_new.csv")
test_data[0:10:]

test_data[0:10].to_csv('test_data.csv',header=True)
