import pandas as pd
pd.set_option('display.max_columns', None)
import pickle
import numpy as np

data = pd.read_csv('../../data/formspring_data_1.csv', sep = '\t')