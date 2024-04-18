import pandas as pd
import numpy as np
#import numba
from concurrent.futures import ProcessPoolExecutor

#@numba.jit(nopython=True)
def scale_values(values, min_val, max_val):
    return (values - min_val) / (max_val - min_val)

def scale_rowwise(df, rows):
    values = df.loc[rows].values
    min_val = np.min(values)
    max_val = np.max(values)
    scaled_values = scale_values(values, min_val, max_val)
    df.loc[rows] = scaled_values
    return df

def scale_dataframe(dataframe):
    # Identify rows
    price_rows = dataframe.index[dataframe.index.str.contains('price')]
    size_rows = dataframe.index[dataframe.index.str.contains('size')]

    # Scale rows separately
    dataframe = scale_rowwise(dataframe, price_rows)
    dataframe = scale_rowwise(dataframe, size_rows)
    
    return dataframe

#def scale_dataframe(dataframe):
#    # Ensure the index is in string format if it contains string data
#    if dataframe.index.dtype == 'object':
#        dataframe.index = dataframe.index.astype(str)
#
#    # Now you can safely use .str

