import ast
import numpy as np
import pandas as pd


def hello():
    print('hello')
    
    
def convert_cuisine(row: [str, np.nan]) -> [list, np.nan]:
    """ Convert string representation of list to list
        np.nan values leaves as it is
    """
    
    if row is not np.nan or not isinstance(row, list):
        return [x.strip() for x in ast.literal_eval(row)]

    return row
