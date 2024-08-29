import numpy as np

def check_numeric(data : np.array):
    """Check if the array consists of numeric

    :param data: 2-dim array
    :type data: boolean
    """
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_):
        raise ValueError("All elements should be numeric.")
    # return True

def check_np(data):
    """Check if the dataset is numpy array for Eigen

    :param data: Table-format data
    :type data: Non
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            check_numeric(data)
            return data
        else:
            raise ValueError("Numpy array must be 2-dim.")
    elif isinstance(data, list):
        array_data = np.array(data)
        if array_data.ndim == 2:
            check_numeric(array_data)
            return array_data
        else:
            raise ValueError("np.array(list) should give 2-dim array.")
    else:
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                array_data = data.values
                check_numeric(array_data)
                return array_data
        except ImportError:
            pass
        # Add polars?
        raise ValueError("Unsupported data type.")