import numpy as np
import pandas as pd


def to_ndarray(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a multiindexed pandas dataframe to a multidimensional array.

    Returns
    -------
    array : np.ndarray
        An array containting the dataframe entries
        each dimensions of the array corresponds to an index of the dataframe
        and the final dimension corresponding to the column index
    """

    dims = [len(df.index.get_level_values(i).unique()) for i in range(df.index.nlevels)]
    dims = dims + [-1]
    array = np.reshape(df.to_numpy(), dims)

    return array
