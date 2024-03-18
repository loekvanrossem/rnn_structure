import numpy as np
import pandas as pd


def to_ndarray(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a multi-indexed pandas dataframe to a multidimensional array.

    Returns
    -------
    array : np.ndarray
        An array containing the dataframe entries
        each dimension of the array corresponds to an index of the dataframe
        and the final dimension corresponds to the column index
    """
    if df.index.nlevels == 1:
        return df.to_numpy()

    groups = df.groupby(level=0)
    sub_arrays = [to_ndarray(group.droplevel(level=0)) for _, group in groups]
    return np.array(sub_arrays)


def to_labels(df: pd.DataFrame) -> list[np.ndarray]:
    """Convert a multiindexed pandas dataframe to multidimensional arrays containing the labels.

    Returns
    -------
    labels : list[np.ndarray]
        An array for each level containing the labels
        each dimension of the array corresponds to an index of the dataframe
        ordering is consistent with to_ndarray function"""
    df = df.copy()
    labels = []
    for level in range(df.index.nlevels):
        label = df.index.get_level_values(level).to_numpy()
        labels.append(label)
    return labels
