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
    if df.index.nlevels == 1:
        return df.to_numpy()

    a = []
    for x in df.groupby(level=0):
        data = x[1].droplevel(level=0)
        a.append(to_ndarray(data))

    a = np.array(a)
    return a


def to_labels(df: pd.DataFrame) -> list[np.ndarray]:
    """Convert a multiindexed pandas dataframe to multidimensional arrays containing the labels.

    Returns
    -------
    labels : list[np.ndarray]
        An array for each level containting the labels
        each dimensions of the array corresponds to an index of the dataframe
        ordering is consistent with to_ndarray function"""
    nlevels = df.index.nlevels
    labels = []
    for level in range(nlevels):
        label = df.copy().iloc[:, 0:1]
        for index, entry in label.iterrows():
            if nlevels == 1:
                label.loc[index] = index
            else:
                label.loc[index] = index[level]
        labels.append(to_ndarray(label).squeeze(axis=-1))
    return labels


# def to_ndarray(df: pd.DataFrame) -> np.ndarray:
#     """
#     Convert a multiindexed pandas dataframe to a multidimensional array.

#     Returns
#     -------
#     array : np.ndarray
#         An array containting the dataframe entries
#         each dimensions of the array corresponds to an index of the dataframe
#         and the final dimension corresponding to the column index
#     """

#     dims = [len(df.index.get_level_values(i).unique()) for i in range(df.index.nlevels)]
#     dims = dims + [-1]
#     array = np.reshape(df.to_numpy(), dims)

#     return array
