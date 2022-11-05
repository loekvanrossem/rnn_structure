import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


def hidden_repr(hidden_states: pd.DataFrame):
    """
    Show a low dimensional representation of the hidden state dynamics

    Parameters
    ----------
    hidden_states : Dataframe (Epoch, Dataset, Input)
        Dataframe containing the hidden dimension values for each input and epoch
    """
    # Apply dimensionality reduction
    index = hidden_states.index
    mds = MDS(n_components=2)
    x = mds.fit_transform(hidden_states)
    x = pd.DataFrame(x, index=index)

    for dataset_name, dataset in x.groupby("Dataset"):
        for input_name, input in x.groupby("Input"):
            ax = plt.scatter(
                input[0],
                input[1],
                label=input_name,
            )
    plt.legend(loc="upper left")
    plt.show()

    return x
