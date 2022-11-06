import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA


def hidden_repr(hidden_states: pd.DataFrame, transform="PCA", n_labels=10):
    """
    Show a low dimensional representation of the hidden state dynamics

    Parameters
    ----------
    hidden_states : Dataframe (Epoch, Dataset, Input)
        Dataframe containing the hidden dimension values for each input and epoch
    transform : String
        The type of dimensionality reduction to use: PCA or MDS
    n_labels : int, default 10
        The number of epochs which will be labeled
    """
    # Apply dimensionality reduction
    index = hidden_states.index
    if transform == "MDS":
        mds = MDS(n_components=2)
        x = mds.fit_transform(hidden_states)
    if transform == "PCA":
        pca = PCA(n_components=2)
        x = pca.fit_transform(hidden_states)
    x = pd.DataFrame(x, index=index)

    for dataset_name, dataset in x.groupby("Dataset"):
        for input_name, input in dataset.groupby("Input"):
            plt.plot(input[0], input[1], label=input_name, zorder=0)
            # Label some points
            first_epoch = hidden_states.index.get_level_values("Epoch").min()
            last_epoch = hidden_states.index.get_level_values("Epoch").max()
            log_range = np.logspace(
                np.log(first_epoch + 0.999),
                np.log(last_epoch + 1),
                num=n_labels,
                base=np.e,
            ).astype(int)
            for epoch in log_range:
                plt.scatter(input[0][epoch], input[1][epoch], s=10, c="Black", zorder=1)
                plt.annotate(
                    epoch,
                    (input[0][epoch] + 0.02, input[1][epoch] + 0.02),
                    zorder=2,
                )
    plt.legend(loc="upper left")
    plt.show()

    return x
