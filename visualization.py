import numpy as np
from sklearn.manifold import MDS


def hidden_repr(hidden_states: list):
    """
    Show a low dimensional representation of the hidden state dynamics

    Parameters
    ----------
    hidden_states : list (n_epochs, n_val_datasets, array(n_datapoints, n_hidden_dim))
        list of arrays per validation dataset per epoch
    """
    n_epochs = len(hidden_states)
    # get all states as points n_hidden_dim dimensional space
    points = []
    for epoch in range(n_epochs):
        for dataset in hidden_states[epoch]:
            for datapoint in dataset:
                points.append(datapoint)
    points = np.array(points)

    # Apply dimensionality reduction
    mds = MDS(n_components=2)
    x = mds.fit_transform(points)

    return x
