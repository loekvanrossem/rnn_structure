def hidden_repr(hidden_states: list):
    """
    Show a low dimensional representation of the hidden state dynamics

    Parameters
    ----------
    hidden_states : list (n_epochs, n_val_datasets, n_datapoints, n_hidden_dim)
        list of arrays per validation dataset per epoch
    """
