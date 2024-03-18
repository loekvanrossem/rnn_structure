from typing import Callable, List

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing import Encoding


def get_activations(
    datasets: List[Dataset], output_function: Callable, encoding: Encoding
) -> pd.DataFrame:
    """
    Get the activations in response to input data.

    Parameters
    ----------
    datasets : List[Dataset]
        A list of datasets
    output_function : Callable
        Function defining the activation response to an input
    encoding : Encoding
        Encoding that was used on the inputs

    Returns
    -------
    activations : pd.DataFrame(Dataset, Input)
        Dataframe containing the activations
    """
    activations = []
    unrecognized_labels = False

    # Iterate through datasets
    for dataset_index, dataset in enumerate(datasets):
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        # Iterate through batches
        for inputs, _ in dataloader:
            # Get labels
            labels = []
            for input in inputs:
                try:
                    decoding = encoding.decode(input.cpu())
                    label = "".join(str(char) for char in decoding)
                except KeyError:
                    label = tuple(np.squeeze(input.cpu()).numpy())
                labels.append(label)

            # Check for unrecognized labels
            if any(label is None for label in labels):
                unrecognized_labels = True

            # Get activations
            activations_this_dataset = output_function(inputs)
            activations_this_dataset = activations_this_dataset.cpu().detach().numpy()

            # Store activations in DataFrame
            activation_df = pd.DataFrame(activations_this_dataset, index=labels)
            activations.append(activation_df)

    # Combine activations into a single DataFrame
    activations = pd.concat(activations, keys=list(range(len(datasets))))
    activations.index = activations.index.set_names(["Dataset", "Input"])

    # Warn if unrecognized labels were encountered
    if unrecognized_labels:
        print("Some input labels were not recognized.")

    return activations
