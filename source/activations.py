from typing import Callable

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing import Encoding


def get_activations(
    datasets: list[Dataset], output_function: Callable, encoding: Encoding
) -> pd.DataFrame:
    """
    Get the activations in response to input data.

    Parameters
    ----------
    dataset : list[Dataset]
        A list of datasets
    output_function : Callable
        Function defining the activation response to an input
    encoding : Encoding
        Encoding that was used on the inputs

    Returns
    -------
    activations : pd.Dataframe(Dataset, Input)
        Dataframe containing the activations
    """
    activations = []
    warn_label = False
    for dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for batch in dataloader:
            inputs, outputs = batch

            # Get labels
            labels = []
            for input in inputs:
                try:
                    decoding = encoding.decode(input.cpu())
                    label = "".join(str(char) for char in decoding)
                except KeyError:
                    label = tuple(np.squeeze(input.cpu()).numpy())
                    warn_label = True
                labels.append(label)

            # Store activities
            act_this_dataset = output_function(inputs)
            act_this_dataset = act_this_dataset.cpu().detach().numpy()
            act_this_dataset = pd.DataFrame(act_this_dataset, labels)
            activations.append(act_this_dataset)
    activations = pd.concat(activations, keys=list(range(len(datasets))))
    activations.index = activations.index.set_names(["Dataset", "Input"])

    if warn_label:
        print("Some input labels were not recognized.")

    return activations
