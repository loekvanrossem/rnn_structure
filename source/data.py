from typing import Callable, Optional

import random
import numpy as np

import torch
from torch.utils.data import TensorDataset

from preprocessing import Encoding


def gen_rand_seq(n, symbols, n_sequences):
    """Generate random sequences of length n"""
    sequences = []
    while len(sequences) < min(n_sequences, len(symbols) ** n):
        seq = [random.choice(list(symbols)) for _ in range(n)]
        if not seq in sequences:
            sequences.append(seq)
    return sequences


def seq_data(
    device: torch.device,
    problem: Callable,
    encoding: Encoding,
    n_datapoints: Optional[int] = None,
    seq_len: int = 4,
) -> TensorDataset:
    """
    Generate data solving some problem on sequences.

    Parameters
    ----------
    device : Device
        The device to put the data on
    problem : function
        A function on sequences used to compute output data
    encoding : Encoding
        How the input and output data will be encoded
    n_datapoints : int, default all
        The maximum number of sequences to generate
    seq_len : int, default 4
        The number of symbols in each sequences

    Returns
    -------
    dataset : TensorDataset
        Contains the inputs and outputs
    """
    symbols = encoding.symbols
    if n_datapoints is None:
        n_datapoints = len(symbols) ** seq_len
    # Generate input sequences
    inputs = gen_rand_seq(seq_len, symbols, n_datapoints)

    # Compute outputs
    outputs = np.apply_along_axis(problem, 1, inputs)

    # Prepare for torch
    inputs, outputs = encoding(inputs), encoding(outputs)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
    outputs = torch.from_numpy(outputs.astype(np.float32)).to(device)
    dataset = TensorDataset(inputs, outputs)
    return dataset


def grid_data(
    device: torch.device,
    dim: int = 2,
    output_dim: int = 2,
    n: int = 100,
    bounds: tuple[float, float] = (0, 1),
) -> TensorDataset:
    """
    Generate a grid of datapoints.

    Usefull for plotting the network as a map.

    Parameters
    ----------
    device : Device
        The device to put the data on
    dim : int, default 2
        The number of dimensions
    n : int, default 100
        The number of values per dimension, total number of datapoints is n**dim
    bounds : (float,float) default (0,1)
        The interval in which the datapoints will be contained

    Returns
    -------
    dataset : TensorDataset
        Contains the inputs and nans for outputs
    """
    # Generate inputs
    X = np.linspace(bounds[0], bounds[1], n)
    grid = np.meshgrid(*[X] * dim)
    inputs = np.array([np.ravel(grid[i]) for i in range(dim)]).T
    inputs = np.array([[input] for input in inputs])

    # Generate outputs
    outputs = np.nan * np.ones([len(inputs), output_dim])

    # Prepare for torch
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
    outputs = torch.from_numpy(outputs.astype(np.float32)).to(device)
    dataset = TensorDataset(inputs, outputs)
    return dataset


def fun_data(
    device: torch.device,
    function: Callable,
    bounds: tuple[float, float],
    n_datapoints: int,
    input_dim: int = 1,
    random: bool = False,
) -> TensorDataset:
    """
    Generate data solving some problem on sequences.

    Parameters
    ----------
    device : Device
        The device to put the data on
    function : function
        A function on inputs used to compute output data
    bounds : tuple(float, float)
        The range from which inputs are drawn
    n_datapoints : int
        The maximum number of datapoints to generate
    input_dim : int, optional, default 1
        The dimension of the input data
    random : bool, optional, default False
        If true, sample points from uniform distribution

    Returns
    -------
    dataset : TensorDataset
        Contains the inputs and outputs
    """
    if random:
        inputs = np.array(
            [np.random.uniform(bounds[0], bounds[1], (input_dim, n_datapoints))]
        )
    else:
        inputs = np.array([np.linspace(bounds[0], bounds[1], n_datapoints)] * input_dim)
    inputs = inputs.reshape(-1, input_dim)

    # Compute outputs
    outputs = np.apply_along_axis(function, 1, inputs)

    # Prepare for torch
    # inputs, outputs = encoding(inputs), encoding(outputs)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
    outputs = torch.from_numpy(outputs.astype(np.float32)).to(device)
    dataset = TensorDataset(inputs, outputs)
    return dataset
