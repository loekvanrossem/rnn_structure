import numpy as np
import torch
from torch.utils.data import TensorDataset
import random


# def gen_seq(n, symbols):
#     """Generate all sequences of length n."""
#     if n == 1:
#         return [[x] for x in symbols]
#     else:
#         return sum(
#             [[seq + [x]] for x in symbols for seq in gen_seq(n - 1, symbols)], []
#         )


def gen_rand_seq(n, symbols, n_sequences):
    """Generate random sequences of length n"""
    sequences = []
    while len(sequences) < min(n_sequences, len(symbols) ** n):
        seq = [random.choice(list(symbols)) for _ in range(n)]
        if not seq in sequences:
            sequences.append(seq)
    return sequences


def seq_data(device, symbols, problem, n_datapoints=None, seq_len=4):
    """
    Generate data solving some problem on sequences.

    Parameters
    ----------
    device : Device
        The device to put the data on
    symbols : set
        The symbols that sequences can contain
    problem : function
        A function on sequences used to compute output data
    n_datapoints : int, default all
        The maximum number of sequences to generate
    seq_len : int, default 4
        The number of symbols in each sequences

    Returns
    -------
    dataset : Torch Dataset
        Contains the inputs and outputs
    """
    if n_datapoints is None:
        n_datapoints = len(symbols) ** seq_len
    # Generate input sequences
    inputs = gen_rand_seq(seq_len, symbols, n_datapoints)
    inputs = np.array(inputs, dtype=np.float32)

    # Compute outputs
    outputs = np.apply_along_axis(problem, 1, inputs).astype(np.float32)

    # Prepare for torch
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 1)
    inputs = torch.from_numpy(inputs).to(device)
    outputs = torch.from_numpy(outputs).to(device)
    dataset = TensorDataset(inputs, outputs)
    return dataset
