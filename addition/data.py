from typing import Callable, Optional

import random
import numpy as np

import torch
from torch.utils.data import TensorDataset

# from preprocessing import Encoding


def random_int(length: int):
    return int("".join([str(np.random.choice(range(10))) for _ in range(length)]))


def addition_datapoint(length: int, full_length: int):
    int_a_len = np.random.randint(1, length - 1)
    int_b_len = length - 1 - int_a_len
    # int_a = np.random.randint(10 ** (int_a_len - 1), 10 ** (int_a_len))
    # int_b = np.random.randint(10 ** (int_b_len - 1), 10 ** (int_b_len))
    int_a, int_b = random_int(int_a_len), random_int(int_b_len)
    input = f"{int_a}+{int_b}"
    output = f"{int_a + int_b}"
    input = input + " " * (full_length - len(input))
    output = output + " " * (full_length - len(output))
    return input, output


def addition_dataset(
    device: torch.device,
    encoding,
    n_datapoints: int,
    seq_len: list[int],
    full_length: int,
) -> TensorDataset:
    """
    Generate data for performing integer addition.

    e.g. input: "23+7" output: "30"

    Parameters
    ----------
    device : Device
        The device to put the data on
    encoding : Encoding
        How the input and output data will be encoded
    n_datapoints : int
        The maximum number of sequences to generate
    seq_len : list[int]
        Possible number of symbols for the input strings, these cannot be less than 3

    Returns
    -------
    dataset : TensorDataset
        Contains the inputs and outputs
    """
    inputs, outputs = [], []
    for _ in range(n_datapoints):
        length = np.random.choice(seq_len)
        input, output = addition_datapoint(length, full_length)
        input, output = encoding(input), encoding(output)
        inputs.append(input)
        outputs.append(output)

    # symbols = encoding.symbols
    # if n_datapoints is None:
    #     n_datapoints = len(symbols) ** seq_len
    # # Generate input sequences
    # inputs = gen_rand_seq(seq_len, symbols, n_datapoints)

    # # Compute outputs
    # outputs = np.apply_along_axis(problem, 1, inputs)

    # Prepare for torch
    inputs, outputs = np.array(inputs), np.array(outputs)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
    outputs = torch.from_numpy(outputs.astype(np.float32)).to(device)
    dataset = TensorDataset(inputs, outputs)
    return dataset
