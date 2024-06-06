from typing import Callable, Optional

import random
import numpy as np

import torch
from torch.utils.data import TensorDataset

# from preprocessing import Encoding


def random_int(length: int, base: int = 10):
    return "".join([str(np.random.choice(range(base))) for _ in range(length)])


def addition_datapoint(
    length: int,
    full_length: int,
    binary: bool = False,
):
    base = 2 if binary else 10

    int_a_len = np.random.randint(1, length - 1)
    int_b_len = length - 1 - int_a_len
    int_a, int_b = random_int(int_a_len, base), random_int(int_b_len, base)
    if binary:
        sum = bin(int(int_a, 2) + int(int_b, 2))[2:]
    else:
        sum = int(int_a, 10) + int(int_b, 10)
    # if binary:
    #     int_a, int_b, sum = [bin(x)[2:] for x in (int_a, int_b, sum)]
    input = f"{int_a}+{int_b}"
    output = f"{sum}"
    input = input + " " * (full_length - len(input))
    output = output + " " * (full_length - len(output))
    return input, output


def addition_dataset(
    device: torch.device,
    encoding,
    n_datapoints: int,
    seq_len: list[int],
    full_length: int,
    binary: bool = False,
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
        input, output = addition_datapoint(length, full_length, binary=binary)
        input, output = encoding(input), encoding(output)
        inputs.append(input)
        outputs.append(output)

    # Prepare for torch
    inputs, outputs = np.array(inputs), np.array(outputs)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
    outputs = torch.from_numpy(outputs.astype(np.float32)).to(device)
    dataset = TensorDataset(inputs, outputs)
    return dataset
