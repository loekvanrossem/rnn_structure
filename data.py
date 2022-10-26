import numpy as np
import torch
from torch.utils.data import TensorDataset


def XOR_data(device, n_datapoints=100, seq_len=4):
    inputs = np.zeros([n_datapoints, seq_len, 1], dtype=np.float32)
    outputs = np.zeros([n_datapoints, 1], dtype=np.float32)
    for i in range(n_datapoints):
        # Generate input sequences
        for j in range(seq_len):
            bit = np.random.choice([0, 1])
            inputs[i, j, 0] = bit

        # Compute output
        problem = np.nansum(inputs[i]) % 2  # Parity
        # problem = np.sum(inputs[i]) ** 2 % 3  # Not multiple of 3
        outputs[i] = problem

    inputs = torch.from_numpy(inputs).to(device)
    outputs = torch.from_numpy(outputs).to(device)
    dataset = TensorDataset(inputs, outputs)
    return dataset
