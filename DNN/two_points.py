import sys

import numpy as np

import torch
from torch.utils.data import TensorDataset

source = "../source"
sys.path.append(source)

from preprocessing import Direct, Encoding, OneHot


def data_set(dx2, dy2, input_dim, output_dim, device):
    inputs = np.array([[-1] * input_dim, [-1 + np.sqrt(dx2)] * input_dim]) / np.sqrt(
        input_dim
    )
    outputs = np.array(
        [[0.6] * output_dim, [0.6 + np.sqrt(dy2)] * output_dim]
    ) / np.sqrt(output_dim)
    names = ["A", "B"]
    data = TensorDataset(
        torch.from_numpy(inputs.astype(np.float32)).to(device),
        torch.from_numpy(outputs.astype(np.float32)).to(device),
    )

    encoding = Encoding(dict(zip(names, inputs)))

    return data, encoding


def get_h_y_w(data, model, hidden_layer):
    input_1 = data[0][0]
    input_2 = data[1][0]
    output_1 = data[0][1]
    output_2 = data[1][1]
    hid_1 = model(input_1)[1][hidden_layer]
    hid_2 = model(input_2)[1][hidden_layer]
    pred_1 = model(input_1)[0]
    pred_2 = model(input_2)[0]
    input_1, input_2, output_1, output_2, hid_1, hid_2, pred_1, pred_2 = [
        a.cpu().detach().numpy()
        for a in (input_1, input_2, output_1, output_2, hid_1, hid_2, pred_1, pred_2)
    ]

    h = np.linalg.norm(hid_2 - hid_1) ** 2
    y = np.linalg.norm(pred_2 - pred_1) ** 2
    w = y - np.dot(output_2 - output_1, pred_2 - pred_1).item()

    return h, y, w
