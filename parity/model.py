import torch
from torch import nn
from torch.optim import Optimizer

import numpy as np

from preprocessing import Encoding


class Model(nn.Module):
    """
    Many to one RNN

    Attributes
    ----------
    encoding : Encoding
        An encoding from input symbols to neural activities
    input_size : int
        The size of one input symbol
    output_size : int
        The size of the output
    hidden_dim : int
        The number of hidden neurons
    n_layers : int
        The number of RNN layers
    device : device
        The device to put the model on
    nonlinearity : str
        Nonlinearity used in recurrent layer
    gain : float, default 0.1
        The gain of the initial rnn weights
    """

    def __init__(
        self,
        encoding: Encoding,
        input_size: int,
        output_size: int,
        hidden_dim: int,
        n_layers: int,
        device: torch.device,
        nonlinearity: str = "tanh",
        gain: float = 0.1,
    ):
        super(Model, self).__init__()

        self.device = device
        self.encoding = encoding
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        self.rnn = nn.RNN(
            input_size,
            hidden_dim,
            n_layers,
            batch_first=True,
            nonlinearity=nonlinearity,
        )
        self.fc = nn.Linear(hidden_dim, output_size)
        self.ReLU = nn.LeakyReLU()

        for par in self.parameters():
            if len(par.shape) == 2:
                nn.init.xavier_normal_(par, gain=gain)
            if len(par.shape) == 1:
                nn.init.zeros_(par)

        self.to(device)

    def forward(self, x):
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)
        out = out[:, -1]
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.ones(
            self.n_layers, batch_size, self.hidden_dim, device=self.device
        )
        return hidden

    def train_step(self, optimizer: Optimizer, criterion, dataloader):
        self.train()
        av_loss = 0
        for batch in dataloader:
            inputs, outputs = batch
            optimizer.zero_grad()

            output, hidden = self(inputs)

            loss = criterion(torch.squeeze(output), torch.squeeze(outputs))

            loss.backward()
            optimizer.step()
            av_loss += loss / len(dataloader)
        return av_loss
