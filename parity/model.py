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

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the self and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        out = out[:, -1]
        out = self.fc(out)
        # out = self.ReLU(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.ones(
            self.n_layers, batch_size, self.hidden_dim, device=self.device
        )
        return hidden

    # def sqr_dist(self, X: torch.Tensor):
    #     dist = (
    #         torch.diagonal(torch.tensordot(X, X, dims=([1], [1])))  # type: ignore
    #         + -2 * torch.tensordot(X, X.mT, dims=([1], [0]))  # type: ignore
    #         + torch.diagonal(torch.tensordot(X.mT, X.mT, dims=([0], [0])))[  # type: ignore
    #             :, np.newaxis
    #         ]
    #     )
    #     return dist

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


class Transformer(nn.Module):
    """


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
        # self.rnn = nn.RNN(
        #     input_size,
        #     hidden_dim,
        #     n_layers,
        #     batch_first=True,
        #     nonlinearity=nonlinearity,
        # )
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12)
        # self.fc = nn.Linear(hidden_dim, output_size)
        # self.ReLU = nn.LeakyReLU()

        for par in self.parameters():
            if len(par.shape) == 2:
                nn.init.xavier_normal_(par, gain=gain)
            if len(par.shape) == 1:
                nn.init.zeros_(par)

        self.to(device)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the self and obtaining outputs
        out, hidden = self.transformer(x, hidden)
        out = out[:, -1]
        # out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.ones(
            self.n_layers, batch_size, self.hidden_dim, device=self.device
        )
        return hidden

    def sqr_dist(self, X: torch.Tensor):
        dist = (
            torch.diagonal(torch.tensordot(X, X, dims=([1], [1])))  # type: ignore
            + -2 * torch.tensordot(X, X.mT, dims=([1], [0]))  # type: ignore
            + torch.diagonal(torch.tensordot(X.mT, X.mT, dims=([0], [0])))[  # type: ignore
                :, np.newaxis
            ]
        )
        return dist

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
