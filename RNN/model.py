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
    output_noise : float, default 0
        Add noise to the true outputs during training
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
        output_noise: float = 0.0,
    ):
        super(Model, self).__init__()

        self.device = device
        self.encoding = encoding
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_noise = output_noise

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
            noise = np.random.normal(size=outputs.size()) * self.output_noise
            outputs += torch.from_numpy(noise).to(self.device)
            optimizer.zero_grad()

            output, hidden = self(inputs)

            # l2_reg = torch.tensor(0.0, device=self.device)
            # for param in self.parameters():
            #     l2_reg += torch.norm(param)

            # dist_reg = torch.norm(self.sqr_dist(torch.squeeze(hidden[-1])))

            loss = criterion(torch.squeeze(output), torch.squeeze(outputs))

            # loss = 0.5 * dist_reg + criterion(
            #     torch.squeeze(output), torch.squeeze(outputs)
            # )
            # loss = 0.01 * l2_reg + criterion(
            #     torch.squeeze(output), torch.squeeze(outputs)
            # )

            loss.backward()
            optimizer.step()
            av_loss += loss / len(dataloader)
        return av_loss
