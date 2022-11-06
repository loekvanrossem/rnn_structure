import torch
from torch import nn

import numpy as np

from compilation import CompileModel


class Model(CompileModel):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, device):
        super(Model, self).__init__()

        self.device = device

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        self.rnn = nn.RNN(
            input_size, hidden_dim, n_layers, batch_first=True, nonlinearity="relu"
        )
        self.fc = nn.Linear(hidden_dim, output_size)
        self.ReLU = nn.ReLU()

        for par in self.rnn.parameters():
            nn.init.normal_(par, mean=0, std=0.3)
        # torch.nn.init.constant(self.rnn, 0)

        self.to(device)

    def forward(self, x):

        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the self and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        out = out[:, -1]
        out = self.fc(out)
        out = self.ReLU(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(
            self.n_layers, batch_size, self.hidden_dim, device=self.device
        )
        return hidden

    def sqr_dist(self, X: torch.tensor):
        dist = (
            torch.diagonal(torch.tensordot(X, X, dims=([1], [1])))
            + -2 * torch.tensordot(X, X.mT, dims=([1], [0]))
            + torch.diagonal(torch.tensordot(X.mT, X.mT, dims=([0], [0])))[
                :, np.newaxis
            ]
        )
        return dist

    def train_step(self, optimizer, criterion, dataloader):
        self.train()
        for batch in dataloader:
            inputs, outputs = batch
            optimizer.zero_grad()

            output, hidden = self(inputs)

            l2_reg = torch.tensor(0.0, device=self.device)
            for param in self.parameters():
                l2_reg += torch.norm(param)

            # dist_reg = torch.norm(self.sqr_dist(torch.squeeze(hidden[-1])))

            # loss = 0.00005 * dist_reg + criterion(
            #     torch.squeeze(output), torch.squeeze(outputs)
            # )
            # loss = 0.0001 * l2_reg + criterion(
            #     torch.squeeze(output), torch.squeeze(outputs)
            # )
            loss = criterion(torch.squeeze(output), torch.squeeze(outputs))

            loss.backward()
            optimizer.step()
        return loss
