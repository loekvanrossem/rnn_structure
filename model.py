import torch
from torch import nn

import numpy as np


class Model(nn.Module):
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

    def validation(self, criterion, validation_datasets, track=False):
        """
        Compute loss of validation datasets

        Parameters
        ----------
        criterion
        validation_datasets : list of datasets
        track : boolean default false
            If true output hidden_states

        Returns
        -------
        losses : np.array(n)
        hidden_states : list arrays(n_datapoints, n_hidden_dim) per dataset
        """
        self.eval()
        losses = np.zeros(len(validation_datasets))
        hidden_states = []
        for i, dataset in enumerate(validation_datasets):
            valloader = torch.utils.data.DataLoader(
                dataset, batch_size=len(dataset), shuffle=False
            )
            for batch in valloader:
                inputs, outputs = batch
                prediction, hidden = self(inputs)
            losses[i] = criterion(torch.squeeze(prediction), torch.squeeze(outputs))
            if track:
                hidden_states.append(torch.squeeze(hidden).cpu().detach().numpy())

        if track:
            return losses, hidden_states
        return losses

    def training_run(
        self,
        optimizer,
        criterion,
        training_datasets,
        validation_datasets,
        n_epochs=100,
        batch_size=32,
    ):
        """
        Train the network on training datasets

        Parameters
        ----------
        optimizer
        criterion
        training_datasets: list[datasets]
        validation_datasets: list[datasets]
        n_epochs: int, default 100
        batch_size: int default 32
            batch size during training

        Returns
        -------
        training_losses: array(n_epochs)
            The losses of the last training dataset
        val_losses: array(n_epochs, len(validation_datasets))
            The losses per validation dataset
        hidden_states : list (n_epochs, n_val_datasets, array(n_datapoints, n_hidden_dim))
            list of arrays per validation dataset per epoch
        """
        trainloaders = []
        for dataset in training_datasets:
            trainloaders.append(
                torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True
                )
            )

        train_losses = np.zeros(n_epochs)
        val_losses = np.zeros([n_epochs, len(validation_datasets)])
        hidden_states = []
        for epoch in range(1, n_epochs + 1):
            for trainloader in trainloaders:
                train_losses[epoch - 1] = self.train_step(
                    optimizer, criterion, trainloader
                )

            val_losses[epoch - 1, :], hidden = self.validation(
                criterion, validation_datasets, track=True
            )
            hidden_states.append(hidden)

            if epoch % 10 == 0:
                print("Epoch: {}/{}.............".format(epoch, n_epochs), end=" ")
                print("Loss: {:.5f}".format(train_losses[epoch - 1].item()), end=" ")
                print("Validation Loss: {:.5f}".format(val_losses[epoch - 1, 0].item()))

        return train_losses, val_losses, hidden_states
