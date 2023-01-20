import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from tqdm import trange

import traceback

## TODO: take model as attribute instead of use self
class CompileModel(nn.Module):
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
        hidden_states : list arrays(n_inputs, n_hidden_dim) per dataset
        output_values : list arrays(n_inputs, n_output_dim) per dataset
        """
        self.eval()
        losses = np.zeros(len(validation_datasets))
        hidden_states, output_values = [], []
        for i, dataset in enumerate(validation_datasets):
            # Compute loss
            valloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            for batch in valloader:
                inputs, outputs = batch
                prediction, hidden = self(inputs)
                losses[i] = criterion(torch.squeeze(prediction), torch.squeeze(outputs))

                # Track hidden states
                if track:
                    labels = []
                    for input in inputs:
                        try:
                            decoding = self.encoding.decode(input.cpu())
                            label = "".join(str(int(char)) for char in decoding)
                        except KeyError:
                            label = tuple(np.squeeze(input.cpu()).numpy())
                        labels.append(label)

                    hidden = torch.squeeze(hidden[-1]).cpu().detach().numpy()
                    prediction = torch.squeeze(prediction).cpu().detach().numpy()
                    hidden = pd.DataFrame(hidden, labels)
                    prediction = pd.DataFrame(prediction, labels)
                    hidden_states.append(hidden)
                    output_values.append(prediction)

        if track:
            return losses, hidden_states, output_values
        return losses

    def grid_values(self, grid_dataset):
        gridloader = DataLoader(
            grid_dataset, batch_size=len(grid_dataset), shuffle=False
        )
        for batch in gridloader:
            grid_points, _ = batch
            grid_points = torch.swapaxes(grid_points, 0, 1)
            hidden_values = {}
            for symbol in self.encoding.symbols:
                input = self.encoding([[symbol]] * grid_points.shape[1])
                input = torch.from_numpy(input.astype(np.float32)).to(self.device)
                hidden_values[symbol], _ = self.rnn(input, grid_points)
            output_values = self.fc(grid_points)

        return hidden_values, output_values

    def training_run(
        self,
        optimizer,
        criterion,
        training_datasets,
        validation_datasets,
        grid_dataset=None,
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
        grid_dataset: dataset, default none
        n_epochs: int, default 100
        batch_size: int default 32
            batch size during training

        Returns
        -------
        training_losses: array(n_epochs)
            The losses of the last training dataset
        val_losses: array(n_epochs, len(validation_datasets))
            The losses per validation dataset
        hidden_states : Dataframe (n_epochs, n_val_datasets, n_inputs)
            Dataframe containing the hidden dimension values for each input and epoch
        output_values : Dataframe (n_epochs, n_val_datasets, n_inputs)
            Dataframe containing the predicted output dimension values for each input and epoch
        grid_hiddens : list[ndarray(n_grid_points)]
            The rnn hidden map values each epoch
        grid_outputs : list[ndarray(n_grid_points)]
            The rnn output map values each epoch
        """
        n_val_datasets = len(validation_datasets)
        # Generate trainloaders
        trainloaders = []
        for dataset in training_datasets:
            trainloaders.append(
                DataLoader(dataset, batch_size=batch_size, shuffle=True)
            )

        # Train
        train_losses = np.zeros(n_epochs)
        val_losses = np.zeros([n_epochs, len(validation_datasets)])
        hidden_states, output_values = [], []
        grid_hiddens, grid_outputs = [], []

        try:
            with trange(n_epochs, desc="Training", unit="steps") as iterator:
                for epoch in iterator:
                    # Validate
                    val_losses[epoch, :], hidden, output = self.validation(
                        criterion, validation_datasets, track=True
                    )

                    # Store intermediate states
                    hidden = pd.concat(hidden, keys=list(range(n_val_datasets)))
                    output = pd.concat(output, keys=list(range(n_val_datasets)))
                    hidden_states.append(hidden)
                    output_values.append(output)
                    if grid_dataset is not None:
                        grid_hidden, grid_output = self.grid_values(grid_dataset)
                        grid_hiddens.append(grid_hidden)
                        grid_outputs.append(grid_output)

                    # Training step
                    for trainloader in trainloaders:
                        train_losses[epoch] = self.train_step(
                            optimizer, criterion, trainloader
                        )

                    iterator.set_postfix(
                        train_loss="{:.5f}".format(train_losses[epoch].item()),
                        val_loss="{:.5f}".format(val_losses[epoch, 0].item()),
                    )
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            # Make dataframe with hidden states and outputs
            hidden_states = pd.concat(hidden_states, keys=list(range(n_epochs)))
            hidden_states.index = hidden_states.index.set_names(
                ["Epoch", "Dataset", "Input"]
            )
            output_values = pd.concat(output_values, keys=list(range(n_epochs)))
            output_values.index = output_values.index.set_names(
                ["Epoch", "Dataset", "Input"]
            )

            if grid_dataset is None:
                return train_losses, val_losses, hidden_states, output_values
            else:
                return (
                    train_losses,
                    val_losses,
                    hidden_states,
                    output_values,
                    grid_hiddens,
                    grid_outputs,
                )
