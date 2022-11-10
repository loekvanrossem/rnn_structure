import torch
from torch import nn

import numpy as np
import pandas as pd

from tqdm import trange


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
        """
        self.eval()
        losses = np.zeros(len(validation_datasets))
        hidden_states, output_values = [], []
        for i, dataset in enumerate(validation_datasets):
            # Compute loss
            valloader = torch.utils.data.DataLoader(
                dataset, batch_size=len(dataset), shuffle=False
            )
            for batch in valloader:
                inputs, outputs = batch
                prediction, hidden = self(inputs)
                losses[i] = criterion(torch.squeeze(prediction), torch.squeeze(outputs))

                # Track hidden states
                if track:
                    labels = []
                    for input in inputs:
                        label = "".join(str(int(char[0])) for char in input)
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
        hidden_states : Dataframe (n_epochs, n_val_datasets, n_inputs)
            Dataframe containing the hidden dimension values for each input and epoch
            as well as the predicted outputs
        """
        # Generate trainloaders
        trainloaders = []
        for dataset in training_datasets:
            trainloaders.append(
                torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True
                )
            )

        # Train
        train_losses = np.zeros(n_epochs)
        val_losses = np.zeros([n_epochs, len(validation_datasets)])
        hidden_states, output_values = [], []

        try:
            with trange(n_epochs, desc="Training", unit="steps") as iterator:
                for epoch in iterator:

                    for trainloader in trainloaders:
                        train_losses[epoch] = self.train_step(
                            optimizer, criterion, trainloader
                        )

                    val_losses[epoch, :], hidden, output = self.validation(
                        criterion, validation_datasets, track=True
                    )
                    hidden = pd.concat(hidden, keys=np.arange(len(validation_datasets)))
                    output = pd.concat(output, keys=np.arange(len(validation_datasets)))
                    hidden_states.append(hidden)
                    output_values.append(output)

                    iterator.set_postfix(
                        train_loss="{:.5f}".format(train_losses[epoch].item()),
                        val_loss="{:.5f}".format(val_losses[epoch, 0].item()),
                    )
        except Exception as e:
            print(e)
            raise e
        finally:
            # Make dataframe with hidden states and outputs
            hidden_states = pd.concat(hidden_states, keys=np.arange(n_epochs))
            hidden_states.index = hidden_states.index.set_names(
                ["Epoch", "Dataset", "Input"]
            )
            output_values = pd.concat(output_values, keys=np.arange(n_epochs))
            output_values.index = output_values.index.set_names(
                ["Epoch", "Dataset", "Input"]
            )

            return train_losses, val_losses, hidden_states, output_values
