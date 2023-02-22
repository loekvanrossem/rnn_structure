import traceback
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer

import numpy as np
import pandas as pd

from tqdm import trange

from model import Model


class Tracker(ABC):
    """
    Store some data of the model each epoch.

    Attributes
    ----------
    model : Model
        The model of which layers are to be tracked.

    Methods
    -------
    track(datasets):
        Store the current activitities. Call this every epoch.
    get_trace() -> pd.DataFrame
        Return the stored data.
    reset():
        Delete stored data
    """

    def __init__(self):
        self._trace = []

    @abstractmethod
    def track(self, *args) -> None:
        """Store the data of this epoch. Should be called each epoch."""
        ...

    def get_trace(self) -> pd.DataFrame:
        """
        Return the stored activities.

        Returns
        -------
        trace : Dataframe (n_epochs, ...)
            dataframe containing the tracked quantity for each epoch.
        """
        index_names = self._trace[0].index.names
        trace = pd.concat(self._trace, keys=list(range(len(self._trace))))
        trace.index = trace.index.set_names(["Epoch"] + index_names)
        return trace

    def reset(self) -> None:
        """Delete stored data"""
        self._trace = []


class ScalarTracker(Tracker):
    """Stores a scalar quantity."""

    def __init__(self, track_function: Callable):
        self.track_function = track_function
        super().__init__()

    def track(self, *args) -> None:
        data = self.track_function()
        self._trace.append(data)


# class LossTracker(Tracker):
#     """Stores average loss of datasets."""

#     def __init__(self, model: Model, criterion):
#         self.model = model
#         self.criterion = criterion
#         super().__init__()

#     def track(self, datasets: list[TensorDataset]) -> None:
#         """Store the data of this epoch. Should be called each epoch."""
#         loss = pd.DataFrame()
#         for i, dataset in enumerate(datasets):
#             dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
#             for batch in dataloader:
#                 inputs, outputs = batch
#                 loss_this_dataset = self.criterion(
#                     torch.squeeze(self.model(inputs)[0]), torch.squeeze(outputs)
#                 )
#                 loss_this_dataset = (
#                     torch.squeeze(loss_this_dataset).cpu().detach().numpy()
#                 )
#                 loss_this_dataset = pd.DataFrame([loss_this_dataset], [i])
#                 loss = pd.concat([loss, loss_this_dataset])

#         loss.index = loss.index.set_names(["Dataset"])
#         self._trace.append(loss)


class ActivationTracker(Tracker):
    """Stores the activations of a layer in response to datasets."""

    def __init__(self, model: Model, track_function: Callable[[Tensor], Tensor]):
        self.model = model
        self.track_function = track_function
        super().__init__()

    def track(self, datasets: list[TensorDataset]) -> None:
        """Store the data of this epoch. Should be called each epoch."""
        act_this_epoch = []
        for dataset in datasets:
            dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            for batch in dataloader:
                inputs, outputs = batch

                # Get labels
                labels = []
                for input in inputs:
                    try:
                        decoding = self.model.encoding.decode(input.cpu())
                        label = "".join(str(int(char)) for char in decoding)
                    except KeyError:
                        label = tuple(np.squeeze(input.cpu()).numpy())
                    labels.append(label)

                # Store activities
                act_this_dataset = self.track_function(inputs)
                act_this_dataset = (
                    torch.squeeze(act_this_dataset).cpu().detach().numpy()
                )
                act_this_dataset = pd.DataFrame(act_this_dataset, labels)
                act_this_epoch.append(act_this_dataset)

        act_this_epoch = pd.concat(act_this_epoch, keys=list(range(len(datasets))))
        act_this_epoch.index = act_this_epoch.index.set_names(["Dataset", "Input"])
        self._trace.append(act_this_epoch)


class Compiler:
    """
    Responsible for training models.

    Attributes
    ----------
    model : Model
        The model that will be trained.
    criterion
        The training criterion.
    optimizer : Optimizer
        The optimizer used in training.validation
    trackers : dict[str, Tracker]
        Trackers that are used during training
    """

    def __init__(
        self,
        model: Model,
        criterion,
        optimizer: Optimizer,
        trackers: Optional[dict[str, Tracker]] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        if trackers is None:
            self.trackers = {
                # "loss": ScalarTracker(lambda: self.validation(tracked_datasets))
            }
        else:
            self.trackers = trackers

    # def validation(self, validation_datasets) -> np.ndarray:
    #     """
    #     Compute loss of validation datasets

    #     Parameters
    #     ----------
    #     validation_datasets : list of datasets

    #     Returns
    #     -------
    #     losses : np.ndarray(n)
    #     """
    #     self.model.eval()
    #     losses = np.zeros(len(validation_datasets))
    #     for i, dataset in enumerate(validation_datasets):
    #         # Compute loss
    #         valloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    #         for batch in valloader:
    #             inputs, outputs = batch
    #             prediction, _ = self.model(inputs)
    #             losses[i] = self.criterion(
    #                 torch.squeeze(prediction), torch.squeeze(outputs)
    #             )

    #     return losses

    def validation(self, datasets) -> pd.DataFrame:
        loss = pd.DataFrame()
        for i, dataset in enumerate(datasets):
            dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            for batch in dataloader:
                inputs, outputs = batch
                loss_this_dataset = self.criterion(
                    torch.squeeze(self.model(inputs)[0]), torch.squeeze(outputs)
                )
                loss_this_dataset = (
                    torch.squeeze(loss_this_dataset).cpu().detach().numpy()
                )
                loss_this_dataset = pd.DataFrame([loss_this_dataset], [i])
                loss = pd.concat([loss, loss_this_dataset])

        loss.index = loss.index.set_names(["Dataset"])
        return loss

    # def grid_values(self, grid_dataset):
    #     gridloader = DataLoader(
    #         grid_dataset, batch_size=len(grid_dataset), shuffle=False
    #     )
    #     for batch in gridloader:
    #         grid_points, _ = batch
    #         grid_points = torch.swapaxes(grid_points, 0, 1)
    #         hidden_values = {}
    #         for symbol in self.model.encoding.symbols:
    #             input = self.model.encoding([[symbol]] * grid_points.shape[1])
    #             input = torch.from_numpy(input.astype(np.float32)).to(self.model.device)
    #             hidden_values[symbol], _ = self.model.rnn(input, grid_points)
    #         output_values = self.model.fc(grid_points)

    #     return hidden_values, output_values

    def training_run(
        self,
        training_datasets: list[TensorDataset],
        tracked_datasets: list[TensorDataset],
        n_epochs=100,
        batch_size=32,
    ):
        """
        Train the network on training datasets

        Parameters
        ----------
        training_datasets: list[datasets]
        tracked_datasets: list[datasets]
        n_epochs: int, default 100
        batch_size: int default 32
            batch size during training
        """
        # Generate trainloaders
        trainloaders = []
        for dataset in training_datasets:
            trainloaders.append(
                DataLoader(dataset, batch_size=batch_size, shuffle=True)
            )

        # Train
        # train_losses = np.zeros(n_epochs)
        # val_losses = np.zeros([n_epochs, len(tracked_datasets)])

        try:
            iterator = trange(n_epochs, desc="Training", unit="steps")
            for epoch in iterator:

                # Store intermediate states
                for tracker in self.trackers.values():
                    tracker.track(tracked_datasets)

                # Training step
                for trainloader in trainloaders:
                    train_loss = self.model.train_step(
                        self.optimizer, self.criterion, trainloader
                    )
                val_loss = (
                    self.trackers["loss"]
                    .get_trace()
                    .query("Dataset==0")
                    .to_numpy()[-1, 0]
                )

                iterator.set_postfix(
                    train_loss="{:.5f}".format(train_loss),
                    val_loss="{:.5f}".format(val_loss),
                )
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            # return train_losses, val_losses
            return
