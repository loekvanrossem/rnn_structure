from abc import ABC, abstractmethod
from typing import Callable, Optional
from xmlrpc.client import Boolean

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Optimizer

import numpy as np
import pandas as pd

from tqdm import trange

from activations import get_activations
from preprocessing import Encoding


class Tracker(ABC):
    """
    Store some data of the model each epoch.

    Methods
    -------
    track(datasets):
        Store the current activitities. Call this every epoch.
    get_trace() -> pd.DataFrame
        Return the stored data.
    get_entry(epoch: int):
        Return data at a specific epoch.
    reset():
        Delete stored data
    """

    def __init__(self):
        self._trace = []

    @abstractmethod
    def track(self, *args) -> None:
        """Store the data of this epoch. Should be called each epoch."""

    def get_trace(self) -> pd.DataFrame:
        """
        Return all stored data.

        Returns
        -------
        trace : Dataframe (n_epochs, ...)
            dataframe containing the tracked quantity for each epoch.
        """
        if not self._trace:
            raise ValueError("No data stored.")

        index_names = self._trace[0].index.names
        trace = pd.concat(self._trace, keys=list(range(len(self._trace))))
        trace.index = trace.index.set_names(["Epoch"] + index_names)
        return trace

    def get_entry(self, epoch):
        """Return data at specific epoch."""
        return self._trace[epoch]

    def reset(self) -> None:
        """Delete stored data"""
        self._trace.clear()


class ScalarTracker(Tracker):
    """
    Stores a scalar quantity.

    Attributes
    ----------
    track_function : Callable
        Function computing the scalar quantity each epoch.
    """

    def __init__(self, track_function: Callable) -> None:
        self.track_function = track_function
        super().__init__()

    def track(self, *args) -> None:
        data = self.track_function()
        self._trace.append(data)


class ActivationTracker(Tracker):
    """
    Stores the activations of a layer in response to datasets.

    Attributes
    ----------
    encoding : Encoding
    track function : function(Tensor) -> Tensor
        The activations to be tracked as a function of the inputs
    datasets : list[Dataset]
        The datasets of which the activations to track
    initial : function() -> Tensor, default None
        If provided, also track the initial hidden activations
    """

    def __init__(
        self,
        encoding: Encoding,
        track_function: Callable[[Tensor], Tensor],
        datasets: list[Dataset],
        initial: Optional[Callable[[], Tensor]] = None,
    ) -> None:
        self.encoding = encoding
        self.track_function = track_function
        self.datasets = datasets
        self.initial = initial
        super().__init__()

    def track(self) -> None:
        """Store the data of this epoch. Should be called each epoch."""
        act_this_epoch = get_activations(
            self.datasets,
            self.track_function,
            self.encoding,
        )

        if self.initial is not None:
            initial_hidden = pd.DataFrame(
                self.initial().cpu().detach().numpy(),
                index=[np.array([-1]), np.array(["initial"])],
            )
            initial_hidden.index = initial_hidden.index.set_names(["Dataset", "Input"])

            act_this_epoch = pd.concat([act_this_epoch, initial_hidden])

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
        model: nn.Module,
        criterion,
        optimizer: Optimizer,
        trackers: Optional[dict[str, Tracker]] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.trackers = trackers or {}

    def validation(self, datasets: list[TensorDataset]) -> pd.DataFrame:
        """
        Compute validation loss on the given datasets.

        Parameters
        ----------
            datasets (List[TensorDataset]): List of datasets for validation.

        Returns:
            pd.DataFrame: Validation loss for each dataset.s
        """
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

    def training_run(
        self,
        training_datasets: list[TensorDataset],
        n_epochs: int,
        batch_size: int,
        progress_bar: bool = True,
        conv_thresh: float = 0,
    ):
        """
        Train the network on training datasets

        Parameters
        ----------
        training_datasets: list[datasets]
        n_epochs: int
        batch_size: int
            batch size during training
        progress_bar : bool, optional
            Whether to show a progress bar
        conv_thresh: float, optional
            if provided stop training when loss is below this value
        """
        # Generate trainloaders
        trainloaders = [
            DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for dataset in training_datasets
        ]
        n_train_data = sum(len(dataset) for dataset in training_datasets)

        # Train
        iterator = trange(
            n_epochs, desc="Training", unit="steps", disable=not progress_bar
        )
        for epoch in iterator:
            # Store intermediate states
            self.model.eval()
            for tracker in self.trackers.values():
                tracker.track()

            # Training step
            train_loss = sum(
                self.model.train_step(self.optimizer, self.criterion, trainloader)
                * (len(dataset) / n_train_data)
                for trainloader, dataset in zip(trainloaders, training_datasets)
            )

            val_loss = (
                self.trackers.get("loss")
                .get_entry(-1)
                .query("Dataset==0")
                .to_numpy()[0, 0]
                if "loss" in self.trackers
                else np.NaN
            )

            iterator.set_postfix(
                train_loss="{:.5f}".format(train_loss),
                val_loss="{:.5f}".format(val_loss),
            )

            if train_loss < conv_thresh:
                return
