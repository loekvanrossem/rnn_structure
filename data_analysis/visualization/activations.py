from typing import Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import matplotlib.patheffects as pe
from matplotlib import axes
import matplotlib.style as mplstyle

from preprocessing import Encoding
from data_analysis.visualization.basic_plotting import axes_scale
from data_analysis.visualization import animation

mplstyle.use("fast")


class ActivationsAnimation(animation.AnimationSubPlot):
    """
    Plot activations varying per epoch

    Attributes
    ----------
    activations : Dataframe (Epoch, Dataset, Input)
        Dataframe containing the activations values for each input and epoch
    transform : str
        The type of dimensionality reduction to use: none, PCA or MDS
    n_labels : int, default 10
        The number of epochs which will be labeled
    fig_size : float, default 5
        The size of the figure
    encoding : Encoding, default None
        If provided, also plot the encoded symbol values
    plot_labels : Bool, default True
        If true plot the input names of each datapoint
    """

    def __init__(
        self,
        activations: pd.DataFrame,
        transform: str,
        n_labels: int = 10,
        encoding: Optional[Encoding] = None,
        plot_labels: bool = True,
    ):
        self.activations = activations
        self.transform = transform
        self.n_labels = n_labels
        self.encoding = encoding
        self.plot_labels = plot_labels

    def plot(self, ax: axes.Axes):
        self.ax = ax
        scale = axes_scale(self.ax)

        index = self.activations.index
        data = self.activations
        if data.shape[1] == 1:
            data["Null"] = pd.Series(0, index=index)

        # Apply dimensionality reduction
        match self.transform:
            case "none":
                data_red = data.iloc[:, 0:2]
            case "MDS":
                mds = MDS(n_components=2)
                data_red = mds.fit_transform(data)
            case "PCA":
                pca = PCA(n_components=2)
                data_red = pca.fit_transform(data)
            case _:
                raise ValueError("Invalid transform")
        data_red = pd.DataFrame(data_red, index=index)

        points = []
        for dataset_name, dataset in data_red.groupby("Dataset"):
            for input_name, input in dataset.groupby("Input"):
                x_values = input.iloc[:, 0]
                y_values = input.iloc[:, 1]
                line = self.ax.plot(x_values, y_values, label=input_name, zorder=0)
                color = line[0].get_color()
                first_epoch = data.index.get_level_values("Epoch").min()
                last_epoch = data.index.get_level_values("Epoch").max()
                # Label some points
                log_range = np.logspace(
                    np.log(first_epoch + 0.999),
                    np.log(last_epoch + 1),
                    num=self.n_labels,
                    base=np.e,
                ).astype(int)
                for epoch in log_range:
                    self.ax.scatter(
                        x_values[epoch], y_values[epoch], s=10, c="Black", zorder=1
                    )
                    self.ax.annotate(
                        epoch, (x_values[epoch], y_values[epoch]), zorder=2
                    )
                # Add time moveable points
                point = self.ax.plot(
                    x_values[last_epoch],
                    y_values[last_epoch],
                    "o",
                    c=color,
                    zorder=3,
                    markeredgecolor="black",
                )
                if self.plot_labels and isinstance(input_name, str):
                    label = self.ax.text(
                        x_values[last_epoch],
                        y_values[last_epoch],
                        input_name,
                        path_effects=[
                            pe.Stroke(linewidth=2, foreground="w"),
                            pe.Normal(),
                        ],
                        zorder=4,
                    )
                else:
                    label = None
                points.append((point, input, label))
        self.points = points

        if self.encoding is not None:
            if self.transform != "none":
                warnings.warn(
                    "Plotting output decoding does not work for transformations"
                )
            for symbol in self.encoding.symbols:
                point = self.ax.plot(
                    self.encoding(symbol)[0],
                    self.encoding(symbol)[1],
                    "o",
                    c="black",
                    zorder=3,
                )
                label = self.ax.text(
                    self.encoding(symbol)[0] + 0.06 * scale,
                    self.encoding(symbol)[1] + 0.06 * scale,
                    f"Output {symbol}",
                    path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
                    zorder=4,
                )

    def update(self, parameter: int):
        epoch = parameter
        labels_x_pos = []
        labels_y_pos = []
        scale = axes_scale(self.ax)
        for point, input, label in self.points:
            x = float(input.iloc[:, 0][epoch])
            y = float(input.iloc[:, 1][epoch])

            # Position point
            point[0].set_xdata(x)
            point[0].set_ydata(y)

            # Position label
            if label is not None:
                pos = [x + 0.06 * scale, y + 0.02 * scale]
                if len(labels_x_pos) > 0:
                    while (
                        self._smallest_dist(pos, labels_x_pos, labels_y_pos)
                        < 0.15 * scale
                    ):
                        if pos[1] > self.ax.get_ylim()[1] - 0.8 * scale:
                            break
                        pos[1] += 0.4 * scale

                label.set_position(pos)
                labels_x_pos.append(pos[0])
                labels_y_pos.append(pos[1])

    def _smallest_dist(self, point, labels_x_pos, labels_y_pos):
        distances = 0.25 * (point[0] - np.array(labels_x_pos)) ** 2 + (
            (point[1] - np.array(labels_y_pos)) ** 2
        )
        return np.sqrt(min(distances))
