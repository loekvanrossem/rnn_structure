from typing import Optional, Callable
import warnings

import numpy as np
import pandas as pd
from pyparsing import line
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import matplotlib.patheffects as pe
from matplotlib import axes
import matplotlib.style as mplstyle

from preprocessing import Encoding
from data_analysis.visualization.basic_plotting import axes_scale
from data_analysis.visualization import animation
from utils.dataframes import to_ndarray

mplstyle.use("fast")


## Add static points
class PointAnimation(animation.AnimationSubPlot):
    """
    Plot 2D-points varying per epoch

    Attributes
    ----------
    activations : np.ndarray[Epoch, Point, 2]
        Dataframe containing the point values for each epoch
    labels : np.ndarray, optional
        If provided, label the points
    plot_trails : bool, optional, default True
        Plot the trajectory for each point
    """

    def __init__(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        plot_trails: bool = True,
    ):
        self.points = points
        self.labels = labels
        self.plot_trails = plot_trails

    def plot(self, ax: axes.Axes):
        self.ax = ax
        if self.labels is None:
            self.labels = [None] * len(self.points[0])

        graphics = []
        for label, point in zip(self.labels, np.swapaxes(self.points, 0, 1)):
            x_values, y_values = point[:, 0], point[:, 1]
            x_value, y_value = x_values[0], y_values[0]

            # Draw point
            point_graphic = self.ax.plot(
                x_value,
                y_value,
                "o",
                # c=color,
                zorder=3,
                markeredgecolor="black",
            )
            color = point_graphic[0].get_color()

            # Draw label
            if isinstance(label, str):
                label = self.ax.text(
                    x_value,
                    y_value,
                    label,
                    path_effects=[
                        pe.Stroke(linewidth=2, foreground="w"),
                        pe.Normal(),
                    ],
                    zorder=4,
                )
            else:
                label = None
            graphics.append((point_graphic, point, label))

            # Draw line
            if self.plot_trails:
                line = self.ax.plot(x_values, y_values, c=color, label=label, zorder=0)

        self._points = graphics

    def update(self, parameter: int):
        epoch = parameter
        labels_x_pos = []
        labels_y_pos = []
        scale = axes_scale(self.ax)
        for point_graphic, point, label in self._points:
            x = float(point[epoch, 0])
            y = float(point[epoch, 1])

            # Position point
            point_graphic[0].set_xdata(x)
            point_graphic[0].set_ydata(y)

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


class ActivationsAnimation(PointAnimation):
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
    plot_trails : bool, optional, default True
        Plot the trajectory for each point
    """

    def __init__(
        self,
        activations: pd.DataFrame,
        transform: str,
        n_labels: int = 10,
        encoding: Optional[Encoding] = None,
        plot_labels: bool = True,
        plot_trails: bool = True,
    ):
        labels = activations.loc[0].index.get_level_values("Input")
        index = activations.index
        data = activations
        if data.shape[1] == 1:
            data["Null"] = pd.Series(0, index=index)

        # Apply dimensionality reduction
        match transform:
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
        data_red = data_red.reset_index(level=1, drop=True)
        data_red = to_ndarray(data_red)

        # Add outputs
        if encoding is not None:
            if transform != "none":
                warnings.warn(
                    "Plotting output decoding does not work for transformations"
                )
            for symbol in encoding.symbols:
                x, y = encoding(symbol)[0], encoding(symbol)[1]
                label = f"Output {symbol}"

                data_red = np.append(data_red, [[[x, y]]] * data_red.shape[0], axis=1)
                labels = np.append(labels, [label])

        self.points = data_red
        if plot_labels:
            self.labels = labels
        else:
            self.labels = None
        self.plot_trails = plot_trails


class FunctionAnimation(PointAnimation):
    """
    Plot output function varying per epoch.

    Attributes
    ----------
    outputs : Dataframe (Epoch, Dataset, Input)
        Dataframe containing the models outputs for each input
    true_function : function
        The function providing correct outputs
    y_bounds: Optional[tuple[float, float]]
    """

    def __init__(
        self,
        outputs_train: pd.DataFrame,
        outputs_test: pd.DataFrame,
        true_function: Optional[Callable] = None,
        y_bounds: Optional[tuple[float, float]] = None,
    ):
        train_points = self._to_xy(outputs_train)
        test_points = self._to_xy(outputs_test)

        self.line_data = []

        self.points = train_points
        self.labels = None
        self.plot_trails = False
        self.y_bounds = y_bounds

        if true_function:
            true_points_train = self._true_points(train_points, true_function)
            true_points_test = self._true_points(test_points, true_function)

            self.points = np.concatenate((self.points, true_points_train), axis=1)
            self.line_data.append(true_points_test)

        self.line_data.append(test_points)

    def plot(self, ax: axes.Axes):
        # Draw lines
        self._lines = []
        for data in self.line_data:
            x_values = data[0, :, 0]
            y_values = data[0, :, 1]
            line_graphic = ax.plot(
                x_values,
                y_values,
                zorder=3,
                markeredgecolor="black",
            )
            self._lines.append((line_graphic, data))

        super().plot(ax)

        if self.y_bounds:
            ax.set_ylim(self.y_bounds[0], self.y_bounds[1])

    def update(self, parameter: int):
        # Update line
        for line_graphic, outputs in self._lines:
            for point in line_graphic:
                x_values = outputs[parameter, :, 0]
                y_values = outputs[parameter, :, 1]
                point.set_xdata(x_values)
                point.set_ydata(y_values)

        super().update(parameter)

    def _to_xy(self, outputs: pd.DataFrame):
        outputs = outputs.reset_index(level=1, drop=True)
        y = to_ndarray(outputs)[:, :, 0]
        n_epochs = y.shape[0]
        x = outputs.index.unique(level="Input").to_numpy()
        x = np.array(list(map(float, x)))

        points = np.dstack((np.repeat([x], n_epochs, axis=0), y))

        return points

    def _true_points(self, points: np.ndarray, true_function: Callable):
        x = points[0, :, 0]
        y_true = list(map(true_function, x))
        true_points = np.dstack((x, y_true))
        true_points = np.repeat(true_points, points.shape[0], axis=0)
        return true_points
