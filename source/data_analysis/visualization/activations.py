from typing import Optional, Callable

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import plotly.express as px
import matplotlib.patheffects as pe
from matplotlib import axes
import matplotlib.style as mplstyle

from data_analysis.visualization.basic_plotting import axes_scale
from data_analysis.visualization import animation
import utils.dataframes as dataframes

mplstyle.use("fast")

COLORS = px.colors.qualitative.Plotly


## Add static points
class PointAnimation(animation.AnimationSubPlot):
    """
    Plot 2D-points varying per epoch

    Attributes
    ----------
    points : np.ndarray[Epoch, Point, 2]
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
        colors: Optional[list[int]] = None,
        plot_trails: bool = True,
    ):
        self.points = points
        self.labels = labels
        self.plot_trails = plot_trails
        if colors is None:
            self.colors = [0] * points.shape[1]
        else:
            self.colors = colors

    def plot(self, ax: axes.Axes):
        self.ax = ax
        if self.labels is None:
            self.labels = [None] * len(self.points[0])

        graphics = []
        for label, point, color in zip(
            self.labels, np.swapaxes(self.points, 0, 1), self.colors
        ):
            x_values, y_values = point[:, 0], point[:, 1]
            x_value, y_value = x_values[0], y_values[0]

            # Draw point
            point_graphic = self.ax.plot(
                x_value,
                y_value,
                "o",
                c=COLORS[color],
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
        scale_x, scale_y = axes_scale(self.ax)
        for point_graphic, point, label in self._points:
            x = float(point[epoch, 0])
            y = float(point[epoch, 1])

            # Position point
            point_graphic[0].set_xdata(x)
            point_graphic[0].set_ydata(y)

            # Position label
            if label is not None:
                pos = [x + 0.06 * scale_x, y + 0.02 * scale_y]
                if len(labels_x_pos) > 0:
                    while (
                        self._smallest_dist(pos, labels_x_pos, labels_y_pos)
                        < 0.15 * scale_y
                    ):
                        if pos[1] > self.ax.get_ylim()[1] - 0.8 * scale_y:
                            break
                        pos[1] += 0.4 * scale_y

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

    plot_labels : Bool, default True
        If true plot the input names of each datapoint
    plot_trails : bool, optional, default True
        Plot the trajectory for each point
    """

    def __init__(
        self,
        activations: pd.DataFrame,
        transform: str,
        fixed_points: Optional[dict] = None,
        colors: Optional[list[int]] = None,
        plot_labels: bool = True,
        plot_trails: bool = True,
    ):
        data = activations.copy()
        data.columns = data.columns.astype(str)
        self.epochs = list(set(data.index.get_level_values("Epoch")))
        self.epochs.sort()

        # Add outputs
        if fixed_points is not None:
            epochs = set(data.index.get_level_values("Epoch"))
            for label, point in fixed_points.items():
                for epoch in epochs:
                    data.loc[epoch, -1, f"{label}"] = np.array(point)

        index = data.index
        if data.shape[1] == 1:
            data["Null"] = pd.Series(0, index=index)

        # Apply dimensionality reduction
        match transform:
            case "none":
                data_red = data.iloc[:, 0:2]
            case "MDS":
                mds = MDS(n_components=2, normalized_stress="auto")
                data_red = mds.fit_transform(data)
            case "PCA":
                pca = PCA(n_components=2)
                data_red = pca.fit_transform(data)
            case "PCA_per_epoch":
                data_red = data.copy().iloc[:, 0:2]
                for epoch in data.groupby("Epoch"):
                    pca = PCA(n_components=2)
                    data_red.loc[epoch[0]] = pca.fit_transform(epoch[1])
            case "MDS_per_epoch":
                data_red = data.copy().iloc[:, 0:2]
                for epoch in data.groupby("Epoch"):
                    mds = MDS(n_components=2, normalized_stress="auto")
                    data_red.loc[epoch[0]] = mds.fit_transform(epoch[1])
            case _:
                raise ValueError("Invalid transform")

        data_red = pd.DataFrame(data_red, index=index)
        datasets = dataframes.to_labels(
            data_red.query("Epoch == 0").reset_index(level=(0, 2), drop=True)
        )[0]
        data_red = data_red.reset_index(level=1, drop=True)
        points = dataframes.to_ndarray(data_red)
        data_red = data_red.loc[0]
        if plot_labels:
            labels = dataframes.to_labels(data_red)[0]
        else:
            labels = None
        if colors is None:
            colors = [int(c) for c in datasets]

        super().__init__(points, labels, colors, plot_trails)

    def update(self, parameter: int):
        super().update(self.epochs.index(parameter))


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

        points = train_points
        self.y_bounds = y_bounds

        if true_function:
            true_points_train = self._true_points(train_points, true_function)
            true_points_test = self._true_points(test_points, true_function)

            points = np.concatenate((points, true_points_train), axis=1)
            self.line_data.append(true_points_test)

        self.line_data.append(test_points)

        super().__init__(points, None, None, False)

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
        y = dataframes.to_ndarray(outputs)[:, :, 0]
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
