from typing import Optional

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.axes as axes

from data_analysis.visualization import animation
from data_analysis.visualization.publication import pub_show


class EpochAnimation(animation.AnimationSubPlot):
    """
    An animation that plots the data for each epoch.

    Attributes
    ----------
    graphs: dict[str, np.ndarray]
        Data per epoch to be plotted.
    unitless_graphs: dict[str, np.ndarray]
        Data per epoch to be plotted invariant of bounds.
    x_bounds: Optional[tuple[float, float]]
    y_bounds: Optional[tuple[float, float]]
    """

    def __init__(
        self,
        graphs: dict[str, np.ndarray],
        unitless_graphs: Optional[dict[str, np.ndarray]] = None,
        x_bounds: Optional[tuple[float, float]] = None,
        y_bounds: Optional[tuple[float, float]] = None,
    ):
        self.graphs = graphs
        self.unitless_graphs = unitless_graphs
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def plot(self, ax: axes.Axes):
        for name, data in self.graphs.items():
            line = ax.plot(data, label=name, zorder=1)
            x_axis = line[0].get_xdata()
            ax.fill_between(x_axis, data.squeeze(), alpha=0.2, zorder=0)

        if self.x_bounds:
            plt.xlim(self.x_bounds[0], self.x_bounds[1])
        if self.y_bounds:
            plt.ylim(self.y_bounds[0], self.y_bounds[1])

        if self.unitless_graphs:
            for name, data in self.unitless_graphs.items():
                data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
                data_unitless = (
                    data_normalized * (ax.get_ylim()[1] + ax.get_ylim()[0])
                    - ax.get_ylim()[0]
                ) * 0.5
                ax.plot(data_unitless, label=name, zorder=2)

        self._vline = ax.axvline(x=0, color="red", linestyle="--")
        self._vline.set_visible(False)

        plt.legend(loc="upper left")
        plt.xlabel("Epoch")

        plt.show()

    def update(self, epoch: int):
        self._vline.set_visible(True)
        self._vline.set_xdata(epoch)
