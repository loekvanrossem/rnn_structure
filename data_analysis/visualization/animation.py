from abc import ABC, abstractmethod

import math
from tqdm import trange

import matplotlib.pyplot as plt
from matplotlib import axes
from ipywidgets import Layout, interact, IntSlider
from IPython.display import display
import gif


class AnimationSubPlot(ABC):
    """
    A plot changing depending on some parameter.

    Methods
    -------
    plot(axes : Axes):
        Initial plot.
    update(parameter : int):
        Update this plot given a certain parameter value.
    """

    @abstractmethod
    def plot(self, axes: axes.Axes) -> None:
        """Initial plot."""

    @abstractmethod
    def update(self, parameter: int) -> None:
        """Update this plot given a certain parameter value."""


class SliderAnimation:
    """
    A slider controlled animation consisting of some subplots.

    Attributes
    ----------
    plots : list[SliderSubPlot]
        The subplots
    parameters : list
        The possible parameters for the slider
    fig_size : float, optional, default 5
        The size of the figure

    Methods
    -------
    to_gif(path : str)
        Make a gif
    """

    def __init__(
        self, plots: list[AnimationSubPlot], parameters: list[int], fig_size: float = 5
    ) -> None:
        self.plots = plots
        self.parameters = parameters
        self.fig_size = fig_size
        self._start()

    def _start(self) -> None:
        n_plots = len(self.plots)
        n_columns = 2
        n_rows = math.ceil(n_plots / n_columns)
        fig = plt.figure(figsize=(n_columns * self.fig_size, n_rows * self.fig_size))
        for n, plot in enumerate(self.plots):
            axes = fig.add_subplot(n_rows * 100 + n_columns * 10 + (n + 1))
            plot.plot(axes)
        slider = IntSlider(
            description="Epoch:",
            value=self.parameters[0],
            min=self.parameters[0],
            max=self.parameters[-1],
            step=1,
            layout=Layout(width=f"{int(self.fig_size*8)}%"),
        )

        @interact(parameter=slider)
        def slider_epochs(parameter):
            for plot in self.plots:
                plot.update(parameter)
            display(fig)

    def to_gif(self, path: str) -> None:
        """
        Make a gif.

        Parameters
        ----------
        path : str
            Save the gif here
        """
        step_size = 20

        @gif.frame
        def frame(epoch):
            fig, points_1, points_2, ax_1, ax_2 = self._initial_plot()
            self._update(points_1, points_2, ax_1, ax_2, epoch)

        frames = []
        iterator = trange(
            self.parameters[0],
            self.parameters[-1],
            step_size,
            desc="Making gif",
            unit="frames",
        )
        for parameter in iterator:
            frames.append(frame(parameter))

        gif.save(frames, path + ".gif", duration=50)
