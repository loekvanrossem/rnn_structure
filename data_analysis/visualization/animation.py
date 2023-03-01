from abc import ABC, abstractmethod
from abc import ABC, abstractmethod

from tqdm import trange

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import axes
from ipywidgets import Layout, interact, IntSlider
from IPython.display import display
import gif


class SliderSubPlot(ABC):
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
        ...

    @abstractmethod
    def update(self, parameter: int) -> None:
        """Update this plot given a certain parameter value."""
        ...


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
        self, plots: list[SliderSubPlot], parameters: list[int], fig_size: float = 5
    ) -> None:
        self.plots = plots
        self.parameters = parameters
        self.fig_size = fig_size
        self._start()

    def _start(self) -> None:

        fig = plt.figure(figsize=(2 * self.fig_size, self.fig_size))
        for n, plot in enumerate(self.plots):
            axis = fig.add_subplot(120 + (n + 1))
            plot.plot(axis)
        slider = IntSlider(
            description="Epoch:",
            value=self.parameters[0],
            min=self.parameters[0],
            max=self.parameters[-1],
            step=1,
            layout=Layout(width=f"{int(self.fig_size*7)}%"),
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
