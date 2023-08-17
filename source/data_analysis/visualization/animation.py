from abc import ABC, abstractmethod


import math
from tqdm import trange, tqdm

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import axes, figure
from ipywidgets import Layout, interact, IntSlider, SelectionSlider
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
    def plot(self, ax: axes.Axes) -> None:
        """Initial plot."""

    @abstractmethod
    def update(self, parameter: int) -> None:
        """Update this plot given a certain parameter value."""


class SliderAnimation:
    """
    A slider controlled animation consisting of some subplots.

    Attributes
    ----------
    plots : dict[str, AnimationSubPlot]
        The subplots, indexed by their titles
    parameters : list
        The possible parameters for the slider
    parameter_name : str
        Name of the parameter
    fig_size : float, optional, default 5
        The size of the figure

    Methods
    -------
    to_gif(path : str)
        Make a gif
    """

    def __init__(
        self,
        plots: dict[str, AnimationSubPlot],
        parameters: list[int],
        parameter_name: str,
        fig_size: float = 5,
    ) -> None:
        self.plots = plots
        self.parameters = parameters
        self.parameter_name = parameter_name
        self.fig_size = fig_size
        self._buffer = {}
        self._start()

    def _start(self) -> None:
        self._fig = self._plot()

        slider = SelectionSlider(
            options=self.parameters,
            value=self.parameters[0],
            description=f"{self.parameter_name}:",
        )

        @interact(parameter=slider)
        def slider_parameter(parameter):
            return self._get_frame(parameter)

    def _get_frame(self, parameter) -> Image.Image:
        fig = self._fig
        try:
            return self._buffer[parameter]
        except KeyError:
            for plot in self.plots.values():
                plot.update(parameter)
            fig.canvas.draw()
            image = Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            self._buffer[parameter] = image
            return image

    def _plot(self) -> figure.Figure:
        n_plots = len(self.plots)
        n_columns = 2
        n_rows = math.ceil(n_plots / n_columns)
        fig = plt.figure(figsize=(n_columns * self.fig_size, n_rows * self.fig_size))
        fig.subplots_adjust(
            left=0.1, right=0.9, top=0.94, bottom=0.06, wspace=0.1, hspace=0.2
        )
        for n, (title, plot) in enumerate(self.plots.items()):
            ax = fig.add_subplot(n_rows * 100 + n_columns * 10 + (n + 1))
            plot.plot(ax)
            ax.set_title(title)
        return fig

    def to_gif(self, path: str, frame_duration: int = 70) -> None:
        """
        Make a gif.

        Parameters
        ----------
        path : str
            Save the gif here
        frame_duration : int, optional, default 70
            The number of milliseconds each frame is displayed
        """
        frames = []
        iterator = tqdm(
            self.parameters,
            desc="Making gif",
            unit="frames",
        )
        for parameter in iterator:
            frames.append(self._get_frame(parameter))

        if not path.endswith(".gif"):
            path += ".gif"
        gif.save(frames, path, duration=frame_duration)
