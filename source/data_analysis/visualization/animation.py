from abc import ABC, abstractmethod


import math
from tqdm import trange

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import axes, figure
from ipywidgets import Layout, interact, IntSlider
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
        self._buffer = {}
        self._start()

    def _start(self) -> None:
        self._fig = self._plot()

        slider = IntSlider(
            description="Epoch:",
            value=self.parameters[0],
            min=self.parameters[0],
            max=self.parameters[-1],
            step=1,
            layout=Layout(width=f"{int(15 + self.fig_size*6)}%"),
        )

        @interact(parameter=slider)
        def slider_parameter(parameter):
            return self._get_frame(parameter)

    def _get_frame(self, parameter) -> Image.Image:
        fig = self._fig
        try:
            return self._buffer[parameter]
        except KeyError:
            for plot in self.plots:
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
            left=0.06, right=0.94, top=0.94, bottom=0.06, wspace=0.1, hspace=0.1
        )
        for n, plot in enumerate(self.plots):
            ax = fig.add_subplot(n_rows * 100 + n_columns * 10 + (n + 1))
            plot.plot(ax)
        return fig

    def to_gif(self, path: str, step_size: int = 1, frame_duration: int = 70) -> None:
        """
        Make a gif.

        Parameters
        ----------
        path : str
            Save the gif here
        step_size : int, optional, default 1
            Number of parameter steps inbetween frames
        frame_duration : int, optional, default 70
            The number of milliseconds each frame is displayed
        """
        frames = []
        iterator = trange(
            self.parameters[0],
            self.parameters[-1],
            step_size,
            desc="Making gif",
            unit="frames",
        )
        for parameter in iterator:
            frames.append(self._get_frame(parameter))

        if not path.endswith(".gif"):
            path += ".gif"
        gif.save(frames, path, duration=frame_duration)
