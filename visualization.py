import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from ipywidgets import Layout, interact, IntSlider
from IPython.display import display
import gif

from typing import Optional
from tqdm import trange

from preprocessing import Encoding


def hidden_repr(
    hidden_states: pd.DataFrame,
    outputs: pd.DataFrame,
    transform: tuple[str, str] = ("PCA", "PCA"),
    n_labels: int = 10,
    fig_size: int = 5,
    encoding: Optional[Encoding] = None,
    plot_labels: bool = True,
    gif_path: Optional[str] = None,
):
    """
    Show a low dimensional representation of the hidden state dynamics

    Parameters
    ----------
    hidden_states : Dataframe (Epoch, Dataset, Input)
        Dataframe containing the hidden dimension values for each input and epoch
    transform : (String, String)
        The type of dimensionality reduction to use: PCA or MDS
    n_labels : int, default 10
        The number of epochs which will be labeled
    fig_size : float, default 5
        The size of the figure
    encoding : Encoding, default None
        If provided, also plot the encoded symbol values
    plot_labels : Bool, default True
        If true plot the input names of each datapoint
    gif_path: str, default None
        If provided, save a gif here
    """

    def _plot(ax, data: pd.DataFrame, transform: str, encoding=None):
        index = data.index
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

        points = []
        for dataset_name, dataset in data_red.groupby("Dataset"):
            for input_name, input in dataset.groupby("Input"):
                x_values = input.iloc[:, 0]
                y_values = input.iloc[:, 1]
                line = ax.plot(x_values, y_values, label=input_name, zorder=0)
                color = line[0].get_color()
                # Label some points
                first_epoch = data.index.get_level_values("Epoch").min()
                last_epoch = data.index.get_level_values("Epoch").max()
                log_range = np.logspace(
                    np.log(first_epoch + 0.999),
                    np.log(last_epoch + 1),
                    num=n_labels,
                    base=np.e,
                ).astype(int)
                for epoch in log_range:
                    ax.scatter(
                        x_values[epoch], y_values[epoch], s=10, c="Black", zorder=1
                    )
                    ax.annotate(epoch, (x_values[epoch], y_values[epoch]), zorder=2)
                # Add time moveable points
                point = ax.plot(
                    x_values[last_epoch],
                    y_values[last_epoch],
                    "o",
                    c=color,
                    zorder=3,
                    markeredgecolor="black",
                )
                if plot_labels and (type(input_name) == str):
                    label = ax.text(
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

        scale = (ax.get_ylim()[1] - ax.get_ylim()[0]) / fig_size

        # Add fixed points (WARNING: does not work for transformations)
        if encoding is not None:
            for symbol in encoding.symbols:
                point = ax.plot(
                    encoding(symbol)[0],
                    encoding(symbol)[1],
                    "o",
                    c="black",
                    zorder=3,
                )
                label = ax.text(
                    encoding(symbol)[0] + 0.06 * scale,
                    encoding(symbol)[1] + 0.06 * scale,
                    f"Output {symbol}",
                    path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
                    zorder=4,
                )

        return points

    first_epoch = hidden_states.index.get_level_values("Epoch").min()
    last_epoch = hidden_states.index.get_level_values("Epoch").max()
    n_epochs = last_epoch - first_epoch

    def smallest_dist(point, labels_x_pos, labels_y_pos):
        distances = 0.25 * (point[0] - np.array(labels_x_pos)) ** 2 + (
            (point[1] - np.array(labels_y_pos)) ** 2
        )
        return np.sqrt(min(distances))

    def make_figure():
        fig = plt.figure(figsize=(2 * fig_size, fig_size))
        ax_1 = fig.add_subplot(121)
        points_1 = _plot(
            ax_1,
            hidden_states,
            transform=transform[0],
            encoding=None,
        )
        ax_2 = fig.add_subplot(122)
        points_2 = _plot(
            ax_2,
            outputs,
            transform=transform[1],
            encoding=encoding,
        )
        return fig, points_1, points_2, ax_1, ax_2

    def update_positions(points_1, points_2, ax_1, ax_2, epoch):
        for points, ax in ((points_1, ax_1), (points_2, ax_2)):
            scale = (ax.get_ylim()[1] - ax.get_ylim()[0]) / fig_size
            labels_x_pos = []
            labels_y_pos = []
            for point, input, label in points:
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
                            smallest_dist(pos, labels_x_pos, labels_y_pos)
                            < 0.15 * scale
                        ):
                            if pos[1] > ax.get_ylim()[1] - 0.2 * scale:
                                break
                            pos[1] += 0.05 * scale
                    label.set_position(pos)
                    labels_x_pos.append(pos[0])
                    labels_y_pos.append(pos[1])

    ## Make gif
    n_steps = 500
    step_size = int(np.ceil(n_epochs / n_steps))
    if gif_path is not None:

        @gif.frame
        def frame(epoch):
            fig, points_1, points_2, ax_1, ax_2 = make_figure()
            update_positions(points_1, points_2, ax_1, ax_2, epoch)

        frames = []
        iterator = trange(
            first_epoch, last_epoch, step_size, desc="Making gif", unit="frames"
        )
        for epoch in iterator:
            frames.append(frame(epoch))

        gif.save(frames, gif_path + ".gif", duration=50)

    ## Make interactive plot
    fig_interactive, points_1, points_2, ax_1, ax_2 = make_figure()
    int_wdgt = IntSlider(
        description="Epoch:",
        value=first_epoch,
        min=first_epoch,
        max=last_epoch,
        step=1,
        layout=Layout(width=f"{int(fig_size*7)}%"),
    )

    @interact(epoch=int_wdgt)
    def slider_epochs(epoch):
        update_positions(points_1, points_2, ax_1, ax_2, epoch)
        display(fig_interactive)
