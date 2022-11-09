import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from adjustText import adjust_text
from ipywidgets import Layout, interact, IntSlider, Button
from IPython.display import display, clear_output


def hidden_repr(hidden_states: pd.DataFrame, transform="PCA", n_labels=10, fig_size=5):
    """
    Show a low dimensional representation of the hidden state dynamics

    Parameters
    ----------
    hidden_states : Dataframe (Epoch, Dataset, Input)
        Dataframe containing the hidden dimension values for each input and epoch
    transform : String
        The type of dimensionality reduction to use: PCA or MDS
    n_labels : int, default 10
        The number of epochs which will be labeled
    fig_size : float, default 5
        The size of the figure
    """
    # Apply dimensionality reduction
    index = hidden_states.index
    if transform == "MDS":
        mds = MDS(n_components=2)
        x = mds.fit_transform(hidden_states)
    if transform == "PCA":
        pca = PCA(n_components=2)
        x = pca.fit_transform(hidden_states)
    x = pd.DataFrame(x, index=index)

    # Plot
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111)
    points = []
    for dataset_name, dataset in x.groupby("Dataset"):
        for input_name, input in dataset.groupby("Input"):
            line = ax.plot(input[0], input[1], label=input_name, zorder=0)
            color = line[0].get_color()
            # Label some points
            first_epoch = hidden_states.index.get_level_values("Epoch").min()
            last_epoch = hidden_states.index.get_level_values("Epoch").max()
            log_range = np.logspace(
                np.log(first_epoch + 0.999),
                np.log(last_epoch + 1),
                num=n_labels,
                base=np.e,
            ).astype(int)
            for epoch in log_range:
                ax.scatter(input[0][epoch], input[1][epoch], s=10, c="Black", zorder=1)
                ax.annotate(epoch, (input[0][epoch], input[1][epoch]), zorder=2)
            # Add time moveable points
            point = ax.plot(
                input[0][first_epoch],
                input[1][first_epoch],
                "o",
                c=color,
                zorder=3,
                markeredgecolor="black",
            )
            label = ax.text(
                input[0][first_epoch],
                input[1][first_epoch],
                input_name,
                path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
                zorder=4,
            )
            points.append((point, input, label))
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    # Add slider for time movable points
    int_wdgt = IntSlider(
        description="Epoch:",
        value=first_epoch,
        min=first_epoch,
        max=last_epoch,
        step=1,
        layout=Layout(width=f"{int(fig_size*7)}%"),
    )

    scale = (ax.get_ylim()[1] - ax.get_ylim()[0]) / fig_size

    def smallest_dist(point, labels_x_pos, labels_y_pos):
        if len(labels_x_pos) == 0:
            return np.inf
        distances = (point[0] - np.array(labels_x_pos)) ** 2 + (
            (point[1] - np.array(labels_y_pos)) ** 2
        )
        return np.sqrt(min(distances))

    @interact(epoch=int_wdgt)
    def slider_epochs(epoch):
        labels_x_pos = []
        labels_y_pos = []
        for point, input, label in points:
            x = float(input[0][epoch])
            y = float(input[1][epoch])

            # Position point
            point[0].set_xdata(x)
            point[0].set_ydata(y)

            # Position label
            pos = [x + 0.06 * scale, y + 0.02 * scale]
            while smallest_dist(pos, labels_x_pos, labels_y_pos) < 0.15 * scale:
                pos[1] += 0.05 * scale
            label.set_position(pos)
            labels_x_pos.append(pos[0])
            labels_y_pos.append(pos[1])
        display(fig)
        clear_output(wait=True)
