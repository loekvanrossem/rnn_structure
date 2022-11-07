import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

from ipywidgets import Layout, interact, IntSlider
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
            ax.plot(input[0], input[1], label=input_name, zorder=0)
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
                ax.annotate(
                    epoch,
                    (input[0][epoch] + 0.02, input[1][epoch] + 0.02),
                    zorder=2,
                )
            # Add time moveable points
            point = ax.plot(
                input[0][first_epoch], input[1][first_epoch], "ko", alpha=0, zorder=3
            )
            points.append((point, input))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    # Add slider for time movable points
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
        for point, input in points:
            point[0].set_xdata(input[0][epoch])
            point[0].set_ydata(input[1][epoch])
            point[0].set_alpha(1)
        display(fig)
        clear_output(wait=True)
