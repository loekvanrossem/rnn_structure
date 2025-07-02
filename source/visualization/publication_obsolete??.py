from typing import Optional

import numpy as np

import matplotlib
from matplotlib import rc, rcParams, cycler
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.axis import Axis
from matplotlib.colors import LinearSegmentedColormap


COLORS_MIXED = [
    [
        "138086",
        "DC8665",
        "534686",
        "FEB462",
        "AF87CE",
        "6096FD",
        "EF6642",
        "A2C579",
        "860A35",
        "ADE4DB",
    ],
    ["FF044F", "760AC0", "00D950", "9F82C9", "471664", "05DFD7"],
]

cdict_seq = [
    {
        "red": (
            (0.0, 64 / 255, 64 / 255),
            (0.2, 112 / 255, 112 / 255),
            (0.4, 230 / 255, 230 / 255),
            (0.6, 253 / 255, 253 / 255),
            (0.8, 244 / 255, 244 / 255),
            (1.0, 169 / 255, 169 / 255),
        ),
        "green": (
            (0.0, 57 / 255, 57 / 255),
            (0.2, 198 / 255, 198 / 255),
            (0.4, 241 / 255, 241 / 255),
            (0.6, 219 / 255, 219 / 255),
            (0.8, 109 / 255, 109 / 255),
            (1.0, 23 / 255, 23 / 255),
        ),
        "blue": (
            (0.0, 144 / 255, 144 / 255),
            (0.2, 162 / 255, 162 / 255),
            (0.4, 146 / 255, 146 / 255),
            (0.6, 127 / 255, 127 / 255),
            (0.8, 69 / 255, 69 / 255),
            (1.0, 69 / 255, 69 / 255),
        ),
    },
    {
        "red": (
            (0.0, 75 / 255, 75 / 255),
            (0.33, 63 / 255, 63 / 255),
            (0.67, 84 / 255, 84 / 255),
            (1.0, 249 / 255, 249 / 255),
        ),
        "green": (
            (0.0, 201 / 255, 201 / 255),
            (0.33, 55 / 255, 55 / 255),
            (0.67, 13 / 255, 13 / 255),
            (1.0, 36 / 255, 36 / 255),
        ),
        "blue": (
            (0.0, 241 / 255, 241 / 255),
            (0.33, 202 / 255, 202 / 255),
            (0.67, 171 / 255, 171 / 255),
            (1.0, 133 / 255, 133 / 255),
        ),
    },
    {
        "red": (
            (0.0, 42 / 255, 42 / 255),
            (0.33, 0 / 255, 0 / 255),
            (0.67, 100 / 255, 100 / 255),
            (1.0, 250 / 255, 250 / 255),
        ),
        "green": (
            (0.0, 72 / 255, 72 / 255),
            (0.33, 137 / 255, 137 / 255),
            (0.67, 201 / 255, 201 / 255),
            (1.0, 250 / 255, 250 / 255),
        ),
        "blue": (
            (0.0, 88 / 255, 88 / 255),
            (0.33, 138 / 255, 138 / 255),
            (0.67, 135 / 255, 135 / 255),
            (1.0, 110 / 255, 110 / 255),
        ),
    },
]

for n, cdict in enumerate(cdict_seq):
    try:
        matplotlib.colormaps.register(
            LinearSegmentedColormap(f"Colormap_seq_{n}", cdict)
        )
    except ValueError:
        pass


def set_color_mixed(index=0):
    rcParams["axes.prop_cycle"] = cycler(color=COLORS_MIXED[index])


def set_color_gradient(index=0, N=10):
    cmap = matplotlib.colors.LinearSegmentedColormap(
        "COLORMAP_SEQUENTIAL", segmentdata=cdict_seq[index], N=N
    )
    color_scheme = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    rcParams["axes.prop_cycle"] = cycler(color=color_scheme)

    plt.rcParams["image.cmap"] = f"Colormap_seq_{index}"


def skip_colors(n):
    for _ in range(n):
        plt.plot(0, 0)


def pub_show(
    save_path: Optional[str] = None, border_color="0.25", show_plot=True, **kwargs
):
    fig = plt.gcf()
    ax = plt.gca()

    rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    SMALL_SIZE = 13
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ax.spines[:].set_capstyle("round")
    ax.spines[:].set_linewidth(3)
    ax.spines[:].set_color(border_color)
    for axis in ("x", "y"):
        ax.tick_params(axis=axis, colors=border_color, width=3, length=4)

    if save_path:
        kwargs.setdefault("bbox_inches", "tight")
        plt.savefig(save_path, dpi=200, **kwargs)
        plt.savefig(save_path + ".eps", format="eps", **kwargs)

    if show_plot:
        plt.show()


def plt_show(no_axes=False, **kwargs):
    fig = plt.gcf()
    ax = plt.gca()

    # Lines
    for line in ax.get_lines():
        line.set_linewidth(2)
        line.set_solid_capstyle("round")

    # Legend
    # n_labels = len(ax.get_legend_handles_labels()[0])
    # legend = ax.get_legend()
    if ax.get_legend():
        legend = plt.legend(
            # [text.get_text() for text in legend.get_texts()],
            loc="upper right",
            # fancybox=True,
            frameon=False,
            framealpha=0.9,
            shadow=True,
            borderpad=0.6,
            edgecolor="0.7",
            handlelength=1.5,
        )
        for line in legend.get_lines():
            line.set_linewidth(2.5)
        legend.get_frame().set_linewidth(2)
        legend.get_frame().set_alpha(0)

    # Axes
    n_ticks = int(2.5 + fig.get_figheight() * 0.5)
    plt.locator_params(nbins=n_ticks - 1, min_n_ticks=n_ticks)
    ax.grid("on", alpha=0.4, linestyle="--")
    ax.spines[["right", "top"]].set_visible(False)

    # Scatter
    for points in ax.collections:
        points.set_alpha(0.5)
        label = points.get_label()
        if label[0] == "_":
            continue
        mean = np.mean(points.get_offsets().data, axis=0)
        x_pos = max(min(mean[0], ax.get_xlim()[1]), ax.get_xlim()[0])
        y_pos = max(min(mean[1], ax.get_ylim()[1]), ax.get_ylim()[0])
        color = points.get_facecolor()
        # color = color**2
        color = color - np.min(color)
        color[0][3] = 1
        plt.text(
            x_pos,
            y_pos,
            label,
            color=color,
            path_effects=[
                pe.Stroke(linewidth=3, foreground="w"),
                pe.Normal(),
            ],
        )
    if len(ax.get_lines()) > 0 and len(ax.collections) == 1:
        ax.collections[0].set_color("0.5")

        # Add glow
        alpha = 0.05
        color = "w"
        linewidth = 2.5
        path_effects = [
            pe.SimpleLineShadow(
                linewidth=linewidth, offset=(1.4, 0), alpha=alpha, foreground=color
            ),
            pe.SimpleLineShadow(
                linewidth=linewidth, offset=(-1.4, 0), alpha=alpha, foreground=color
            ),
            pe.SimpleLineShadow(
                linewidth=linewidth, offset=(0, -1.4), alpha=alpha, foreground=color
            ),
            pe.SimpleLineShadow(
                linewidth=linewidth, offset=(0, 1.4), alpha=alpha, foreground=color
            ),
            pe.SimpleLineShadow(
                linewidth=linewidth, offset=(1, 1), alpha=alpha, foreground=color
            ),
            pe.SimpleLineShadow(
                linewidth=linewidth, offset=(-1, 1), alpha=alpha, foreground=color
            ),
            pe.SimpleLineShadow(
                linewidth=linewidth, offset=(1, -1), alpha=alpha, foreground=color
            ),
            pe.SimpleLineShadow(
                linewidth=linewidth, offset=(-1, -1), alpha=alpha, foreground=color
            ),
            pe.Normal(),
        ]
        for line in ax.get_lines():
            Axis.set_path_effects(line, path_effects)

    if no_axes:
        ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
        plt.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            right=False,
            top=False,
            labelbottom=False,
            labelleft=False,
        )
    pub_show(border_color="0.25", **kwargs)


def plt_legend(legend, save_path, expand=[-5, -15, 10, 5]):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    ax = plt.gca()
    ax.set_axis_off()

    plt_show(save_path=save_path, bbox_inches=bbox)


def _remove_none_labels(labeling):
    locations = [loc for loc in np.arange(len(labeling)) if labeling[loc] is not None]
    labels = [label for label in labeling if label is not None]
    return locations, labels


def im_show(
    x_labels: Optional[list] = None,
    y_labels: Optional[list] = None,
    colorbar=True,
    **kwargs,
):
    fig = plt.gcf()
    ax = plt.gca()

    ax.spines[["right", "top"]].set_visible(True)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")

    # lims = ax.get_images()[0].get_clim()

    if colorbar:
        cb = plt.colorbar(ax=ax, shrink=0.8, aspect=15)
        cb.outline.set_linewidth(2.5)

    if x_labels is not None:
        locations, labels = _remove_none_labels(x_labels)
        ax.set_xticks(locations, labels=labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    else:
        plt.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
        )
    if y_labels is not None:
        locations, labels = _remove_none_labels(y_labels)
        ax.set_yticks(locations, labels=labels)
    else:
        plt.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
        )

    pub_show(border_color="0.1", **kwargs)
