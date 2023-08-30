import matplotlib
from matplotlib import rc, rcParams, cycler
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.axis import Axis

COLORS_GRADIENT = [
    "f92485",
    "b4189f",
    "7409b9",
    "540dab",
    "480da7",
    "3d0ba2",
    "3f37ca",
    "4461ed",
    "4a93f1",
    "4bc9f1",
]
COLORS_MIXED = [
    "138086",
    "DC8665",
    "534686",
    "FEB462",
    "AF87CE",
    "6096FD",
    "EF6642",
]
COLORS_CONTRAST = ["FF044F", "760AC0", "00D950", "9F82C9", "471664"]


cdict_seq = {
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
}
COLORMAP_SEQUENTIAL = matplotlib.colors.LinearSegmentedColormap(
    "COLORMAP_SEQUENTIAL", segmentdata=cdict_seq
)
try:
    matplotlib.colormaps.register(COLORMAP_SEQUENTIAL)
except ValueError:
    pass


def pub_show(colors="mixed"):
    match colors:
        case "mixed":
            color_scheme = COLORS_MIXED
        case "gradient":
            color_scheme = COLORS_GRADIENT
        case "contrast":
            color_scheme = COLORS_CONTRAST

    # rc("font", **{"sans-serif": "Go Medium"})
    # rc("font", **{"sans-serif": "Noto Sans Math"})
    rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

    SMALL_SIZE = 13
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 20

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig = plt.gcf()
    ax = plt.gca()

    # rcParams["figure.dpi"] = 200

    alpha = 0.1
    color = "w"
    linewidth = 3.5
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

    # Lines
    rcParams["axes.prop_cycle"] = cycler(color=color_scheme)
    for line in ax.get_lines():
        line.set_linewidth(3)
        line.set_solid_capstyle("round")
        Axis.set_path_effects(line, path_effects)

    # Legend
    n_labels = len(ax.get_legend_handles_labels()[0])
    # legend = ax.get_legend()
    if n_labels > 0:
        legend = plt.legend(
            # [text.get_text() for text in legend.get_texts()],
            loc="upper right",
            fancybox=True,
            framealpha=0.9,
            shadow=True,
            borderpad=0.6,
            edgecolor="0.7",
        )
        for line in legend.get_lines():
            line.set_linewidth(3)
        legend.get_frame().set_linewidth(2)

    # Axes
    n_ticks = 3 + int(fig.get_figheight() * 0.5)
    plt.locator_params(nbins=n_ticks - 1, min_n_ticks=n_ticks)
    ax.grid("on", alpha=0.4, linestyle="--")
    border_color = "0.25"
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[:].set_capstyle("round")
    ax.spines[:].set_linewidth(3)
    ax.spines[:].set_color(border_color)
    for axis in ("x", "y"):
        ax.tick_params(axis=axis, colors=border_color, width=3, length=4)

    # Scatter
    for n, points in enumerate(ax.collections):
        points.set_alpha(0.5)
    if len(ax.get_lines()) > 0 and len(ax.collections) == 1:
        ax.collections[0].set_color("0.5")

    # Images
    rc("image", cmap="COLORMAP_SEQUENTIAL")
    if len(ax.get_images()) == 1:
        ax.grid(False)
        ax.spines[["right", "top"]].set_visible(True)
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
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
        fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=COLORMAP_SEQUENTIAL), ax=ax, shrink=1
        )

    plt.show()
