from matplotlib import axes


def axes_scale(axes: axes.Axes) -> float:
    ax_h = axes.bbox.height
    scale = 30 * (axes.get_ylim()[1] - axes.get_ylim()[0]) / ax_h
    return scale
