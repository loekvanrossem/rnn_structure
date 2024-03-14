from matplotlib import axes


def axes_scale(axes: axes.Axes) -> float:
    ax_w, ax_h = axes.bbox.width, axes.bbox.height
    scale_x = 30 * (axes.get_xlim()[1] - axes.get_xlim()[0]) / ax_w
    scale_y = 30 * (axes.get_ylim()[1] - axes.get_ylim()[0]) / ax_h
    return scale_x, scale_y
