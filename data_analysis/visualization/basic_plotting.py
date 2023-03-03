from matplotlib import axes


def axes_scale(axes: axes.Axes) -> float:
    fig_size = axes.get_figure().get_figheight()
    scale = (axes.get_ylim()[1] - axes.get_ylim()[0]) / fig_size
    return scale
