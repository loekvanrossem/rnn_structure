import numpy as np

from visualization import animation


def display_hierarchy(
    automaton: Automaton,
    ax: Optional[axes.Axes] = None,
    width: float = 1.0,
    height: float = 1.0,
):
    """
    Display a drawing of a given automaton.

    Parameters
    ----------
    Automaton : Automaton
        The automaton to display
    axes : axes.Axes, optional, default None
        If provided plot on this axes
    width : float, optional, default 1.0
        The width of the image
    height : float, optional, default 1.0
        The height of the image
    """
    transitions = automaton.transition_function
    outputs = automaton.output_function
    initial_state = automaton.initial_state
    layers = state_placement(automaton)
    coordinates = layers_to_coordinates(layers, width=width, height=height)

    if not ax:
        figure, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    scale = axes_scale(ax)

    for state, (x, y) in coordinates.items():
        # Plot states
        radius = 0.02
        circle = plt.Circle((x, y), radius=radius, color="blue")
        ax.add_artist(circle)
        circle_boundary = plt.Circle((x, y), radius=radius, color="black", fill=False)
        ax.add_artist(circle_boundary)

        # Plot names
        # group = state.name.split(", ")
        # for n, input in enumerate(group):
        #     input_label = plt.text(
        #         x,
        #         y + (n + 1 / 2) * 1.3 * radius,
        #         input,
        #         fontsize=10,
        #         path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
        #         clip_on=True,
        #     )
        # axes.add_artist(input_label)
        # output = outputs[state]

        # Plot outputs
        output = outputs[state]
        if output is not None:
            ax.text(
                x - 0.08,
                y - 0.04,
                f"{tuple(np.round(output,2))}",
                fontsize=8,
                path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
                clip_on=True,
            )

        # Plot initial state name
        (x, y) = coordinates[initial_state]
        ax.text(
            x - 0.04,
            y + 0.03,
            "initial",
            fontsize=8,
            path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
            clip_on=True,
        )

        # Plot transitions
        symbols = []
        for (prev_state, input_symbol), next_state in transitions.items():
            if input_symbol not in symbols:
                symbols.append(input_symbol)
            x_prev, y_prev = coordinates[prev_state]
            x_next, y_next = coordinates[next_state]
            # slack = 0
            # x_start = (1 - slack) * x_prev + slack * x_next
            # y_start = (1 - slack) * y_prev + slack * y_next
            # y_end = slack * y_prev + (1 - slack) * y_next
            # x_end = slack * x_prev + (1 - slack) * x_next
            x_start, y_start, x_end, y_end = x_prev, y_prev, x_next, y_next
            angleA = 90 + (y_prev - y_next) * 36
            if prev_state == next_state:
                x_start -= 0.025
                y_start += 0.01
                x_end += 0.035
                y_end += 0.01
                angleB = 115
            else:
                angleB = 180
            transition = patches.FancyArrowPatch(
                (x_start, y_start),
                (x_end, y_end),
                arrowstyle="Fancy,head_length=3,head_width=3",
                connectionstyle=f"angle3,angleA={angleA},angleB={angleB}",
                shrinkA=10,
                shrinkB=10,
            )
            ax.add_artist(transition)
            # Add input symbol label
            symbol_number = symbols.index(input_symbol)
            ax.text(
                0.55 * x_prev + 0.45 * x_next + 0.2 * symbol_number * scale,
                0.2 * y_prev + 0.8 * y_next + 0.2 * scale,
                input_symbol,
                fontsize=8,
                path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
                clip_on=True,
            )

    ax.set_aspect(1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


class HierarchyAnimation(animation.AnimationSubPlot):
    """


    automaton_history : AutomatonHistory
        The automaton history to be displayed
    """

    def __init__(self, automaton_history: AutomatonHistory):
        self.automaton_history = automaton_history

    def plot(self, ax: axes.Axes):
        try:
            display_automata(self.automaton_history[-1], ax=ax)
        except KeyError:
            pass  # Some epochs might not have generated a valid automata
        self.ax = ax

    def update(self, parameter: int):
        try:
            self.ax.clear()
            display_automata(self.automaton_history[parameter], ax=self.ax)
        except KeyError:
            self.ax.clear()  # Some epochs might not have generated a valid automata
