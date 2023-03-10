from typing import Optional

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
from matplotlib import axes
import matplotlib.style as mplstyle

from data_analysis.visualization.basic_plotting import axes_scale
from data_analysis.automata import Automaton, AutomatonHistory, State
from data_analysis.visualization import animation

mplstyle.use("fast")


def state_placement(automaton: Automaton) -> list[list[State]]:
    """
    Divide automaton states in layers with transitions going mostly from left to right.

    Parameters
    ----------
    automaton : Automaton
        The automaton from which the states locations are computed

    Returns
    -------
    layers : list[list[State]]
        The states divided into layers
    """
    transitions = automaton.transition_function

    initial_state = automaton.initial_state
    layers = [[initial_state]]
    unused_states = automaton.states.copy()
    unused_states.remove(initial_state)
    for layer in layers:
        new_layer = []
        for state in layer:
            state_trans = {i: s_n for (s, i), s_n in transitions.items() if s == state}
            for next_state in state_trans.values():
                if next_state not in unused_states:
                    continue
                new_layer.append(next_state)
                unused_states.remove(next_state)
        if new_layer:
            layers.append(new_layer)

    return layers


def layers_to_coordinates(
    layers: list[list[State]], width: float, height: float
) -> dict[State, tuple[int, int]]:
    """
    Assign coordinates to states divided in layers.

    Parameters
    ----------
    layers : list[list[State]]
        The states divided into layers
    width : float
        The width of the frame
    height : float
        The height of the frame

    Returns
    -------
    coordinates : dict[State, tuple[int, int]]
        A pair of x,y coordinates for each state
    """
    margin_size = 0.5
    x_step = width / (len(layers) - 1 + 2 * margin_size)

    coordinates = {}
    for n_x, layer in enumerate(layers):
        x_coordinate = (n_x + margin_size) * x_step
        y_step = height / (len(layer) - 1 + 2 * margin_size)
        for n_y, state in enumerate(layer):
            y_coordinate = (n_y + margin_size) * y_step
            coordinates[state] = (x_coordinate, y_coordinate)

    return coordinates


def display_automata(
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
                y - 0.03,
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
            slack = 0.05
            x_start = (1 - slack) * x_prev + slack * x_next
            y_start = (1 - slack) * y_prev + slack * y_next
            y_end = slack * y_prev + (1 - slack) * y_next
            x_end = slack * x_prev + (1 - slack) * x_next
            if prev_state == next_state:
                x_start -= 0.03
                y_start += 0.01
                x_end += 0.03
                y_end += 0.01
                angle = 115
            else:
                angle = 180
            transition = patches.FancyArrowPatch(
                (x_start, y_start),
                (x_end, y_end),
                arrowstyle="Fancy,head_length=3,head_width=3",
                connectionstyle=f"angle3,angleA=90,angleB={angle}",
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


class AutomatonAnimation(animation.AnimationSubPlot):
    """
    Display an automaton varying over time

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
