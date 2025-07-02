import warnings
from typing import Optional

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
from matplotlib import axes
import matplotlib.style as mplstyle

from visualization.basic_plotting import axes_scale
from data_analysis.automata import (
    Automaton,
    AutomatonHistory,
    State,
    reduce_automaton,
    has_all_transitions,
)
from visualization import animation

mplstyle.use("fast")


def state_placement(automaton: Automaton) -> list[list[State]]:
    """
    Divide automaton states into layers based on their transitions.

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
    visited_states = set(layers[0])

    while visited_states != set(automaton.states):
        new_layer = []
        for state in layers[-1]:
            next_states = {
                transitions.get((state, i), None) for i in automaton.alphabet
            }
            next_states.discard(None)  # Remove None values (unreachable states)
            next_states -= visited_states  # Remove states already visited
            new_layer.extend(next_states)
            visited_states.update(next_states)  # Mark next states as visited
        if new_layer:
            layers.append(new_layer)
        else:
            break  # No new states found, terminate loop

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
    margin_size = 1
    num_layers = len(layers)
    if num_layers == 0:
        return {}

    x_step = width / (num_layers - 1 + 2 * margin_size)
    coordinates = {}

    for n_x, layer in enumerate(layers):
        x_coordinate = (n_x + margin_size) * x_step
        num_states_in_layer = len(layer)
        if num_states_in_layer == 0:
            continue

        y_step = height / (num_states_in_layer - 1 + 2 * margin_size)
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
    automaton : Automaton
        The automaton to display
    axes : axes.Axes, optional, default None
        If provided plot on this axes
    width : float, optional
        The width of the image
    height : float, optional
        The height of the image
    """
    transitions = automaton.transition_function
    outputs = automaton.output_function
    initial_state = automaton.initial_state
    layers = state_placement(automaton)
    coordinates = layers_to_coordinates(layers, width=width, height=height)
    symbols = list(automaton.alphabet)
    symbols.sort()
    symbol_indices = {symbol: idx for idx, symbol in enumerate(symbols)}

    if not ax:
        figure, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    scale_x, scale_y = axes_scale(ax)

    fontsize = 10

    # Plot states
    for state, (x, y) in coordinates.items():
        radius = 0.03 / (
            1 + np.exp(-len(state.name.split(", ")))
        )  # Vary state size based on number of sequences

        if state == initial_state:
            color = "#333333"
            opacity = 1
        elif has_all_transitions(state, automaton.transition_function):
            color = "#d4ede8"
            opacity = 1
        else:
            # color = "#808080"
            color = "#d4ede8"
            opacity = 0.5

        circle = plt.Circle(
            (x, y),
            radius=radius,
            facecolor=color,
            edgecolor="#333333",
            linewidth=2,
            alpha=opacity,
        )  # Add transparency
        ax.add_artist(circle)

        # # Plot outputs
        # output = outputs[state]
        # if output is not None:
        #     ax.text(
        #         x - 0.08,
        #         y - 0.04,
        #         f"{tuple(np.round(output,2))}",
        #         fontsize=fontsize,
        #         path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
        #         clip_on=True,
        #     )
        # Plot outputs
        output = outputs[state]
        if output is not None:
            label = f"{np.argmax(tuple(np.round(output,2)))}"
            ax.text(
                x - 0.012,
                y - 0.015,
                label,
                fontsize=fontsize + 2,
                path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
                clip_on=True,
                color="#552620",
            )

        # Plot initial state name
        (x_init, y_init) = coordinates[initial_state]
        ax.text(
            x_init - 0.04,
            y_init + 0.03,
            "initial",
            fontsize=fontsize,
            path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
            clip_on=True,
        )

    # Plot transitions
    for (prev_state, input_symbol), next_state in transitions.items():
        x_prev, y_prev = coordinates[prev_state]
        x_next, y_next = coordinates[next_state]
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
            if y_start == y_end:
                angleB = 170
                if x_start < x_end:
                    y_end += 0.01
                else:
                    y_end -= 0.01
        transition = patches.FancyArrowPatch(
            (x_start, y_start),
            (x_end, y_end),
            arrowstyle="Fancy,head_length=3,head_width=3",
            connectionstyle=f"angle3,angleA={angleA},angleB={angleB}",
            shrinkA=10,
            shrinkB=10,
            color="#333333",
        )
        ax.add_artist(transition)

        # Compute label position based on transition angle
        label_offset_x = 0.2 * scale_x * np.cos(np.radians(angleA))
        label_offset_y = 0.2 * scale_y * np.sin(np.radians(angleA))
        if x_start > x_end:
            label_offset_y *= -1

        # Add input symbol label
        symbol_number = symbol_indices[input_symbol]
        ax.text(
            0.55 * x_prev
            + 0.45 * x_next
            + label_offset_x
            + 0.3 * symbol_number * scale_x,
            0.2 * y_prev + 0.8 * y_next + label_offset_y + 0.2 * scale_y,
            input_symbol,
            fontsize=fontsize,
            path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
            clip_on=True,
            color="black",
        )

    ax.set_aspect(1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


class AutomatonAnimation(animation.AnimationSubPlot):
    """
    Display an automaton varying over time.

    Parameters
    ----------
    automaton_history : AutomatonHistory
        The automaton history to be displayed.
    reduce_automate : Boolean, default False
        If true, merge nondistinguishable states
    """

    def __init__(self, automaton_history: AutomatonHistory, reduce_automata=False):
        self.automaton_history = automaton_history
        self.reduce_automata = reduce_automata

    def plot(self, ax: axes.Axes):
        self.ax = ax
        ax.spines[:].set_capstyle("round")
        ax.spines[:].set_linewidth(3)
        ax.spines[:].set_color("0.25")
        self._display_current_automaton()

    def update(self, parameter: int):
        self.ax.clear()
        self._display_current_automaton(parameter)

    def _display_current_automaton(self, index: int = -1):
        """
        Display the automaton at the given index in the history.

        Parameters
        ----------
        index : int, optional
            The index of the automaton in the history. Default is -1 (last).
        """
        try:
            automaton = self.automaton_history[index]
            if self.reduce_automata:
                try:
                    automaton = reduce_automaton(automaton)
                except StopIteration:
                    warnings.warn("Invalid automaton encountered", UserWarning)
            display_automata(automaton, ax=self.ax)
        except KeyError:
            warnings.warn("Invalid automaton encountered", UserWarning)
