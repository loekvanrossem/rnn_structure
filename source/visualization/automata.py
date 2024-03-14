from enum import auto
from typing import Optional

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
from matplotlib import axes
import matplotlib.style as mplstyle

from visualization.basic_plotting import axes_scale
from data_analysis.automata import Automaton, AutomatonHistory, State
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


def has_all_transitions(
    state: State, transition_function: dict[tuple[State, str], State]
) -> bool:
    """
    Return True if the state has transitions for all possible input symbols.

    Parameters
    ----------
    state : State
        The state to check for transitions
    transition_function : dict[tuple[State, str], State]
        The transition function of the automaton

    Returns
    -------
    bool
        True if transitions exist for all possible input symbols, False otherwise
    """
    # Extract input symbols from transition_function keys
    input_symbols = {symbol for (_, symbol) in transition_function.keys()}

    # Check if transitions exist for all possible input symbols
    for symbol in input_symbols:
        if (state, symbol) not in transition_function:
            return False
    return True


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

    if not ax:
        figure, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    scale_x, scale_y = axes_scale(ax)

    for state, (x, y) in coordinates.items():
        # Plot states
        radius = 0.02
        if has_all_transitions(state, automaton.transition_function):
            color = "blue"
        else:
            color = "gray"
        circle = plt.Circle((x, y), radius=radius, color=color)
        ax.add_artist(circle)
        circle_boundary = plt.Circle((x, y), radius=radius, color="black", fill=False)
        ax.add_artist(circle_boundary)

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
                0.55 * x_prev + 0.45 * x_next + 0.2 * symbol_number * scale_x,
                0.2 * y_prev + 0.8 * y_next + 0.2 * scale_y,
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
