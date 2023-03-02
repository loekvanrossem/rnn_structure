from typing import Optional

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
from matplotlib import axes

from data_analysis.automata import Automaton, AutomatonHistory, State
from data_analysis.visualization import animation


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

    unused_states = automaton.states.copy()
    layers = [[automaton.initial_state]]
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
    coordinates = {}
    x_step = width / (len(layers) + 1)
    for n_x, layer in enumerate(layers):
        x_coordinate = (n_x + 1) * x_step
        y_step = height / (len(layer) + 1)
        for n_y, state in enumerate(layer):
            y_coordinate = (n_y + 1) * y_step
            coordinates[state] = (x_coordinate, y_coordinate)

    return coordinates


def display_automata(
    automaton: Automaton,
    axes: Optional[axes.Axes] = None,
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

    if not axes:
        figure, axes = plt.subplots()
    axes.set_xlim(0, width)
    axes.set_ylim(0, height)
    for state, (x, y) in coordinates.items():
        # Plot states
        radius = 0.02
        circle = plt.Circle((x, y), radius=radius, color="blue")
        axes.add_artist(circle)
        circle_boundary = plt.Circle((x, y), radius=radius, color="black", fill=False)
        axes.add_artist(circle_boundary)

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
            axes.text(
                x - 0.07,
                y - 0.03,
                f"{tuple(np.round(output,2))}",
                fontsize=8,
                path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
                clip_on=True,
            )

        # Plot initial state name
        (x, y) = coordinates[initial_state]
        axes.text(
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
            slack = 0.15
            x_start = (1 - slack) * x_prev + slack * x_next
            y_start = (1 - slack) * y_prev + slack * y_next
            y_end = slack * y_prev + (1 - slack) * y_next
            x_end = slack * x_prev + (1 - slack) * x_next
            if prev_state == next_state:
                x_start -= 0.02
                y_start += 0.02
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
            )
            axes.add_artist(transition)
            axes.text(
                0.45 * x_prev + 0.55 * x_next,
                0.35 * y_prev + 0.65 * y_next,
                input_symbol,
                fontsize=8,
                path_effects=[pe.Stroke(linewidth=2, foreground="w"), pe.Normal()],
                clip_on=True,
            )

    axes.set_aspect(1)
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)


class AutomatonAnimation(animation.AnimationSubPlot):
    """
    Display an automaton varying over time

    automaton_history : AutomatonHistory
        The automaton history to be displayed
    """

    def __init__(self, automaton_history: AutomatonHistory):
        self.automaton_history = automaton_history

    def plot(self, axes: axes.Axes):
        try:
            display_automata(self.automaton_history[-1], axes=axes)
        except KeyError:
            pass  # Some epochs might not have generated a valid automata
        self.axes = axes

    def update(self, parameter: int):
        try:
            self.axes.clear()
            display_automata(self.automaton_history[parameter], axes=self.axes)
        except KeyError:
            pass  # Some epochs might not have generated a valid automata
