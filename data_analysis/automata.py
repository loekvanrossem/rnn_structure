from typing import Callable

import pandas as pd
import numpy as np


class State:
    """
    A state in an automaton.
    """

class Automaton:
    """
    A finite state automaton.

    Attributes
    ----------
    states : list
        The states the automata can be in during computation
    transition_function : function(state, input_symbol) -> state
        A function that says to which state each state transitions for a given input symbol 
    output_funtion : function(state) -> output_symbol
        A function assigning an output_symbol to each state

    Methods
    -------
    compute(input : str) -> str
    """

    def __init__(
        self, input_alphabet : set, output_alphabet : set, states: list[State], transition_function: Callable[[State,str],State], output_function: Callable[[State],str]
    ) -> None:
        self.states = states
        self.transition_function = transition_function
        self.output_function = output_function

    def compute(input : list[str]) -> str
    """ """
        state = initial_state
        for input_symbol in input:
            state, output = self.transition_function(state, input_symbol)

        return output


class AutomatonHistory:
    """ """


def to_automaton(hidden_states: pd.DataFrame, output_values : pd.DataFrame, merge_distance : float):
    """

    Parameters
    ----------
    hidden_states : Dataframe (Epoch, Dataset, Input)
        Dataframe containing the hidden dimension values for each input and epoch

    Returns
    -------

    """
