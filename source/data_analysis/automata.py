from typing import Any
from xmlrpc.client import Boolean

import pandas as pd
import numpy as np
from tqdm import tqdm

from compilation import ActivationTracker


class State:
    """
    A state in an automaton.

    Attributes
    ----------
    name : str
        The name of this state
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"State({self.name})"


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


class Automaton:
    """
    A finite state automaton.

    Attributes
    ----------
    states : list[State]
        The states the automaton can be in during computation
    initial_state : State
        The starting state
    transition_function : dict((state, input_symbol),state)
        A dictionary that says to which state each state transitions for a given input symbol
    output_funtion : dict((state),output)
        A dictionary assigning an output to each state
    alphabet : set[str]
        The alphabet of input symbols

    Methods
    -------
    compute(input_string : str) -> np.ndarray
        Compute the output for a given input string.
    """

    def __init__(
        self,
        states: list[State],
        initial_state: State,
        transition_function: dict[tuple[State, str], State],
        output_function: dict[State, Any],
    ) -> None:
        self.states = states
        self.initial_state = initial_state
        self.transition_function = transition_function
        self.output_function = output_function

    @property
    def alphabet(self) -> set[str]:
        """The alphabet of input symbols."""
        return {symbol for (_, symbol) in self.transition_function.keys()}

    def compute(self, input_string: str) -> np.ndarray:
        """
        Compute the output for a given input string.

        Parameters
        ----------
        input_string : str
            The input string to compute the output for.

        Returns
        -------
        output : np.ndarray
            The output computed by the automaton.
        """
        for symbol in input_string:
            if symbol not in self.alphabet:
                raise ValueError(
                    f"Symbol '{symbol}' is not in the alphabet of the automaton."
                )
        state = self.initial_state
        for input_symbol in input_string:
            state = self.transition_function[state, input_symbol]
        output = self.output_function[state]
        return output

    def is_finite(self) -> bool:
        for state in self.states:
            if not has_all_transitions(state, self.transition_function):
                return False
        return True


class AutomatonHistory:
    """
    An automaton for each epoch.

    Attributes
    ----------
    states : list[list[State]]
        A history of states in the automaton
    initial_states : list[State]
        A history of the initial state
    transitions : dict[tuple[int, State, str], State]
        A history of the transitions of each state
    outputs : dict[tuple[int, State], np.ndarray]
        A history of the outputs per state

    Methods
    -------
    get_state_changes() -> Tuple[List[int], List[int]]:
        Return the number of state mergers and splits per epoch.
    """

    def __init__(
        self,
        states: list[list[State]],
        initial_states: list[State],
        transitions: dict[tuple[int, State, str], State],
        outputs: dict[tuple[int, State], np.ndarray],
    ):
        self.states = states
        self.initial_states = initial_states
        self.transitions = transitions
        self.outputs = outputs
        self._automata = {}

    def __len__(self):
        return len(self.initial_states)

    def __getitem__(self, epoch: int) -> Automaton:
        """Get the automaton at a certain epoch."""
        if epoch < 0:
            epoch = len(self) + epoch
        if epoch in self._automata:
            automaton = self._automata[epoch]
        else:
            states = self.states[epoch]
            initial_state = self.initial_states[epoch]
            transitions = {
                (s, i): s_n for (e, s, i), s_n in self.transitions.items() if e == epoch
            }
            outputs = {s: o for (e, s), o in self.outputs.items() if e == epoch}
            automaton = Automaton(states, initial_state, transitions, outputs)
            self._automata[epoch] = automaton
        return automaton

    @staticmethod
    def _compute_state_changes(n_removed: int, n_added: int) -> tuple[int, int]:
        """Compute the number of mergers and splits from the number of removed and added states."""
        n_mergers = (1 / 3) * (2 * n_removed - n_added)
        n_splits = (1 / 3) * (2 * n_added - n_removed)
        return int(n_mergers), int(n_splits)

    def get_state_changes(self) -> tuple[list[int], list[int]]:
        """
        Return the number of state mergers and splits per epoch.

        Returns
        -------
        Tuple[List[int], List[int]]:
            The number of state mergers and splits per epoch.
        """
        mergers = []
        splits = []
        old_states = self.states[0]
        for states in self.states[1:]:
            n_removed = len(set(old_states) - set(states))
            n_added = len(set(states) - set(old_states))
            n_mergers, n_splits = self._compute_state_changes(n_removed, n_added)
            mergers.append(n_mergers)
            splits.append(n_splits)
            old_states = states

        return mergers, splits


def group_activations(
    hidden_states: pd.DataFrame, merge_distance: float
) -> list[tuple[list[str], list[np.ndarray]]]:
    """
    Group activations that are close to eachother.

    Parameters
    ----------
    hidden_states : Dataframe (Dataset, Input)
        Dataframe containing the hidden dimension values for each input
    merge_distance : float
        The smallest distance that activations can be apart to still be considered the same state

    Returns
    -------
    grouped_activations : list[tuple[list[str], list[np.ndarray]]]
        A list of groups of (input names, activation vectors)
    """
    grouped_activations = []
    for input_string, activations in hidden_states.groupby("Input"):
        activation = np.array(activations)[0]
        for group in grouped_activations:
            if np.linalg.norm(activation - group[1][0]) < merge_distance:
                group[0].append(input_string)
                group[1].append(activation)
                break
        else:
            new_group = ([input_string], [activation])
            grouped_activations.append(new_group)

    return grouped_activations


def to_automaton_history(
    hidden_states: pd.DataFrame,
    output_values: pd.DataFrame,
    merge_distance: float,
) -> AutomatonHistory:
    """
    Extract an automaton at each epoch from the activations of a neural network.

    Parameters
    ----------
    hidden_states : Dataframe (Epoch, Dataset, Input)
        Dataframe containing the hidden dimension values for each input and epoch,
        assumes that the initial values for the hidden dimension have input label "initial"
    output_values : Dataframe(Epoch, Dataset, Input)
        Dataframe containing the output dimension values for each input and epoch
    merge_distance : float
        The smallest distance that activations can be apart to still be considered the same state

    Returns
    -------
    automaton_history : AutomatonHistory
        An automaton at each epoch representing the neural network during training.
    """
    outputs_per_input = output_values.droplevel("Dataset")

    states = []
    states_last_epoch = []
    initial_states = []
    transitions = {}
    outputs = {}
    for epoch, hidden_states_current in tqdm(
        hidden_states.groupby("Epoch"), desc="Computing automata"
    ):
        # Get states
        states_this_epoch = []
        grouped_activations = group_activations(hidden_states_current, merge_distance)
        for group, _ in grouped_activations:
            state_name = ", ".join(group)
            already_existing_state = next(
                (state for state in states_last_epoch if state.name == state_name), None
            )
            if already_existing_state is not None:
                states_this_epoch.append(already_existing_state)
            else:
                states_this_epoch.append(State(state_name))
        states.append(states_this_epoch)

        # Get transition function
        for state in states_this_epoch:
            group = state.name.split(", ")
            for input_string in group:
                if input_string == "initial":
                    initial_states.append(state)
                    continue
                last_input = input_string[-1]
                prev_state_name = input_string[:-1]
                if prev_state_name == "":
                    prev_state_name = "initial"
                prev_state = next(
                    (
                        s
                        for s in states_this_epoch
                        if prev_state_name in s.name.split(", ")
                    ),
                    None,
                )
                if prev_state is not None:
                    transitions[epoch, prev_state, last_input] = state
        try:
            output_this_epoch = outputs_per_input.loc[epoch]
        except KeyError:
            pass
        # Get output function
        for state in states_this_epoch:
            input_string = state.name.split(", ")[0]
            if input_string == "initial":
                if len(state.name.split(", ")) > 1:
                    input_string = state.name.split(", ")[1]
                else:
                    outputs[epoch, state] = None
                    continue
            output = output_this_epoch.loc[input_string].to_numpy()
            outputs[epoch, state] = output

        states_last_epoch = states_this_epoch

    automaton_history = AutomatonHistory(states, initial_states, transitions, outputs)

    return automaton_history


def to_automaton(
    hidden_function,
    output_function,
    initial_hidden,
    datasets,
    encoding,
    merge_distance_frac=0.1,
):
    """Return automaton for model on given data."""

    ## Get data
    hidden_tracker = ActivationTracker(
        encoding,
        hidden_function,
        datasets,
        initial=lambda: initial_hidden,
    )
    out_tracker = ActivationTracker(encoding, output_function, datasets)
    hidden_tracker.track()
    out_tracker.track()
    data_hid = hidden_tracker.get_trace()
    data_out = out_tracker.get_trace()

    std = float(np.linalg.norm(data_hid.std()))
    automaton = to_automaton_history(
        data_hid,
        data_out,
        merge_distance=merge_distance_frac * std,
    )[0]

    return automaton


def nondistinguishable_partition(automata):
    r"""
    Apply Hopcroft's algorithm to find a partition of nondistinguishable states.

    Pseudocode:

    P := {F, Q \ F}
    W := {F, Q \ F}

    while (W is not empty) do
        choose and remove a set A from W
        for each c in Σ do
            let X be the set of states for which a transition on c leads to a state in A
            for each set Y in P for which X ∩ Y is nonempty and Y \ X is nonempty do
                replace Y in P by the two sets X ∩ Y and Y \ X
                if Y is in W
                    replace Y in W by the same two sets
                else
                    if |X ∩ Y| <= |Y \ X|
                        add X ∩ Y to W
                    else
                        add Y \ X to W
    """

    out_0 = {s for s, o in automata.output_function.items() if np.argmax(o) == 0}
    out_1 = {s for s, o in automata.output_function.items() if np.argmax(o) == 1}
    P = [out_0, out_1]
    W = [out_0, out_1]

    while len(W) > 0:
        A = W.pop()
        for char in automata.alphabet:
            X = {
                s
                for (s, i), s_n in automata.transition_function.items()
                if (s_n in A and i == char)
            }  # set of states which transition via char to A
            for Y in P:
                Y_and_X = Y.intersection(X)
                Y_min_X = Y.difference(X)
                if len(Y_and_X) > 0 and len(Y_min_X) > 0:
                    P.remove(Y)
                    P.append(Y_and_X)
                    P.append(Y_min_X)
                    if Y in W:
                        W.remove(Y)
                        W.append(Y_and_X)
                        W.append(Y_min_X)
                    else:
                        if len(Y_and_X) <= len(Y_min_X):
                            W.append(Y_and_X)
                        else:
                            W.append(Y_min_X)
    return P


def reduce_automaton(automaton: Automaton):
    """
    Compute a reduced version of an automaton with redundant states merged.
    """
    P = nondistinguishable_partition(automaton)

    states = [State(", ".join([s.name for s in group])) for group in P]
    initial_state = next((s for s in states if "initial" in s.name.split(", ")), None)
    transition_function = {
        (
            next(
                (
                    s_group
                    for s_group in states
                    if set(s.name.split(", ")) <= set(s_group.name.split(", "))
                )
            ),
            i,
        ): next(
            (
                s_group
                for s_group in states
                if set(s_n.name.split(", ")) <= set(s_group.name.split(", "))
            )
        )
        for (s, i), s_n in automaton.transition_function.items()
    }
    output_function = {
        s_group: next(
            o
            for s, o in automaton.output_function.items()
            if set(s.name.split(", ")) <= set(s_group.name.split(", "))
        )
        for s_group in states
    }

    return Automaton(states, initial_state, transition_function, output_function)
