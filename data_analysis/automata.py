import pandas as pd
import numpy as np


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

    Methods
    -------
    compute(input_string : str) -> str
        Compute the output for a given input string.
    """

    def __init__(
        self,
        states: list[State],
        initial_state: State,
        transition_function: dict[tuple[State, str], State],
        output_function: dict[State, np.ndarray],
    ) -> None:
        self.states = states
        self.initial_state = initial_state
        self.transition_function = transition_function
        self.output_function = output_function

    def compute(self, input_string: str) -> np.ndarray:
        """Compute the output for a given input string."""
        state = self.initial_state
        for input_symbol in input_string:
            state = self.transition_function[state, input_symbol]
        output = self.output_function[state]
        return output


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

    def __getitem__(self, epoch: int) -> Automaton:
        """Get the automaton at a certain epoch."""
        states = self.states[epoch]
        initial_state = self.initial_states[epoch]
        transitions = {
            (s, i): s_n for (e, s, i), s_n in self.transitions.items() if e == epoch
        }
        outputs = {s: o for (e, s), o in self.outputs.items() if e == epoch}
        automaton = Automaton(states, initial_state, transitions, outputs)
        return automaton


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
            if np.linalg.norm(activation - np.mean(group[1], axis=0)) < merge_distance:
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
    states = []
    states_last_epoch = []
    initial_states = []
    transitions = {}
    outputs = {}
    for epoch, hidden_states_current in hidden_states.groupby("Epoch"):
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

        # Get output function
        for state in states_this_epoch:
            input_string = state.name.split(", ")[0]
            if input_string == "initial":
                try:
                    input_string = state.name.split(", ")[1]
                except IndexError:
                    outputs[epoch, state] = None
                    continue
            output = output_values.query(
                f"Epoch == {epoch} and Input == '{input_string}'"
            ).to_numpy()[0]
            outputs[epoch, state] = output

        states_last_epoch = states_this_epoch

    automaton_history = AutomatonHistory(states, initial_states, transitions, outputs)

    return automaton_history
