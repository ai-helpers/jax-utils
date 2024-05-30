"""High-level abstractions for decision problems (Markov Decision processes, etc...)"""

from typing import Protocol, Tuple, TypeVar

State = TypeVar("State")
State_contra = TypeVar("State_contra", contravariant=True)
Action = TypeVar("Action")
Action_contra = TypeVar("Action_contra", contravariant=True)
Observation = TypeVar("Observation")
Observation_co = TypeVar("Observation_co", covariant=True)
Observation_contra = TypeVar("Observation_contra", contravariant=True)
Cost = TypeVar("Cost")
Cost_co = TypeVar("Cost_co", covariant=True)
Cost_contra = TypeVar("Cost_contra", contravariant=True)
RegularizedCost = TypeVar("RegularizedCost")


class Dynamics(Protocol[State, Action_contra, Cost_co, Observation_co]):
    """Interface defining the dynamics of a
    `(Partially Observable) <https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process>`_
    `Markov Decision Process <https://en.wikipedia.org/wiki/Markov_decision_process>`_.

    When an "agent" interacting with the (PO)MDP plays an "action" (a.k.a. "control")
    in a given "state", the (PO)MDP transitions to a new state and the agent observes
    some signal/feedback in the form of a "cost"/"reward" as well as additional "observations".

    A `Dynamics` is therefore a callable that maps a state-action pair to a state-cost-observation tuple.
    """

    def __call__(
        self, state: State, action: Action_contra
    ) -> Tuple[State, Cost_co, Observation_co]:
        pass


class CostRegularizer(
    Protocol[State_contra, Action_contra, Cost_contra, Observation_contra, Cost_co]
):
    """Interface for callables that map any state-action-cost-observation tuple to a new "regularized" cost.

    `More about regularization <https://en.wikipedia.org/wiki/Regularization_(mathematics)>`_.

    Example: one may want to penalize `action` with high norms, etc...
    """

    def __call__(
        self,
        state: State_contra,
        action: Action_contra,
        cost: Cost_contra,
        observation: Observation_contra,
    ) -> Cost_co:
        pass


class RegularizedDynamics(
    Dynamics[State, Action, RegularizedCost, Observation],
    Protocol[State, Action, Cost, RegularizedCost, Observation],
):
    """Interface defining a wrapper around class :class:`jax_utils.markov_decision_process.Dynamics` that allows to add
    a regularization to the cost.

    A ``RegularizedDynamics`` is itself a :class:`jax_utils.markov_decision_process.Dynamics``.

    `More about regularization <https://en.wikipedia.org/wiki/Regularization_(mathematics)>`_.

    Args:
        dynamics (Dynamics[State, Action, Cost, Observation]): callable defining the dynamics of a (PO)MDP
        cost_regularizer (CostRegularizer[State, Action, Cost, Observation, RegularizedCost]): callable
        defining a cost transformation (regularization)
    """

    dynamics: Dynamics[State, Action, Cost, Observation]
    cost_regularizer: CostRegularizer[State, Action, Cost, Observation, RegularizedCost]

    def __call__(self, state: State, action: Action) -> Tuple[State, RegularizedCost, Observation]:
        state, cost, observation = self.dynamics(state, action)
        regularized_cost = self.cost_regularizer(state, action, cost, observation)
        return state, regularized_cost, observation
