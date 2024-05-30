"""High-level abstractions for decision problems (Markov Decision processes, etc...)
involving jax arrays transformations"""

from typing import Hashable, Protocol, Tuple, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
import optax
from jax import value_and_grad

from jax_utils.compilation import jit_when_compilation_enabled
from jax_utils.jax_tensor import AverageableJaxArrayLike
from jax_utils.vectorization import JaxDataclassNestedConvertibleToAxes, vectorize
from jax_utils.markov_decision_process import CostRegularizer, Dynamics, RegularizedDynamics

State = TypeVar("State")
State_contra = TypeVar("State_contra", contravariant=True)
Action_contra = TypeVar("Action_contra", contravariant=True)
Action = TypeVar("Action")
Observation = TypeVar("Observation")
Observation_co = TypeVar("Observation_co", covariant=True)
Cost = TypeVar("Cost", bound=AverageableJaxArrayLike)
Cost_co = TypeVar("Cost_co", covariant=True, bound=AverageableJaxArrayLike)
RegularizedCost = TypeVar("RegularizedCost")
OptimizerState = TypeVar("OptimizerState", bound=optax.OptState)
GradientUpdates = TypeVar("GradientUpdates", bound=optax.Updates)
AxisType = TypeVar("AxisType", bound=Hashable)
AxisType_contra = TypeVar("AxisType_contra", contravariant=True, bound=Hashable)


class JaxDynamics(
    JaxDataclassNestedConvertibleToAxes[AxisType_contra],
    Dynamics[State, Action_contra, Cost_co, Observation_co],
    Protocol[AxisType_contra, State, Action_contra, Cost_co, Observation_co],
):
    """A :class:`jax_utils.markov_decision_process.Dynamics` involving Jax arrays transformations. The cost of such
    dynamics can be differentiated w.r.t. the action (see method
    :meth:`jax_utils.dynamics.JaxDynamics.compute_gradient`).
    """

    @jit_when_compilation_enabled()
    def scalar_cost(self, state: State, action: Action_contra) -> jnp.ndarray:
        """Averages the cost associated to the dynamics of an MDP.

        Args:
            state (State): a state of the MDP
            action (Action_contra): an action of the MDP

        Returns:
            jnp.ndarray: a scalar Jax array corresponding to the mean cost
        """
        return self(state, action)[1].mean()

    @jit_when_compilation_enabled()
    def compute_gradient(
        self, state: State, action: Action_contra
    ) -> Tuple[Cost_co, Action_contra]:
        """The cost of a :class:`jax_utils.dynamics.JaxDynamics` can always be differentiated
        w.r.t. the action.

        Args:
            state (State): a state of the MDP
            action (Action_contra): an action of the MDP

        Returns:
            Tuple[Cost_co, Action_contra]: the associated cost and the gradient of the action
        """
        return value_and_grad(self.scalar_cost, argnums=1)(state, action)


@jdc.pytree_dataclass(frozen=True)
class VectorizedJaxDynamics(JaxDynamics[AxisType, State, Action_contra, Cost_co, Observation_co]):
    """A :class:`jax_utils.dynamics.JaxDynamics` that can be
    `vectorized <https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html>`_ over some "named" axis.

    Args:
        dynamics (JaxDynamics[AxisType, State, Action_contra, Cost_co, Observation_co]): the
            :class:`jax_utils.dynamics.JaxDynamics` to be vectorized
        vectorized_axis (jdc.Static[AxisType]): "named" axis to map over
    """

    dynamics: JaxDynamics[AxisType, State, Action_contra, Cost_co, Observation_co]
    vectorized_axis: jdc.Static[AxisType]

    @vectorize()
    def __call__(
        self, state: State, action: Action_contra
    ) -> Tuple[State, Cost_co, Observation_co]:
        return self.dynamics(state, action)


@jdc.pytree_dataclass(frozen=True)
class RegularizedJaxDynamics(
    RegularizedDynamics[State, Action, Cost, RegularizedCost, Observation],
    JaxDynamics[AxisType_contra, State, Action, Cost, Observation],
):
    """A :class:`jax_utils.dynamics.JaxDynamics` with cost regularization.
    See also :class:`jax_utils.markov_decision_process.RegularizedDynamics`.

    Args:
        dynamics (JaxDynamics[AxisType_contra, State, Action, Cost, Observation]): the
            :class:`jax_utils.dynamics.JaxDynamics` to be vectorized
        cost_regularizer (CostRegularizer[State, Action, Cost, Observation, RegularizedCost]): a cost regularizer
    """

    dynamics: JaxDynamics[AxisType_contra, State, Action, Cost, Observation]
    cost_regularizer: CostRegularizer[State, Action, Cost, Observation, RegularizedCost]

    @jit_when_compilation_enabled()
    def __call__(self, state: State, action: Action) -> Tuple[State, RegularizedCost, Observation]:
        return super().__call__(state, action)
