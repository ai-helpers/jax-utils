"""Classes and methods to compute the gradient of functions involving jax arrays transformations"""

from __future__ import annotations

from typing import Any, Generic, Hashable, Protocol, Tuple, TypeVar

import jax_dataclasses as jdc
import optax

from jax_utils.compilation import jit_when_compilation_enabled
from jax_utils.dynamics import JaxDynamics
from jax_utils.jax_tensor import AverageableJaxArrayLike
from jax_utils.vectorization import JaxDataclassNestedConvertibleToAxes, vectorize

State = TypeVar("State")
State_contra = TypeVar("State_contra", contravariant=True)
Action = TypeVar("Action")
Observation_co = TypeVar("Observation_co", covariant=True)
Cost_co = TypeVar("Cost_co", covariant=True, bound=AverageableJaxArrayLike)
OptimizerState = TypeVar("OptimizerState", bound=optax.OptState)
GradientUpdates = TypeVar("GradientUpdates", bound=optax.Updates)
AxisType = TypeVar("AxisType", bound=Hashable)
AxisType_contra = TypeVar("AxisType_contra", contravariant=True, bound=Hashable)


class BaseGradientStep(
    JaxDataclassNestedConvertibleToAxes[AxisType_contra],
    Protocol[AxisType_contra, State_contra, Action, Cost_co, OptimizerState],
):
    """Interface for gradient steps on the  :class:`jax_utils.markov_decision_process.Dynamics` cost of a Markov
    Decision Process."""

    # pylint: disable=C0116
    def init_optimizer(self, init_action: Action) -> OptimizerState: ...

    # pylint: disable=C0116
    def compute_gradient(
        self,
        state: State_contra,
        action: Action,
    ) -> Tuple[Cost_co, Action]: ...

    # pylint: disable=C0116
    def update(
        self, action: Action, gradient_value: GradientUpdates, opt_state: OptimizerState
    ) -> Tuple[Action, OptimizerState]: ...

    def __call__(
        self,
        state: State_contra,
        action: Action,
        opt_state: OptimizerState,
    ) -> Tuple[Action, OptimizerState, Cost_co]: ...


@jdc.pytree_dataclass(frozen=True)
class GradientStep(
    BaseGradientStep[AxisType, State, Action, Cost_co, OptimizerState],
    Generic[AxisType, State, Action, Cost_co, Observation_co, OptimizerState],
):
    """The gradient step on the :class:`jax_utils.markov_decision_process.Dynamics` cos
    of a Markov Decision Process.

    Args:
        optimizer (jdc.Static[optax.GradientTransformation]): an optimizer (defining the
            stochastic gradient descent variant: RMSProp, Adam, etc..)
        dynamics (JaxDynamics[Any, State, Action, Cost_co, Observation_co]): an MDP dynamics
    """

    optimizer: jdc.Static[optax.GradientTransformation]
    dynamics: JaxDynamics[Any, State, Action, Cost_co, Observation_co]

    @jit_when_compilation_enabled()
    def init_optimizer(self, init_action: Action) -> OptimizerState:
        """Initialize optimizer

        Args:
            init_action (Action): initial action before optimization starts

        Returns:
            OptimizerState: initialized optimizer state
        """
        return self.optimizer.init(init_action)

    @jit_when_compilation_enabled()
    def compute_gradient(
        self,
        state: State,
        action: Action,
    ) -> Tuple[Cost_co, Action]:
        """Given a state-action pair of the MDP, computes the cost and gradient
        of the cost of a ``dynamics`` w.r.t. the action.

        Args:
            state (State): state of the MDP
            action (Action): action of the MDP

        Returns:
            Tuple[Cost_co, Action]: cost and gradient of the action
        """
        return self.dynamics.compute_gradient(
            state=state,
            action=action,
        )

    @jit_when_compilation_enabled()
    def update(
        self, action: Action, gradient_value: GradientUpdates, opt_state: OptimizerState
    ) -> Tuple[Action, OptimizerState]:
        """Update the action and optimizer state

        Args:
            action (Action): action of the MDP
            gradient_value (GradientUpdates): gradient of the action
            opt_state (OptimizerState): optimizer state

        Returns:
            Tuple[Action, OptimizerState]: the new action and optimizer state
        """
        updates, updated_opt_state = self.optimizer.update(gradient_value, opt_state, action)
        updated_action = optax.apply_updates(action, updates)
        return updated_action, updated_opt_state

    @jit_when_compilation_enabled()
    def __call__(
        self,
        state: State,
        action: Action,
        opt_state: OptimizerState,
    ) -> Tuple[Action, OptimizerState, Cost_co]:
        """Computes the gradient of the cost w.r.t. the action and updates the action and optimizer state.

        Args:
            state (State): state of the MDP
            action (Action): action of the MDP
            opt_state (OptimizerState): optimizer state

        Returns:
            Tuple[Action, OptimizerState, Cost_co]: a new action, optimizer state and the value of the cost
                associated to the previous action
        """
        cost_value, gradient_value = self.compute_gradient(
            state=state,
            action=action,
        )
        updated_action, updated_opt_state = self.update(action, gradient_value, opt_state)
        return updated_action, updated_opt_state, cost_value


@jdc.pytree_dataclass(frozen=True)
class VectorizedGradientStep(BaseGradientStep[AxisType, State, Action, Cost_co, OptimizerState]):
    """A gradient step that is `vectorized <https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html>`_ on a
    specific axis

    Args:
        gradient_step (BaseGradientStep[AxisType, State, Action, Cost_co, OptimizerState]): the gradient step to be
            vectorized
        vectorized_axis (jdc.Static[AxisType]): "named" axis to be vectorized
    """

    gradient_step: BaseGradientStep[AxisType, State, Action, Cost_co, OptimizerState]
    vectorized_axis: jdc.Static[AxisType]

    @jit_when_compilation_enabled()
    def init_optimizer(self, init_action: Action) -> OptimizerState:
        return self.gradient_step.init_optimizer(init_action)

    @vectorize()
    def compute_gradient(
        self,
        state: State,
        action: Action,
    ) -> Tuple[Cost_co, Action]:
        return self.gradient_step.compute_gradient(
            state,
            action,
        )

    @jit_when_compilation_enabled()
    def update(
        self, action: Action, gradient_value: GradientUpdates, opt_state: OptimizerState
    ) -> Tuple[Action, OptimizerState]:
        return self.gradient_step.update(action, gradient_value, opt_state)

    @jit_when_compilation_enabled()
    def __call__(
        self,
        state: State,
        action: Action,
        opt_state: OptimizerState,
    ) -> Tuple[Action, OptimizerState, Cost_co]:
        cost_value, gradient_value = self.compute_gradient(state, action)
        updated_action, updated_opt_state = self.update(action, gradient_value, opt_state)
        return updated_action, updated_opt_state, cost_value
