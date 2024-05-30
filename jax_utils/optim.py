"""Classes and methods to optimize functions involving jax arrays transformations via gradient descent"""

from __future__ import annotations

import functools
from collections import UserList, deque
from dataclasses import dataclass, replace
from typing import Deque, Generic, Hashable, List, Optional, Protocol, Tuple, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
import optax
from matplotlib import pyplot as plt
from tqdm.autonotebook import trange
from typing_extensions import Self

from jax_utils.compilation import BaseJaxCompilable, jit_when_compilation_enabled
from jax_utils.gradient import BaseGradientStep
from jax_utils.jax_tensor import AverageableJaxArrayLike, JaxTensor

State = TypeVar("State")
Action = TypeVar("Action")
Observation = TypeVar("Observation")
Cost = TypeVar("Cost", bound=AverageableJaxArrayLike)
JaxTensorType = TypeVar("JaxTensorType", bound=JaxTensor)
OptimizerState = TypeVar("OptimizerState", bound=optax.OptState)
AxisType = TypeVar("AxisType", bound=Hashable)


@dataclass(frozen=True)
class OptimizationState(Generic[State, Action, Cost, OptimizerState]):
    """The current state of an iterative optimization procedure involving the
    :class:`jax_utils.markov_decision_process.Dynamics`
    of a Markov Decision Process (typically a cost minimization).

    Args:
        iteration (int): the current iteration step
        state (State): the MDP state
        action (Action): the MDP current action
        cost (Optional[Cost], optional): the current cost associated to the state-action pair. Defaults to None.
        optimizer_state (Optional[OptimizerState], optional): The current ``optax.OptState``. Defaults to None.
    """

    iteration: int
    state: State
    action: Action
    cost: Optional[Cost] = None
    optimizer_state: Optional[OptimizerState] = None

    @property
    def scalar_cost(self) -> float:
        """Returns the average ``self.cost``

        Returns:
            float: average cost
        """
        return float("inf") if self.cost is None else float(self.cost.mean())


class OptimStoppingCondition(BaseJaxCompilable, Protocol[State, Action, Cost, OptimizerState]):
    """Interface for all stopping conditions of an iterative optimization procedure involving the
    :class:`jax_utils.markov_decision_process.Dynamics` of a Markov Decision Process (typically a cost minimization).

    The stopping condition depends on an ``OptimizationState`` provided as input but may
    also collect information over steps.
    """

    def stop(
        self, optimization_state: OptimizationState[State, Action, Cost, OptimizerState]
    ) -> bool:
        """Returns a boolean inidicating whether the optimization procedure should be stopped.
        This class should be overriven in every concrete class.

        Args:
            optimization_state (OptimizationState[State, Action, Cost, OptimizerState]): _description_

        Returns:
            bool: `True`` if the optimization procedure should be stopped, `False`` otherwise.
        """

    def reset(self) -> Self:
        """Resets the stopping condition to an initial configuration."""
        return self

    @property
    def nb_iterations_upper_bound(self) -> int:
        """An upper bound on the maximal number of iterations. Default is 1e20."""
        return int(1e20)

    def __and__(
        self, other_stopping_condition: OptimStoppingCondition[State, Action, Cost, OptimizerState]
    ) -> OptimStoppingConditionIntersection:
        """Constructs a stopping condition that stops if and only if both ``self`` AND ``other_stopping_condition``
        stop.

        Args:
            other_stopping_condition (OptimStoppingCondition[State, Action, Cost, OptimizerState]): an other
                stopping condition

        Returns:
            OptimStoppingConditionIntersection: intersection of two stopping conditions
        """
        return OptimStoppingConditionIntersection(
            stopping_condition_1=self, stopping_condition_2=other_stopping_condition
        )

    def __or__(
        self, other_stopping_condition: OptimStoppingCondition[State, Action, Cost, OptimizerState]
    ) -> OptimStoppingConditionUnion:
        """Constructs a stopping condition that stops if and only if any of ``self`` OR ``other_stopping_condition``
        stops.

        Args:
            other_stopping_condition (OptimStoppingCondition[State, Action, Cost, OptimizerState]): an other
                stopping condition

        Returns:
            OptimStoppingConditionUnion: union of two stopping conditions
        """
        return OptimStoppingConditionUnion(
            stopping_condition_1=self, stopping_condition_2=other_stopping_condition
        )


class OptimStoppingConditionsCombination(
    OptimStoppingCondition[State, Action, Cost, OptimizerState], Protocol
):
    """This interface represents the combination of 2 stopping conditions.

    Args:
        stopping_condition_1 (OptimStoppingCondition[State, Action, Cost, OptimizerState]): a stopping condition
        stopping_condition_2 (OptimStoppingCondition[State, Action, Cost, OptimizerState]): a stopping condition
    """

    stopping_condition_1: OptimStoppingCondition[State, Action, Cost, OptimizerState]
    stopping_condition_2: OptimStoppingCondition[State, Action, Cost, OptimizerState]

    def enable_compilation(self) -> Self:
        self.stopping_condition_1.enable_compilation()
        self.stopping_condition_2.enable_compilation()
        return super().enable_compilation()

    def disable_compilation(self) -> Self:
        self.stopping_condition_1.disable_compilation()
        self.stopping_condition_2.disable_compilation()
        return super().disable_compilation()


@dataclass(frozen=True)
class OptimStoppingConditionIntersection(
    OptimStoppingConditionsCombination[State, Action, Cost, OptimizerState]
):
    """A combination of 2 stopping conditions using an "AND" operation.

    Args:
        stopping_condition_1 (OptimStoppingCondition[State, Action, Cost, OptimizerState]): a stopping condition
        stopping_condition_2 (OptimStoppingCondition[State, Action, Cost, OptimizerState]): a stopping condition
    """

    stopping_condition_1: OptimStoppingCondition[State, Action, Cost, OptimizerState]
    stopping_condition_2: OptimStoppingCondition[State, Action, Cost, OptimizerState]

    def stop(
        self, optimization_state: OptimizationState[State, Action, Cost, OptimizerState]
    ) -> bool:
        return self.stopping_condition_1.stop(
            optimization_state
        ) and self.stopping_condition_2.stop(optimization_state)

    def reset(self) -> OptimStoppingConditionIntersection[State, Action, Cost, OptimizerState]:
        self.stopping_condition_1.reset()
        self.stopping_condition_2.reset()
        return self

    def __repr__(self) -> str:
        return f"({repr(self.stopping_condition_1)} & {repr(self.stopping_condition_2)})"

    @property
    def nb_iterations_upper_bound(self) -> int:
        return max(
            self.stopping_condition_1.nb_iterations_upper_bound,
            self.stopping_condition_2.nb_iterations_upper_bound,
        )


@dataclass(frozen=True)
class OptimStoppingConditionUnion(
    OptimStoppingConditionsCombination[State, Action, Cost, OptimizerState]
):
    """A combination of 2 stopping conditions using an "OR" operation.

    Args:
        stopping_condition_1 (OptimStoppingCondition[State, Action, Cost, OptimizerState]): a stopping condition
        stopping_condition_2 (OptimStoppingCondition[State, Action, Cost, OptimizerState]): a stopping condition
    """

    stopping_condition_1: OptimStoppingCondition[State, Action, Cost, OptimizerState]
    stopping_condition_2: OptimStoppingCondition[State, Action, Cost, OptimizerState]

    def stop(
        self, optimization_state: OptimizationState[State, Action, Cost, OptimizerState]
    ) -> bool:
        return self.stopping_condition_1.stop(optimization_state) or self.stopping_condition_2.stop(
            optimization_state
        )

    def reset(self) -> OptimStoppingConditionUnion[State, Action, Cost, OptimizerState]:
        self.stopping_condition_1.reset()
        self.stopping_condition_2.reset()
        return self

    def __repr__(self) -> str:
        return f"({repr(self.stopping_condition_1)} | {repr(self.stopping_condition_2)})"

    @property
    def nb_iterations_upper_bound(self) -> int:
        return min(
            self.stopping_condition_1.nb_iterations_upper_bound,
            self.stopping_condition_2.nb_iterations_upper_bound,
        )


@dataclass(frozen=True)
class MaxIterationsStoppingCondition(OptimStoppingCondition[State, Action, Cost, OptimizerState]):
    """A stopping condition that stops when the number of iterations exceeds a given threshold.

    Args:
        max_iterations (int): maximal number of iterations before the stopping condition is raised
    """

    max_iterations: int

    def stop(
        self, optimization_state: OptimizationState[State, Action, Cost, OptimizerState]
    ) -> bool:
        if optimization_state.iteration >= self.max_iterations:
            print(f"Maximum number of {self.max_iterations} iterations reached.")
            return True
        return False

    def __repr__(self) -> str:
        return f"MaxIterations({self.max_iterations})"

    @property
    def nb_iterations_upper_bound(self) -> int:
        return self.max_iterations


@dataclass(frozen=True)
class MinIterationsStoppingCondition(OptimStoppingCondition[State, Action, Cost, OptimizerState]):
    """A stopping condition that continues as long as the number of iterations is below a given threshold.

    Args:
        min_iterations (int): minimal number of iterations before the stopping condition is raised
    """

    min_iterations: int

    def stop(
        self, optimization_state: OptimizationState[State, Action, Cost, OptimizerState]
    ) -> bool:
        if optimization_state.iteration <= self.min_iterations:
            return False
        return True

    def __repr__(self) -> str:
        return f"MinIterations({self.min_iterations})"


@jdc.pytree_dataclass
class MinDeltaActionStoppingCondition(
    OptimStoppingCondition[State, JaxTensorType, Cost, OptimizerState]
):
    """Stops when the action stops significantly changing (as defined by relative & absolute tolerance).
    The previous action is saved in memory to compare it to the new action.

    The delta between previous and new action is then compared to absolute and relative tolerance i.e, the
    stopping condition is raised when:
    ``all(abs(new_action - previous_action) <= absolute tolerance + relative_tolerance * maximum(abs(new_action),
    abs(previous_action)))``

    Args:
        relative_tolerance (jnp.ndarray): relative tolerance for action variations. Default is 1e-6.
        absolute_tolerance (jnp.ndarray): absolute tolerance for action variations. Default is 1e-6.
    """

    relative_tolerance: jnp.ndarray = jnp.array(1e-6)
    absolute_tolerance: jnp.ndarray = jnp.array(1e-6)

    @jit_when_compilation_enabled()
    def _stop(self, previous_action: JaxTensorType, new_action: JaxTensorType) -> jnp.ndarray:
        delta_action = jnp.abs(new_action.values - previous_action.values)
        return jnp.all(
            delta_action
            <= self.absolute_tolerance
            + self.relative_tolerance
            * jnp.maximum(jnp.abs(previous_action.values), jnp.abs(new_action.values))
        )

    def stop(
        self, optimization_state: OptimizationState[State, JaxTensorType, Cost, OptimizerState]
    ) -> bool:
        new_action = optimization_state.action
        try:
            previous_action = getattr(self, "_previous_action")
            should_stop = self._stop(previous_action, new_action)
        except AttributeError:
            should_stop = False
        object.__setattr__(self, "_previous_action", new_action)
        if should_stop:
            print("Changes in action are below specified (absolute + relative) tolerance.")
        return should_stop

    def reset(self) -> MinDeltaActionStoppingCondition[State, JaxTensorType, Cost, OptimizerState]:
        if hasattr(self, "_previous_action"):
            object.__delattr__(self, "_previous_action")
        return self


@jdc.pytree_dataclass
class MinDeltaCostStoppingCondition(
    OptimStoppingCondition[State, Action, jnp.ndarray, OptimizerState]
):
    """Stops when the cost stops significantly decreasing (as defined by relative & absolute tolerance).

    The ``window_length`` last cost values are saved in memory and the stopping condition is raised when the delta
    between the min and max of these values is below a threshold i.e., when:
    ``all(maximum(last_window_length_costs) - minimum(last_window_length_costs) < absolute_tolerance +
    relative_tolerance * minimum(last_window_length_costs))``

    Args:
        relative_tolerance (jnp.ndarray): relative tolerance for cost variations. Default is 1e-6.
        absolute_tolerance (jnp.ndarray): absolute tolerance for cost variations. Default is 1e-6.
        window_length (int): maximal length of the queue storing the past history of costs (costs older than
            ``window_length`` time steps are discarded)
    """

    relative_tolerance: jnp.ndarray = jnp.array(1e-6)
    absolute_tolerance: jnp.ndarray = jnp.array(1e-6)
    window_length: int = 2

    @jit_when_compilation_enabled()
    def _stop(self, costs_queue: List[jnp.ndarray]) -> jnp.ndarray:
        max_cost = functools.reduce(lambda x, y: jnp.maximum(x, y), costs_queue)
        min_cost = functools.reduce(lambda x, y: jnp.minimum(x, y), costs_queue)
        delta_cost = max_cost - min_cost
        return jnp.all(
            delta_cost
            <= self.relative_tolerance * jnp.minimum(jnp.abs(max_cost), jnp.abs(min_cost))
            + self.absolute_tolerance
        )

    def stop(
        self, optimization_state: OptimizationState[State, Action, jnp.ndarray, OptimizerState]
    ) -> bool:
        try:
            if not hasattr(self, "_costs_queue"):
                return False
            costs_queue: Deque[jnp.ndarray] = getattr(self, "_costs_queue")
            if len(costs_queue) < self.window_length:
                return False
            should_stop = self._stop(list(costs_queue))
            if should_stop:
                print("Cost decrease is below specified (absolute + relative) tolerance.")
                return True
            return False
        finally:
            if not hasattr(self, "_costs_queue"):
                object.__setattr__(self, "_costs_queue", deque())
            costs_queue: Deque[jnp.ndarray] = getattr(self, "_costs_queue")  # type: ignore[no-redef]
            if optimization_state.cost is not None:
                costs_queue.append(optimization_state.cost)
            if len(costs_queue) > self.window_length:
                costs_queue.popleft()

    def reset(self) -> MinDeltaCostStoppingCondition[State, Action, OptimizerState]:
        if hasattr(self, "_costs_queue"):
            object.__delattr__(self, "_costs_queue")
        return self


class CostHistory(UserList[AverageableJaxArrayLike]):
    """List of costs (typically jax arrays)"""

    def scalar_costs(self) -> List[float]:
        """
        Returns:
            List[float]: list of cost means
        """
        return [float(cost.mean()) for cost in self.data]

    def plot_scalar_costs(self):
        """Plots cost history"""
        plt.plot(self.scalar_costs())
        plt.title("Cost evolution over time")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")


@dataclass(frozen=True)
class GradientDescentOptimizationLoop(
    Generic[AxisType, State, Action, Cost, Observation, OptimizerState]
):
    """Gradient descent optimization procedure involving the :class:`jax_utils.markov_decision_process.Dynamics`
    of a Markov Decision Process. This class is a callable that recursively applies ``gradient_step`` until
    a ``stopping_condition`` is met.

    Args:
        gradient_step (BaseGradientStep[AxisType, State, Action, Cost, OptimizerState]): a gradient step
        stopping_condition (OptimStoppingCondition[State, Action, Cost, OptimizerState]): a stopping condition
    """

    gradient_step: BaseGradientStep[AxisType, State, Action, Cost, OptimizerState]
    stopping_condition: OptimStoppingCondition[State, Action, Cost, OptimizerState]

    def __call__(
        self,
        state: State,
        init_action: Action,
        iteration: int = 0,
        cost: Optional[Cost] = None,
        opt_state: Optional[OptimizerState] = None,
    ) -> Tuple[
        OptimizationState[State, Action, Cost, OptimizerState],
        OptimizationState[State, Action, Cost, OptimizerState],
        CostHistory,
    ]:
        """Iteratively minimizes the cost of the MDP dynamics (in some state) by performing gradient descent
        on the action.

        Args:
            state (State): state of the MDP (does not change during optimization)
            init_action (Action): initial action that will then be optimized
            iteration (int, optional): current  iteration. Defaults to 0.
            cost (Optional[Cost], optional): current cost. Defaults to None.
            opt_state (Optional[OptimizerState], optional): current optimization state. Defaults to None.

        Returns:
            Tuple[ OptimizationState[State, Action, Cost, OptimizerState], OptimizationState[State, Action, Cost, OptimizerState], CostHistory,]: the optimal `OptimizationState`, final `OptimizationState` and history o cost during the optimization iterations  # pylint: disable=line-too-long

        """
        # Initialization
        # --------------------
        cost_history = CostHistory()
        opt_state = (
            self.gradient_step.init_optimizer(init_action) if opt_state is None else opt_state
        )
        action = init_action

        optimization_state = OptimizationState[State, Action, Cost, OptimizerState](
            iteration=iteration, state=state, action=action, cost=cost, optimizer_state=opt_state
        )
        optimal_optimization_state = replace(optimization_state)

        # Optim loop
        # --------------------
        pbar = trange(self.stopping_condition.nb_iterations_upper_bound)
        for it in pbar:
            action, opt_state, cost_value = self.gradient_step(
                state,
                action,
                opt_state,
            )
            cost_history.append(cost_value)
            optimization_state = replace(
                optimal_optimization_state,
                iteration=it,
                action=action,
                cost=cost_value,
                optimizer_state=opt_state,
            )
            scalar_cost = optimization_state.scalar_cost
            pbar.set_postfix({"cost": scalar_cost})

            if scalar_cost < optimal_optimization_state.scalar_cost:
                optimal_optimization_state = optimization_state

            if self.stopping_condition.stop(optimization_state):
                break

        return optimal_optimization_state, optimization_state, cost_history

    def resume(
        self, optimization_state: OptimizationState[State, Action, Cost, OptimizerState]
    ) -> Tuple[
        OptimizationState[State, Action, Cost, OptimizerState],
        OptimizationState[State, Action, Cost, OptimizerState],
        CostHistory,
    ]:
        """Resume optimization loop from ``optimization_state`` (cold start).

        Args:
            optimization_state (OptimizationState[State, Action, Cost, OptimizerState]): state from which to resume the
                optimization procedure.

        Returns:
            OptimizationState[State, Action, Cost, OptimizerState], OptimizationState[State, Action, Cost,OptimizerState], CostHistory,]: same outputs as :meth:`GradientDescentOptimizationLoop.__call__`  # pylint: disable=line-too-long
        """
        return self(
            state=optimization_state.state,
            init_action=optimization_state.action,
            iteration=optimization_state.iteration,
            cost=optimization_state.cost,
            opt_state=optimization_state.optimizer_state,
        )
