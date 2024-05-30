"""Wrappers around jax arrays"""

from numbers import Number
from typing import (
    AbstractSet,
    Generic,
    Hashable,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    runtime_checkable,
)

import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import nn
from typing_extensions import Self

from jax_utils.compilation import BaseJaxCompilable, jit_when_compilation_enabled
from jax_utils.pytree import ConvertibleToAxes
from jax_utils.common_tensor import AverageableArrayLike, RegularizedArrayLikeCost, Tensor

AxisType = TypeVar("AxisType", bound=Hashable)


@runtime_checkable
class JaxTensor(Tensor[jnp.ndarray, AxisType], ConvertibleToAxes[AxisType], Protocol[AxisType]):
    """Interface representing a ``Tensor`` with a
    `JAX <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html>`_ ``array``.

    A ``JaxTensor`` is always "convertible to axes" (see
    :class:`jax_utils.pytree.ConvertibleToAxes` interface) using
    :meth:`jax_utils.pytree.ConvertibleToAxes.convert_to_axes` method. When a ``Tensor`` is
    "converted to axes", the  associated ``array`` attribute is no longer a JAX array but rather an integer indicating
    the dimension index of the specified ``axis`` (argument of method ``convert_to_axes``). If ``axis`` is
    flattened or not present, ``self.array`` is set to ``None`` when converted to axes.

    Converting a :class:`jax_utils.jax_tensor.JaxTensor` to axes is useful to determine the
    ``in_axes`` and ``out_axes`` arguments of
    `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`_ and
    `jax.pmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html>`_.

    Args:
        array (jnp.ndarray): a JAX array (or an optional integer when "converted to axes")
    """

    array: jnp.ndarray

    def __init__(self, array: jnp.ndarray, **kwargs):
        self.array = array
        super().__init__()

    def __post_init__(self):
        if isinstance(self.array, jnp.ndarray):
            # in case the class is instantiated as a PyTreeDef (e.g., for vmap),
            # self.array may be an int, None, ...
            super().__post_init__()

    def is_broadcastable_with(
        self,
        other_tensor_or_shape: Union[Tensor[jnp.ndarray, AxisType], Tuple[int, ...]],
    ) -> bool:
        if isinstance(self.array, jnp.ndarray):
            # in case the class is instantiated as a PyTreeDef (e.g., for vmap),
            # self.array may be an int, None, ...
            return super().is_broadcastable_with(other_tensor_or_shape)
        return True

    @classmethod
    def from_flattened_axes(
        cls, array: jnp.ndarray, flattened_axes: AbstractSet[AxisType], **kwargs
    ) -> Self:
        """
        Constructor when some axes are "flattened" in ``array``.

        This method simply adds missing dimensions accordingly.

        Args:
            array (jnp.ndarray): a JAX array
            flattened_axes (AbstractSet[AxisType]): the named axes that are "flattened"

        Returns:
            Self: an instance of class ``cls``
        """
        return cls(
            array=jnp.expand_dims(array, axis=cls._expand_dims_axis(flattened_axes)), **kwargs
        )

    def convert_to_axes(self, axis: Optional[AxisType]) -> Self:
        """Convert the :class:`jax_utils.jax_tensor.JaxTensor` object so that it can be passed as
        the ``in_axes`` or ``out_axes`` arguments of
        `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`_ and
        `jax.pmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html>`_.

        Args:
            axis (Optional[AxisType]): a "named" axis

        Returns:
            Self: same object as ``self`` but with ``array`` replaced by the index of ``axis`` i.e.,
                if ``axis`` corresponds to the ``i``-th dimension of ``array``, then ``array`` is replaced by ``i``,
                if ``axis`` is a flattened axis or simply not present, then ``array`` is replaced by ``None``
        """
        with jdc.copy_and_mutate(self, validate=False) as jax_tensor:
            if axis is not None and axis in self.actual_axes:
                jax_tensor.array = self.reverse_index(axis)  # type: ignore[assignment]
            else:
                jax_tensor.array = None  # type: ignore[assignment]
        return jax_tensor


class NonNegativeValues(Protocol):
    """Interface to be used in combination with :class:`jax_utils.jax_tensor.JaxTensor` in order
    to implement JAX tensors that are constrained to take non-negative ``values``.

    Args:
        array (jnp.ndarray): a JAX array
    """

    array: jnp.ndarray

    # pylint: disable=C0116
    @property
    def values(self) -> jnp.ndarray:
        return nn.elu(self.array - 1) + 1

    # pylint: disable=C0116
    def reverse_values(self, values: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(values > 1, values, jnp.log(values) + 1)


class NonNegativeBudgetedValues(NonNegativeValues, Protocol):
    """Interface to be used in combination with :class:`jax_utils.jax_tensor.JaxTensor` in order to
    implement JAX tensors that are constrained to both take non-negative ``values`` and have their
    sum over the last axis bounded by  a maximal "budget" (``max_budget``).

    Args:
        array (jnp.ndarray): a JAX array
        max_budget (Number): a non-negative number representing the maximum value that
            the sum over the last axis can take
    """

    array: jnp.ndarray
    max_budget: Number

    @property
    def values(self) -> jnp.ndarray:
        return (
            self.max_budget
            * super().values
            / (super().values.sum(axis=-1)[..., jnp.newaxis] + 1e-6)
        )[..., :-1]

    def reverse_values(self, values: jnp.ndarray) -> jnp.ndarray:
        output = jnp.zeros(self.array.shape).at[..., :-1].set(values)
        output = output.at[..., -1].set(jnp.maximum(self.max_budget - values.sum(axis=-1), 1e-6))
        return super().reverse_values(output)


AverageableJaxArrayLike: TypeAlias = AverageableArrayLike[jnp.ndarray]

JaxCostType = TypeVar("JaxCostType", bound=AverageableJaxArrayLike)
JaxRegularizedCostType = TypeVar("JaxRegularizedCostType", bound=AverageableJaxArrayLike)


@jdc.pytree_dataclass(frozen=True)
class RegularizedJaxCost(
    RegularizedArrayLikeCost[jnp.ndarray],
    BaseJaxCompilable,
    Generic[JaxCostType, JaxRegularizedCostType],
):
    """Interface for "regularized" JAX costs i.e., costs of the form:
    ``cost + lagrangian_coefficient * regularization``

    Args:
        cost (JaxCostType): an JAX array of costs
        regularization (JaxRegularizedCostType): a JAX array of regularization costs
        lagrangian_coefficient (jdc.Static[Number]): a non-negative number quantifying the
            regularization weight
    """

    cost: JaxCostType
    regularization: JaxRegularizedCostType
    lagrangian_coefficient: jdc.Static[Number] = 1  # type: ignore[assignment]

    @jit_when_compilation_enabled()
    def mean(self, *args, **kwargs) -> jnp.ndarray:
        return super().mean(*args, **kwargs)
