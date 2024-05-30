"""Classes and methods to easily vectorize jax tensors"""

from __future__ import annotations

from functools import wraps
from typing import Callable, Hashable, Optional, Protocol, TypeVar, runtime_checkable

import jax_dataclasses as jdc
import optax
from jax import vmap
from typing_extensions import Self

from jax_utils.compilation import (
    BaseJaxCompilable,
    JaxCompilableProtocol,
    jit_when_compilation_enabled,
)
from jax_utils.pytree import ConvertibleToAxes, pytree_to_axes
from jax_utils.jax_tensor import AverageableJaxArrayLike
from jax_utils.typing import DataclassInstance

State = TypeVar("State")
Action_contra = TypeVar("Action_contra", contravariant=True)
Action = TypeVar("Action")
Observation_co = TypeVar("Observation_co", covariant=True)
Cost_co = TypeVar("Cost_co", covariant=True, bound=AverageableJaxArrayLike)
OptimizerState = TypeVar("OptimizerState", bound=optax.OptState)
AxisType = TypeVar("AxisType", bound=Hashable)
AxisType_contra = TypeVar("AxisType_contra", contravariant=True, bound=Hashable)


@runtime_checkable
class JaxVectorizableProtocol(JaxCompilableProtocol, ConvertibleToAxes[AxisType], Protocol):
    """Interface for classes manipulating JAX arrays. It defines a special attribute ``vectorized_axis``
    corresponding to the "named" axis over which to apply
    `vectorization <https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html>`_.

    "Vectorizable" classes are also "compilable" (see interface
    :class:`jax_utils.compilation.JaxCompilableProtocol`) and convertible to axes (see
    :class:`jax_utils.pytree.ConvertibleToAxes` interface).

    Args:
        vectorized_axis (AxisType): "named" axis to be vectorized
    """

    vectorized_axis: AxisType


def vectorize(
    in_default_axis: Optional[int] = None,
    out_default_axis: Optional[int] = -1,
) -> Callable[[Callable], Callable]:
    """Parametrized decorator for methods of classes implementing :class:`JaxVectorizableProtocol`
    interface. It allows to
    `vectorize <https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html>`_
    the decorated methods according to a specified axis, namely attribute
    :attr:`JaxVectorizableProtocol.vectorized_axis` (see :class:`JaxVectorizableProtocol`).

    The `pytrees <https://jax.readthedocs.io/en/latest/pytrees.html>`_ of all inputs and outputs of the decorated
    method are explored and every JAX array as well as every class implementing the
    :class:`jax_utils.pytree.ConvertibleToAxes` interface is considered a leaf of the pytree.
    When a :class:`jax_utils.pytree.ConvertibleToAxes` leaf  is encountered, method
    :meth:`jax_utils.pytree.ConvertibleToAxes.convert_to_axes`
    is used to determine the dimension to vectorize. When a JAX array leaf or any other leaf is encountered instead,
    the default value ``in_default_axis`` (for input arguments) or ``out_default_axis`` (for output arguments) is used
    to determine the dimension to vectorize.

    Args:
        in_default_axis (Optional[int], optional): default index used in ``in_axes`` argument of method
            `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`_
            when a leaf that is not :class:`jax_utils.pytree.ConvertibleToAxes` is encountered
            in an input pytree. Defaults to None.
        out_default_axis (Optional[int], optional): default index used in ``out_axes`` argument of method
            `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`_ when a leaf that is not
            :class:`jax_utils.pytree.ConvertibleToAxes` is encountered in an output pytree.
            Defaults to -1.

    Returns:
        Callable[[Callable], Callable]: decorator with parameters ``in_default_axis`` and ``out_default_axis`` specified
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: JaxVectorizableProtocol, *args, **kwargs):
            if not hasattr(self, "_vmap_functions"):
                object.__setattr__(self, "_vmap_functions", {})

            vmap_functions = getattr(self, "_vmap_functions")

            in_axes = pytree_to_axes(
                (self,) + tuple(args) + tuple(kwargs.values()),
                self.vectorized_axis,
                default_axis=in_default_axis,
            )

            if in_axes not in vmap_functions:
                outputs = func(self, *args, **kwargs)
                out_axes = pytree_to_axes(
                    outputs, self.vectorized_axis, default_axis=out_default_axis
                )

                vmap_functions[in_axes] = vmap(func, in_axes=in_axes, out_axes=out_axes)

            return vmap_functions[in_axes](self, *args, **kwargs)

        return jit_when_compilation_enabled()(wrapper)

    return decorator


@runtime_checkable
class JaxDataclassNestedConvertibleToAxes(
    DataclassInstance,
    ConvertibleToAxes[AxisType_contra],
    BaseJaxCompilable,
    Protocol[AxisType_contra],
):
    """Interface for dataclasses that can be jit-compiled and are "convertible to axes", with a concrete implementation
    of ``convert_to_axes`` method that is specific to dataclasses.
    """

    def convert_to_axes(self, axis: Optional[AxisType_contra]) -> Self:
        """Concrete implementation of ``convert_to_axes`` for dataclasses.
        All the fields of the dataclass are inspected and method ``convert_to_axes``
        is applied whenever the field is ``ConvertibleToAxes``.

        Args:
            axis (Optional[AxisType_contra]): a "named" axis

        Returns:
            Self: same object as ``self`` but with array-like fields converted to axes
        """
        with jdc.copy_and_mutate(self, validate=False) as dc:
            for field in jdc.fields(self):
                field_value = getattr(self, field.name)
                if isinstance(field_value, ConvertibleToAxes):
                    setattr(dc, field.name, field_value.convert_to_axes(axis))
        return dc
