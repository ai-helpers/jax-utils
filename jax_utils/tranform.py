"""Jax arrays transformations (scaling, ...)"""

from __future__ import annotations

from typing import Any, Generic, Hashable, Set, Tuple, Type, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
from jax.tree_util import tree_map

from jax_utils.compilation import BaseJaxCompilable, jit_when_compilation_enabled
from jax_utils.jax_tensor import JaxTensor
from jax_utils.common_tensor import TensorAxes, expand_dims_axis

AxisType = TypeVar("AxisType", bound=Hashable)


def scale_jax_tensor(
    tensor: JaxTensor[AxisType], scaling_factor: jnp.ndarray
) -> JaxTensor[AxisType]:
    """Scale the ``values`` of a ``JaxTensor`` i.e., multiply ``values`` by ``scaling_factor``.
    Note that property ``values`` in ``JaxTensor`` differ from attribute ``array``.

    Args:
        tensor (JaxTensor[AxisType]): jax tensor to be scaled
        scaling_factor (jnp.ndarray): the scaling factors to apply for scaling

    Returns:
        JaxTensor[AxisType]: scaled jax tensor
    """
    with jdc.copy_and_mutate(tensor, validate=True) as scaled_tensor:
        scaled_tensor.array = scaled_tensor.reverse_values(scaled_tensor.values * scaling_factor)
    return scaled_tensor


def unscale_jax_tensor(
    scaled_tensor: JaxTensor[AxisType], scaling_factor: jnp.ndarray
) -> JaxTensor[AxisType]:
    """Reverse transformation of function ``scale_jax_tensor``.

    Args:
        tensor (JaxTensor[AxisType]): jax tensor to be unscaled
        scaling_factor (jnp.ndarray): the scaling factors applied when scaling the original jax tensor

    Returns:
        JaxTensor[AxisType]: unscaled jax tensor
    """
    with jdc.copy_and_mutate(scaled_tensor, validate=True) as tensor:
        tensor.array = scaled_tensor.reverse_values(tensor.values / scaling_factor)
    return tensor


@jdc.pytree_dataclass(frozen=True)
class JaxScaler(BaseJaxCompilable, Generic[AxisType]):
    """When applying gradient descent optimization algorithms, it is often helpful to scale all the arrays/tensors
    involved in the computations.

    This class allows to easily scale/unscale :class:`jax_utils.jax_tensor.JaxTensor`'s (even when
    the tensors are stored in a nested pytree structure).

    Args:
        scaling_factors (jnp.ndarray): Multiplicative factors for scaling. Default to ``jnp.array(1.0)``.
        scaling_axes (jdc.Static[Set[AxisType]]): "Named" axes of ``scaling_factors``. Thus, the number of axes should
            match ``scaling_factors.ndim``. Default to ``jdc.field(default_factory=set)``.
        tensor_types_to_scale (jdc.Static[Tuple[Type[JaxTensor]]]): Object types that should be scaled in a pytree.
            Default to :class:`jax_utils.jax_tensor.JaxTensor`.
        tensor_types_not_to_scale (jdc.Static[Tuple[JaxTensor]]): Object types that should not be scaled in a pytree.
            Default to ``tuple()``.
    """

    scaling_factors: jnp.ndarray = jnp.array(1.0)
    scaling_axes: jdc.Static[Set[AxisType]] = jdc.field(default_factory=set)
    tensor_types_to_scale: jdc.Static[Tuple[Type[JaxTensor]]] = (JaxTensor,)
    tensor_types_not_to_scale: jdc.Static[Tuple[JaxTensor]] = tuple()  # type: ignore[assignment]

    def __post_init__(self):
        if self.scaling_factors.ndim != len(self.scaling_axes):
            raise ValueError(
                "the number of scaling_axes should math the number of dimensions of scaling_factors"
            )
        if len(set(self.tensor_types_to_scale) & set(self.tensor_types_not_to_scale)) > 0:
            raise ValueError(
                "tensor_types_to_scale and tensor_types_not_to_scale should be disjoint"
            )

    @classmethod
    def from_tensor(
        cls,
        tensor: JaxTensor[AxisType],
        scaling_axes: Set[AxisType],
        tensor_types_to_scale: jdc.Static[Tuple[Type[JaxTensor]]] = (JaxTensor,),
        tensor_types_not_to_scale: jdc.Static[Tuple[JaxTensor]] = tuple(),  # type: ignore[assignment]
        factor: float = 1.0,
    ) -> JaxScaler:
        """Alternative constructor that automatically computes the ``scaling_factors`` based on a
        :class:`jax_utils.jax_tensor.JaxTensor`, the ``scaling_axes``  and a target ``factor``.

        If ``my_scaler = JaxScaler.from_tensor(tensor=my_tensor, scaling_axes=my_axes, factor=my_factor)`` then
        ``my_scaler.scale(my_tensor)`` will be a :class:`jax_utils.jax_tensor.JaxTensor` with
        ``my_tensor.mean_over_axes(axes=my_axes)`` being equal to a "constant" Jax array with all elements equal to
        ``my_factor``.

        Args:
            tensor (JaxTensor[AxisType]): a jax tensor
            scaling_axes (Set[AxisType]): a set of "named" axes present in `tensor`
            tensor_types_to_scale (jdc.Static[Tuple[Type[JaxTensor]]]): Object types that should be scaled in a pytree.
                Default to :class:`jax_utils.jax_tensor.JaxTensor`.
            tensor_types_not_to_scale (jdc.Static[Tuple[JaxTensor]]): Object types that should not be scaled in a
                pytree. Default to ``tuple()``.

        Returns:
            JaxScaler: a scaler instance

        """
        axes_to_mean_over = tensor.actual_axes - scaling_axes
        return cls(
            scaling_factors=factor / jnp.maximum(tensor.mean_over_axes(axes=axes_to_mean_over), 1),
            scaling_axes=scaling_axes,
            tensor_types_to_scale=tensor_types_to_scale,
            tensor_types_not_to_scale=tensor_types_not_to_scale,
        )

    @jit_when_compilation_enabled()
    def scale(self, pytree: Any) -> Any:
        """Scales any pytree containing :class:`jax_utils.jax_tensor.JaxTensor`'s

        Args:
            pytree (Any): pytree to be scaled

        Returns:
            Any: the scaled pytree
        """
        return tree_map(
            lambda x: scale_jax_tensor(
                x,
                self.expand_scaling_factors(
                    tensor_axes=x.actual_axes, missing_axes=x.actual_axes - self.scaling_axes
                ),
            )
            if (
                isinstance(x, self.tensor_types_to_scale)
                and not isinstance(x, self.tensor_types_not_to_scale)  # type: ignore[arg-type]
            )
            else x,
            pytree,
            is_leaf=lambda x: isinstance(x, JaxTensor),
        )

    @jit_when_compilation_enabled()
    def unscale(self, scaled_pytree: Any) -> Any:
        """Unscales any pytree containing :class:`jax_utils.jax_tensor.JaxTensor`'s (reverse
        transformation of method ``scale``).

        Args:
            scaled_pytree (Any): pytree to be unscaled

        Returns:
            Any: the unscaled pytree
        """
        return tree_map(
            lambda x: unscale_jax_tensor(
                x,
                self.expand_scaling_factors(
                    tensor_axes=x.actual_axes, missing_axes=x.actual_axes - self.scaling_axes
                ),
            )
            if (
                isinstance(x, self.tensor_types_to_scale)
                and not isinstance(x, self.tensor_types_not_to_scale)  # type: ignore[arg-type]
            )
            else x,
            scaled_pytree,
            is_leaf=lambda x: isinstance(x, JaxTensor),
        )

    @jit_when_compilation_enabled()
    def expand_scaling_factors(
        self, tensor_axes: TensorAxes[AxisType], missing_axes: Set[AxisType]
    ) -> jnp.ndarray:
        """Expands the dimensions of ``self.scaling_factors`` to match the ``tensor_axes`` given as inputs.

        This method only adds new dimensions of size 1.

        Args:
            tensor_axes (TensorAxes[AxisType]): a set of "named" axes
            missing_axes (Set[AxisType]):

        Returns:
            jnp.ndarray: same as ``self.scaling_factors`` but with (empty) additional dimensions.
                The total number of dimensions of the returned array is equal to ``len(tensor_axes)``.
        """
        return jnp.expand_dims(
            self.scaling_factors,
            axis=expand_dims_axis(tensor_axes=tensor_axes, missing_axes=missing_axes),
        )
