"""Classes and methods to easily map pytrees to axes (e.g., for vectorization)"""

from typing import Any, Hashable, Optional, Protocol, TypeVar, runtime_checkable

from jax.tree_util import tree_map
from typing_extensions import Self

AxisType = TypeVar("AxisType", bound=Hashable)
AxisType_contra = TypeVar("AxisType_contra", contravariant=True)


# pylint: disable=C0115
@runtime_checkable
class ConvertibleToAxes(Protocol[AxisType_contra]):
    def convert_to_axes(self, axis: Optional[AxisType_contra]) -> Self:
        """
        Returns an object that can be used in argument ``in_axes`` or ``out_axes`` of
        `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`_
        or `jax.pmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html>`_

        Args:
            axis (Optional[AxisType_contra]): a "named" axis over which to apply
                `vectorization <https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html>`_.

        Returns:
            Self: same object as ``self`` but with all array-like objects replaced by axes
        """


def pytree_to_axes(
    pytree: Any, vectorized_axis: AxisType, default_axis: Optional[int] = None
) -> Any:
    """
    Transform all the :class:`ConvertibleToAxes` leafs of a given
    `pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_ to axes by applying method
    :meth:`ConvertibleToAxes.convert_to_axes`. This is useful for
    `vectorizing <https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html>`_ functions
    involving :class:`ConvertibleToAxes` objects.

    Args:
        pytree (Any): any Python pytree containing :class:`ConvertibleToAxes` leafs
        vectorized_axis (AxisType): a "named" axis over which to apply
            `vectorization <https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html>`_
        default_axis (Optional[int], optional): A default axis value for leafs that are not :class:`ConvertibleToAxes`.
            Defaults to None.

    Returns:
        Any: same pytree as given in input but where all :class:`ConvertibleToAxes` are converted to axes.
    """
    return tree_map(
        lambda x: x.convert_to_axes(vectorized_axis)
        if isinstance(x, ConvertibleToAxes)
        else default_axis,
        pytree,
        is_leaf=lambda x: isinstance(x, ConvertibleToAxes),
    )
