"""Wrappers around numpy/jax arrays"""

from __future__ import annotations

from numbers import Number
from typing import (
    AbstractSet,
    Any,
    ClassVar,
    Dict,
    Hashable,
    Iterable,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    runtime_checkable,
)

import jax.numpy as jnp
import numpy as np
from ordered_set import OrderedSet, OrderedSetInitializer
from typing_extensions import Self

from jax_utils.typing import DataclassInstance

Array: TypeAlias = Union[jnp.ndarray, np.ndarray]
Scalar: TypeAlias = Union[float, int]


def check_ndim_in(array: Array, allowed_ndims: Iterable[int]):
    """Checks if the number of dimensions matches some allowed values.

    Args:
        array (Array): numpy or JAX array
        allowed_ndims (Iterable[int]): number of dimensions allowed for ``array``

    Raises:
        ValueError: when the number of dimensions of ``array`` is not present in ``allowed_shapes``.
    """
    if array.ndim not in allowed_ndims:
        raise ValueError(f"Dimension of delivery should be in {allowed_ndims}")


def is_broadcastable(shape_1: Tuple[int, ...], shape_2: Tuple[int, ...]) -> bool:
    """Whether the shapes of 2 arrays/tensors can be
    `broadcasted <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.

    Args:
        shape_1 (Tuple[int, ...]): shape of the 1st array
        shape_2 (Tuple[int, ...]): shape of the 2nd array

    Returns:
        bool: ``True`` if and only if the 2 arrays/tensors can be broadcasted, False otherwise.
    """
    for dim1, dim2 in zip(shape_1[::-1], shape_2[::-1]):
        if not (dim1 == 1 or dim2 == 1 or dim1 == dim2):
            return False
    return True


T = TypeVar("T", bound=Hashable)


class TensorAxes(OrderedSet[T]):
    """An ordered set of "named" array/tensor axes.

    The position in the set defines the axis index in the tensor.

    The first ``len(self) - tensor_min_dim`` axes are optional:
    if absent, they are assumed to be "flattened".

    The reason for not making the last ``tensor_min_dim`` axes optional instead of the first
    ``len(self) - tensor_min_dim`` axes is to follow the logic of array broadcasting (first dimension can be omitted).
    """

    def __init__(
        self,
        initial: Optional[OrderedSetInitializer[T]] = None,
        tensor_min_dim: int = 0,
    ):
        super().__init__(initial)  # type: ignore
        if tensor_min_dim > len(self):
            raise ValueError(
                "tensor_min_shape_length should be smaller than the number of specified axes"
            )
        if tensor_min_dim < 0:
            raise ValueError("tensor_min_shape_length should be greater than or equal to 0")
        self.tensor_min_nb_dim = tensor_min_dim

    def reverse_index(self, key: T) -> int:
        """Computes the index of element ``key`` but rather than return a non-negative integer like
        method ``index``, it returns a negative integer e.g., -1 if for the last element,
        -2 for the penultimate element, ...

        Args:
            key (T): any element present in ``self`` (which is an ordered set)

        Returns:
            int: negative integer corresponding to the position of ``key`` in ``self``
                (starting from the last and decrementing by 1 for each element)
        """
        return -(len(self) - self.index(key))

    @property
    def _first_mandatory_idx(self) -> int:
        return -self.tensor_min_nb_dim if self.tensor_min_nb_dim > 0 else len(self)

    @property
    def mandatory(self) -> OrderedSet[T]:
        """
        Returns:
            OrderedSet[T]: ordered set of non-optional axes.
        """
        return OrderedSet(list(self[self._first_mandatory_idx :]))

    @property
    def optional(self) -> OrderedSet[T]:
        """
        Returns:
            OrderedSet[T]: ordered set of optional axes.
        """
        return OrderedSet(list(self[: self._first_mandatory_idx]))

    def __repr__(self) -> str:
        return type(self).__name__ + "\n| ".join(
            [""]
            + [f"{self.reverse_index(axis)} (optional): {axis}" for axis in self.optional]
            + [f"{self.reverse_index(axis)}: {axis}" for axis in self.mandatory]
        )


ArrayType = TypeVar("ArrayType", bound=Array)
AxisType = TypeVar("AxisType", bound=Hashable)


def expand_dims_axis(
    tensor_axes: TensorAxes[AxisType], missing_axes: AbstractSet[AxisType]
) -> Tuple[int, ...]:
    """Compute ``axis`` argument to pass to ``numpy.expand_dims(a, axis)`` function when some axes in ``tensor_axes``
    are not present in the numpy array ``a`` (they are "flattened").

    Args:
        tensor_axes (TensorAxes[AxisType]): some array/tensor axes
        missing_axes (AbstractSet[AxisType]): missing axes in the array/tensor

    Raises:
        ValueError: raises an error when ``missing_axes`` is not a subset of ``tensor_axes``

    Returns:
        Tuple[int, ...]: the ``axis`` argument to pass to ``numpy.expand_dims(a, axis)``
    """
    if not set(missing_axes).issubset(tensor_axes):
        raise ValueError(
            f"The following missing axes are invalid: {', '.join(str(axis) for axis in missing_axes - tensor_axes)}. "
            f"Only the following axes are valid: {', '.join(str(axis) for axis in tensor_axes)}"
        )
    return tuple(
        tensor_axes.reverse_index(axis) for axis in tensor_axes[::-1] if axis in missing_axes
    )


ArrayType_co = TypeVar("ArrayType_co", bound=Array, covariant=True)


class AverageableArrayLike(Protocol[ArrayType_co]):
    """Shared interface of all classes with a ``mean`` method return a scalar array (corresponding to the mean values of
    the initial array).

    Example of classes implementing this interface: (jax) numpy arrays, ...
    """

    def mean(self, *args, **kwargs) -> ArrayType_co:
        """
        Returns:
            ArrayType_co: Should return a scalar array
        """


@runtime_checkable
class Tensor(DataclassInstance, AverageableArrayLike, Protocol[ArrayType, AxisType]):
    """
    A wrapper for numpy/jax arrays with the following additional features:

    - axes are "named" to facilitate manipulation and debugging (an axis "name" could be a string or any other
      hashable ``AxisType``)
    - the returned ``values`` of the tensor can be different from the actual ``array`` given as input at construction,
      this allows to easily implement "change of variables" (a.k.a., "substitution")

    Args:
        _tensor_axes (ClassVar[TensorAxes[AxisType]]): class attribute defining the axes of the ``array``
        array (ArrayType): a (jax) numpy array containing all relevant data

    """

    _tensor_axes: ClassVar[TensorAxes[AxisType]]  # type: ignore[misc]
    array: ArrayType
    # TODO: to facilitate vectorization with vmap, it could be convenient  # pylint: disable=W0511
    # for the user to define an optional `values_axes` attribute (None by default).
    # When not None, `values` property would basically perform a reshape of `array` attribute
    # (permutation of axes). This would allow to put the `vectorized_axis` as the first axis in
    # all input tensors of a vectorized method (see vectorization.py)

    def __post_init__(self):
        self.check_array()

    def check_array(self):
        """Check the validity of the ``array`` attribute at construction.
        To be overriden if needed.
        """
        check_ndim_in(
            self.array, range(self.axes.tensor_min_nb_dim, len(type(self)._tensor_axes) + 1)
        )

    def __getitem__(self, key):
        return type(self)(array=self.array[key])  # type: ignore[call-arg]

    def getitem_from_axes(self, axes_keys: Dict[AxisType, Any]) -> Self:
        """Analogue of method ``__getitem__`` but where array slicing/indexing is explicitly applied to named axis.

        Args:
            axes_keys (Dict[AxisType, Any]): a mapping between axes names and slices/list of indices/...

        Returns:
            Self: a new ``Tensor`` of the same type with restricted data
        """
        array_key = tuple(
            axes_keys[axis] if axis in axes_keys else slice(None, None, None)
            for axis in self.actual_axes
        )
        return self[array_key]

    def __neg__(self) -> Self:
        return self.__class__(array=-self.array)  # type: ignore[call-arg]

    def __abs__(self) -> Self:
        return self.__class__(array=abs(self.array))  # type: ignore[call-arg]

    def __add__(self, other: Union[Scalar, Array, Tensor[ArrayType, AxisType]]) -> Self:
        if isinstance(other, Tensor):
            return self.__class__(array=self.array + other.array)  # type: ignore[call-arg]
        return self.__class__(array=self.array + other)  # type: ignore[call-arg]

    def __sub__(self, other: Union[Scalar, Array, Tensor[ArrayType, AxisType]]) -> Self:
        if isinstance(other, Tensor):
            return self.__class__(array=self.array - other.array)  # type: ignore[call-arg]
        return self.__class__(array=self.array - other)  # type: ignore[call-arg]

    def __mul__(self, other: Union[Scalar, Array, Tensor[ArrayType, AxisType]]) -> Self:
        if isinstance(other, Tensor):
            return self.__class__(array=self.array * other.array)  # type: ignore[call-arg]
        return self.__class__(array=self.array * other)  # type: ignore[call-arg]

    def __truediv__(self, other: Union[Scalar, Array, Tensor[ArrayType, AxisType]]) -> Self:
        if isinstance(other, Tensor):
            return self.__class__(array=self.array / other.array)  # type: ignore[call-arg]
        return self.__class__(array=self.array / other)  # type: ignore[call-arg]

    @classmethod
    def _expand_dims_axis(cls, missing_axes: AbstractSet[AxisType]):
        return expand_dims_axis(cls._tensor_axes, missing_axes)

    @property
    def values(self) -> ArrayType:
        """Override to apply transformations on ``self.array`` ("change of variable").

        For example, if one wants to manipulate (jax) tensors constrained to always have non-negative values in any
        situation (even if the tensor gets updated by some gradient descent procedure for instance), then one can
        override ``values`` by returning ``jax.nn.relu(self.array)``.

        By default ``values`` implements the identity i.e., returns ``self.array``.

        Returns:
            ArrayType: the transformed ``array``
        """
        return self.array

    def reverse_values(self, values: ArrayType) -> ArrayType:
        """Inverse of the transformation implemented in ``values`` property.

        By default, ``self.array = self.values`` so calling ``values`` is equivalent to calling ``array``.

        When the transformation mapping ``array`` to ``values`` is not the identity, ``reverse_values`` should be
        overriden accordingly.

        Args:
            values (ArrayType): the output of property ``values``

        Returns:
            ArrayType: the array obtained by apply the inverse transformation of
                ``values``
        """
        return values

    @property
    def dtype(self) -> np.dtype:
        """Get the type of ``self.values``"""
        return self.values.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of ``self.values``"""
        return self.values.shape

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of ``self.values``"""
        return self.values.ndim

    @property
    def axes(self) -> TensorAxes[AxisType]:
        """Possible "named" axes of the tensor"""
        return type(self)._tensor_axes

    @property
    def actual_axes(self) -> TensorAxes[AxisType]:
        """Actual axes of the tensor"""
        return TensorAxes(
            [self.axes[i] for i in range(-len(self.shape), 0)],
            tensor_min_dim=self.axes.tensor_min_nb_dim,
        )

    def mean(self, *args, **kwargs) -> Union[ArrayType, Any]:
        """Compute the mean on ``self.values`` on specified axes"""
        return self.values.mean(*args, **kwargs)

    def mean_over_axes(self, axes: AbstractSet[AxisType]) -> Union[ArrayType, Any]:
        """Averages ``values`` along one or several named axes.

        Args:
            axes (AbstractSet[AxisType]): axes along which the means are computed

        Returns:
            Union[ArrayType, Any]: new array containing the mean ``values``
        """
        axes_to_mean_over = tuple(self.index(axis) for axis in self.actual_axes.intersection(axes))
        return self.mean(axis=axes_to_mean_over)

    def sum(self, *args, **kwargs) -> Union[ArrayType, Any]:
        """Get the sum of the array"""
        return self.values.sum(*args, **kwargs)

    def sum_over_axes(self, axes: AbstractSet[AxisType]) -> Union[ArrayType, Any]:
        """Sums ``values`` along one or several named axes.

        Args:
            axes (AbstractSet[AxisType]): axes along which the sums are computed

        Returns:
            Union[ArrayType, Any]: new array containing the summed ``values``
        """
        axes_to_mean_over = tuple(self.index(axis) for axis in self.actual_axes.intersection(axes))
        return self.sum(axis=axes_to_mean_over)

    def has(self, axis: AxisType) -> bool:
        """Whether ``axis`` is one of the possible axes of the tensor.

        Args:
            axis (AxisType): a "named" axis

        Returns:
            bool: ``True`` if ``axis`` is one of the possible axes of the tensor, ``False`` otherwise.
        """
        return axis in self.axes

    def has_actual(self, axis: AxisType) -> bool:
        """Whether ``axis`` is one of the actual axes of the tensor.

        Args:
            axis (AxisType): a "named" axis

        Returns:
            bool: ``True`` if ``axis`` is one of the actual axes of the tensor, ``False`` otherwise.
        """
        return axis in self.actual_axes

    def index(self, axis: AxisType) -> int:
        """Index of ``axis`` in tensor i.e., returns  ``i`` if and only if ``axis`` is the ``i``-th
        dimension of the tensor.

        Args:
            axis (AxisType): a "named" axis

        Returns:
            int: the corresponding index
        """
        return self.actual_axes.index(axis)

    def reverse_index(self, axis: AxisType) -> int:
        """Index of ``axis`` in tensor but in reverse order i.e., returns  ``-i`` if and only if
        ``axis`` is the ``(n - i)``-th dimension of the tensor (with ``n`` the total number of dimension).
        dimension of the tensor.

        Args:
            axis (AxisType): a "named" axis

        Returns:
            int: the corresponding index in reverse order
        """
        return self.actual_axes.reverse_index(axis)

    def size(
        self,
        axis: AxisType,
    ) -> int:
        """The size of a given ``axis``. Returns 0 when the ``axis`` is "flattened".

        Args:
            axis (AxisType): a "named" axis

        Raises:
            ValueError: if ``axis`` is not a possible axis

        Returns:
            int: the size of ``axis`` (0 when the ``axis`` is "flattened")
        """
        if not self.has(axis):
            raise ValueError(
                f"{axis} is not a valid axis."
                f"Valid axes are: {', '.join(str(a) for a in self.axes)}"
            )
        if not self.has_actual(axis):
            return 0
        reverse_idx = self.reverse_index(axis)
        return self.shape[reverse_idx]

    def is_broadcastable_with(
        self,
        other_tensor_or_shape: Union[Tensor[ArrayType, AxisType], Tuple[int, ...]],
    ) -> bool:
        """Whether this tensor can be
        `broadcasted <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_
        with ``other_tensor_or_shape``.

        Args:
            other_tensor_or_shape (Union[Tensor[ArrayType, AxisType], Tuple[int, ...]]): another
                tensor

        Returns:
            bool: ``True`` if and only if this tensor can be broadcasted with ``other_tensor_or_shape``,
                ``False`` otherwise.
        """
        return is_broadcastable(
            self.shape,
            other_tensor_or_shape
            if isinstance(other_tensor_or_shape, tuple)
            else other_tensor_or_shape.shape,
        )


class RegularizedArrayLikeCost(AverageableArrayLike[ArrayType], Protocol[ArrayType]):
    """Interface for "regularized" costs i.e., costs of the form:
    ``cost + lagrangian_coefficient * regularization``

    `More about regularization <https://en.wikipedia.org/wiki/Regularization_(mathematics)>`_.

    Args:
        cost (AverageableArrayLike[ArrayType]): an array-like collection of costs
        regularization (AverageableArrayLike[ArrayType]): an array-like collection regularization costs
        lagrangian_coefficient (Number): a non-negative number quantifying the regularization weight
    """

    cost: AverageableArrayLike[ArrayType]
    regularization: AverageableArrayLike[ArrayType]
    lagrangian_coefficient: Number = 1  # type: ignore[assignment]

    def mean(self, *args, **kwargs) -> Union[ArrayType, Any]:
        return (
            self.cost.mean(*args, **kwargs)
            + self.regularization.mean(*args, **kwargs) * self.lagrangian_coefficient
        )
