"""Define useful types for package"""

from dataclasses import Field, dataclass
from typing import Any, ClassVar, Hashable, List, Optional, Protocol, Tuple, Union


class DataclassInstance(Protocol):
    """Type for Python dataclasses (see ``dataclasses.dataclass``).

    This code is copy-pasted from ``_typeshed`` (but is simpler to import).
    """

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


class HashableIndexingOrSlicing(DataclassInstance, Hashable, Protocol):
    """Python slices and lists (of indices) are not hashable in Python (slices are hashable for Python version >= 3.12).
    This is a shared interface for hashable slices and indices.
    """

    @property
    def values(self) -> Union[slice, int, List[int]]:
        """
        Property returning index values in the form of a slice, an int (for singletons) or a list of int.

        Returns:
            Union[slice, int, List[int]]: index values
        """


@dataclass(eq=True, frozen=True)
class HashableSlicing(HashableIndexingOrSlicing):
    """Python slices are not hashable for Python version < 3.12.
    This class allows to define hashable slices.

    Args:
        start (Optional[int]): initial value of the slice (included)
        stop (Optional[int]): last value of the slice (excluded)
        step (Optional[int]): step between values
    """

    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    @property
    def values(self) -> slice:
        return slice(self.start, self.stop, self.step)

    def __hash__(self):
        return hash((self.start, self.stop, self.step))


@dataclass(eq=True, frozen=True, init=False)
class HashableIndexing(HashableIndexingOrSlicing):
    """Python lists are not hashable.
    This class allows to define hashable list of indices.

    Args:
        indices (Union[int, Tuple[int]]): index or list of (integer-valued) indices
    """

    indices: Union[int, Tuple[int]]

    def __init__(self, *indices: int):
        object.__setattr__(self, "indices", tuple(indices) if len(indices) > 1 else indices[0])

    @property
    def values(self) -> Union[int, List[int]]:
        return self.indices if isinstance(self.indices, int) else list(self.indices)

    def __hash__(self):
        return hash(self.indices)
