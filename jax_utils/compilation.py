"""Classes and methods to to jit-compile functions involving jax arrays transformations"""

from functools import wraps
from typing import Callable, Protocol

from jax import jit
from typing_extensions import Self


class JaxCompilableProtocol(Protocol):
    """All classes implementing this interface should implement property ``is_compilation_enabled``
    indicating whether the methods involving JAX arrays should be
    `jit-compiled <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_.
    """

    # pylint: disable=C0116
    @property
    def is_compilation_enabled(self) -> bool:
        pass


class BaseJaxCompilable(JaxCompilableProtocol, Protocol):
    """Subclassing ``BaseCompilableJax`` allows to easily enable/disable
    `jit-compilation <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_
    of methods involving JAX arrays.

    Use ``with_optional_jax_jit`` decorator to compile a method only when
    ``is_compilation_enabled`` is ``True`` (``False`` by default).

    To enable (resp. disable) jit-compilation, one only needs to call method ``enable_compilation``
    (resp. ``disable_compilation``). By default, jit-compilation is disabled.
    """

    @property
    def is_compilation_enabled(self) -> bool:
        return hasattr(self, "_is_compilation_enabled") and getattr(self, "_is_compilation_enabled")

    # pylint: disable=C0116
    def enable_compilation(self) -> Self:
        object.__setattr__(self, "_is_compilation_enabled", True)
        return self

    # pylint: disable=C0116
    def disable_compilation(self) -> Self:
        object.__setattr__(self, "_is_compilation_enabled", False)
        return self


def jit_when_compilation_enabled(**jax_jit_args) -> Callable[[Callable], Callable]:
    """Parametrized decorator for methods of classes implementing ``CompilableJaxProtocol`` interface.
    Allows to `jit-compile <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_
    some methods only when compilation is enabled.

    Returns:
        Callable[[Callable], Callable]: decorator with parameters ``jax_jit_args`` specified
    """

    def decorator(
        func: Callable,
    ) -> Callable:
        @wraps(func)
        def wrapper(self: JaxCompilableProtocol, *args, **kwargs):
            if self.is_compilation_enabled:
                return jit(func, **jax_jit_args)(self, *args, **kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
