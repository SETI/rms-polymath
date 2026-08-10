##########################################################################################
# polymath/polynomial.pyi
##########################################################################################
"""Type stub for :mod:`polymath.polynomial`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. These stubs describe the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
taken from the docstrings wherever those state one unambiguously, and are left as `Any`
where they do not, rather than guessed at.
"""

import builtins
from typing import Any

from polymath.qube import _Arraylike
from polymath.vector import Vector

__all__ = ['Polynomial']

class Polynomial(Vector):
    def __add__(self, arg: Any) -> _Arraylike: ...  # type: ignore[override]
    def __eq__(self, arg: object) -> Any: ...
    def __iadd__(self, arg: Any) -> _Arraylike: ...  # type: ignore[override]
    def __imul__(self, arg: Any) -> _Arraylike: ...  # type: ignore[override]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __isub__(self, arg: Any) -> _Arraylike: ...  # type: ignore[override]
    def __itruediv__(self, arg: Any) -> _Arraylike: ...  # type: ignore[override]
    def __mul__(self, arg: Any) -> _Arraylike: ...  # type: ignore[override]
    def __ne__(self, arg: object) -> Any: ...
    def __neg__(self) -> _Arraylike: ...  # type: ignore[override]
    def __pow__(self, arg: Any) -> _Arraylike: ...  # type: ignore[override]
    def __radd__(self, arg: Any) -> _Arraylike: ...  # type: ignore[override]
    def __rmul__(self, arg: Any) -> _Arraylike: ...  # type: ignore[override]
    def __rsub__(self, arg: Any) -> _Arraylike: ...  # type: ignore[override]
    def __sub__(self, arg: Any) -> _Arraylike: ...  # type: ignore[override]
    def __truediv__(self, arg: Any) -> _Arraylike: ...  # type: ignore[override]
    @staticmethod
    def as_polynomial(arg: Any, *, recursive: bool = ...) -> _Arraylike: ...
    def as_vector(self, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    def at_least_order(self, order: builtins.int, *,
        recursive: bool = ...) -> _Arraylike: ...
    def deriv(self, recursive: bool = ...) -> _Arraylike: ...
    def eval(self, x: Any, recursive: bool = ...) -> _Arraylike: ...
    def invert_line(self, *, recursive: bool = ...) -> _Arraylike: ...
    @property
    def order(self) -> builtins.int: ...
    def roots(self, recursive: bool = ...) -> _Arraylike: ...
    def set_order(self, order: builtins.int, *, recursive: bool = ...) -> _Arraylike: ...

##########################################################################################
