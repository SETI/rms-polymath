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

from typing import Any

from polymath.vector import Vector
from polymath.scalar import Scalar
from polymath.typedefs import QubeLike, ScalarLike


__all__ = ['Polynomial']

class Polynomial(Vector):
    def __add__(self, arg: Any) -> Polynomial: ...  # type: ignore[override]
    def __eq__(self, arg: object) -> bool: ...  # type: ignore[override]
    def __iadd__(self, arg: Any) -> Polynomial: ...  # type: ignore[override]
    def __imul__(self, arg: Any) -> Polynomial: ...  # type: ignore[override]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __isub__(self, arg: QubeLike) -> Polynomial: ...  # type: ignore[override, misc]
    def __itruediv__(self, arg: Any) -> Polynomial: ...  # type: ignore[override]
    def __mul__(self, arg: Any) -> Polynomial: ...  # type: ignore[override]
    def __ne__(self, arg: object) -> bool: ...  # type: ignore[override]
    def __neg__(self) -> Polynomial: ...  # type: ignore[override]
    def __pow__(self, arg: Any) -> Polynomial: ...
    def __radd__(self, arg: Any) -> Polynomial: ...  # type: ignore[override]
    def __rmul__(self, arg: Any) -> Polynomial: ...  # type: ignore[override]
    def __rsub__(self, arg: Any) -> Polynomial: ...  # type: ignore[override]
    def __sub__(self, arg: Any) -> Polynomial: ...  # type: ignore[override]
    def __truediv__(self, arg: Any) -> Polynomial: ...  # type: ignore[override]
    @staticmethod
    def as_polynomial(arg: Any, *, recursive: bool = ...) -> Polynomial: ...
    def as_vector(self, *, recursive: bool = ...) -> Vector: ...  # type: ignore[override]
    def at_least_order(self, order: int, *, recursive: bool = ...) -> Polynomial: ...
    def deriv(self, recursive: bool = ...) -> Polynomial: ...
    def eval(self, x: ScalarLike, recursive: bool = ...) -> Scalar: ...
    def invert_line(self, *, recursive: bool = ...) -> Polynomial: ...
    @property
    def order(self) -> int: ...
    def roots(self, recursive: bool = ...) -> Scalar: ...
    def set_order(self, order: int, *, recursive: bool = ...) -> Polynomial: ...

##########################################################################################
