##########################################################################################
# polymath/pair.pyi
##########################################################################################
"""Type stub for :mod:`polymath.pair`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. These stubs describe the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
taken from the docstrings wherever those state one unambiguously, and are left as `Any`
where they do not, rather than guessed at.
"""

from typing import Any

from polymath.vector import Vector
from polymath.scalar import Scalar
from polymath.typedefs import PairLike, ScalarLike


__all__ = ['Pair']

class Pair(Vector):
    HALF: Pair
    IDENTITY: Pair
    INT00: Pair
    INT11: Pair
    MASKED: Pair
    ONES: Pair
    XAXIS: Pair
    YAXIS: Pair
    ZERO: Pair
    ZEROS: Pair
    def angle(self, *, recursive: bool = ...) -> Scalar: ...
    @staticmethod
    def as_pair(arg: Any, *, recursive: bool = ...) -> Pair: ...
    def clip2d(self, lower: PairLike | None, upper: PairLike | None, *,
        remask: bool = ...) -> Pair: ...
    @staticmethod
    def from_scalars(x: ScalarLike | None, y: ScalarLike | None, *,  # type: ignore[override]
        recursive: bool = ..., readonly: bool = ...) -> Pair: ...
    def rot90(self, *, recursive: bool = ...) -> Pair: ...
    def swapxy(self, *, recursive: bool = ...) -> Pair: ...

##########################################################################################
