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

from polymath.qube import _Arraylike
from polymath.vector import Vector

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
    def angle(self, *, recursive: bool = ...) -> _Arraylike: ...
    @staticmethod
    def as_pair(arg: Any, *, recursive: bool = ...) -> _Arraylike: ...
    def clip2d(self, lower: Any, upper: Any, *, remask: bool = ...) -> _Arraylike: ...
    @staticmethod
    def from_scalars(x: Any, y: Any, *, recursive: bool = ...,  # type: ignore[override]
        readonly: bool = ...) -> _Arraylike: ...
    def rot90(self, *, recursive: bool = ...) -> _Arraylike: ...
    def swapxy(self, *, recursive: bool = ...) -> _Arraylike: ...

##########################################################################################
