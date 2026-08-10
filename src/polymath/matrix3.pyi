##########################################################################################
# polymath/matrix3.pyi
##########################################################################################
"""Type stub for :mod:`polymath.matrix3`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. These stubs describe the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
taken from the docstrings wherever those state one unambiguously, and are left as `Any`
where they do not, rather than guessed at.
"""

import builtins
from typing import Any

from polymath.matrix import Matrix
from polymath.qube import Qube, _Arraylike, _ShapeOrTuple

__all__ = ['Matrix3']

class Matrix3(Matrix):
    IDENTITY: Matrix3
    MASKED: Matrix3
    def __add__(self, arg: Any) -> Any: ...  # type: ignore[override]
    def __iadd__(self, arg: Any) -> Any: ...  # type: ignore[override]
    def __imul__(self, arg: Any) -> _Arraylike: ...  # type: ignore[misc, override]
    def __isub__(self, arg: Any) -> Any: ...  # type: ignore[override]
    def __mul__(self, arg: Any, *, recursive: bool = ...) -> Qube: ...  # type: ignore[override]
    def __neg__(self) -> Any: ...  # type: ignore[override]
    def __radd__(self, arg: Any) -> Any: ...  # type: ignore[override]
    def __rmul__(self, arg: Any, *, recursive: bool = ...) -> Qube: ...  # type: ignore[override]
    def __rsub__(self, arg: Any) -> Any: ...  # type: ignore[override]
    def __sub__(self, arg: Any) -> Any: ...  # type: ignore[override]
    @staticmethod
    def as_matrix3(arg: Any, *, recursive: bool = ...) -> _Arraylike: ...
    @staticmethod
    def axis_rotation(angle: Any, axis: builtins.int = ..., *,
        recursive: bool = ...) -> _Arraylike: ...
    @staticmethod
    def from_euler(ai: Any, aj: Any, ak: Any, axes: str = ...) -> _Arraylike: ...
    def mean(self, axis: Any = ..., *, recursive: bool = ..., builtins: Any = ...,  # type: ignore[override]
        dtype: Any = ..., out: Any = ...) -> Any: ...
    @staticmethod
    def pole_rotation(ra: Any, dec: Any) -> _Arraylike: ...
    def reciprocal(self, *, recursive: bool = ..., nozeros: bool = ...) -> _Arraylike: ...
    def rotate(self, arg: Any, *, recursive: bool = ...) -> Qube: ...
    def sum(self, axis: Any = ..., *, recursive: bool = ..., builtins: Any = ...,  # type: ignore[override]
        out: Any = ...) -> Any: ...
    def to_euler(self, axes: str = ...) -> _ShapeOrTuple: ...
    def to_quaternion(self, recursive: bool = ...) -> _Arraylike: ...
    @staticmethod
    def twovec(vector1: _Arraylike, axis1: builtins.int, vector2: _Arraylike,
        axis2: builtins.int, *, recursive: bool = ...) -> _Arraylike: ...
    def unrotate(self, arg: Any, *, recursive: bool = ...) -> Qube: ...
    @staticmethod
    def x_rotation(angle: Any, *, recursive: bool = ...) -> _Arraylike: ...
    @staticmethod
    def y_rotation(angle: Any, *, recursive: bool = ...) -> _Arraylike: ...
    @staticmethod
    def z_rotation(angle: Any, *, recursive: bool = ...) -> _Arraylike: ...

##########################################################################################
