##########################################################################################
# polymath/quaternion.pyi
##########################################################################################
"""Type stub for :mod:`polymath.quaternion`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. These stubs describe the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
taken from the docstrings wherever those state one unambiguously, and are left as `Any`
where they do not, rather than guessed at.
"""

from typing import Any

from numpy.typing import NDArray

from polymath.qube import _Arraylike, _ShapeOrTuple
from polymath.vector import Vector

__all__ = ['Quaternion']

class Quaternion(Vector):
    IDENTITY: Quaternion
    MASKED: Quaternion
    XAXIS: Quaternion
    YAXIS: Quaternion
    ZAXIS: Quaternion
    ZERO: Quaternion
    def __mul__(self, arg: Any, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    def __rmul__(self, arg: Any, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    def __truediv__(self, arg: Any, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    @staticmethod
    def as_quaternion(arg: Any, *, recursive: bool = ...) -> _Arraylike: ...
    def conj(self, *, recursive: bool = ...) -> _Arraylike: ...
    @staticmethod
    def from_euler(ai: Any, aj: Any, ak: Any, axes: str = ...) -> _Arraylike: ...
    @staticmethod
    def from_euler_via_matrix(ai: Any, aj: Any, ak: Any,
        axes: str = ...) -> _Arraylike: ...
    @staticmethod
    def from_matrix3(matrix: _Arraylike, *, recursive: bool = ...) -> _Arraylike: ...
    @staticmethod
    def from_parts(scalar: Any, vector: Any, *, recursive: bool = ...) -> _Arraylike: ...
    @staticmethod
    def from_rotation(angle: _Arraylike, vector: _Arraylike, *,
        recursive: bool = ...) -> _Arraylike: ...
    def identity(self) -> _Arraylike: ...
    @staticmethod
    def mul_values(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]: ...
    def reciprocal(self, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    def to_euler(self, axes: str = ...) -> _ShapeOrTuple: ...
    def to_matrix3(self, *, recursive: bool = ...,
        partials: bool = ...) -> _Arraylike | _ShapeOrTuple: ...
    def to_parts(self, *, recursive: bool = ...) -> _ShapeOrTuple: ...
    def to_rotation(self, *, recursive: bool = ...) -> _ShapeOrTuple: ...

##########################################################################################
