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


from polymath.vector import Vector
from polymath.matrix import Matrix
from polymath.matrix3 import Matrix3
from polymath.scalar import Scalar
from polymath.typedefs import Matrix3Like, ScalarLike, Vector3Like


__all__ = ['Quaternion']

class Quaternion(Vector):
    IDENTITY: Quaternion
    MASKED: Quaternion
    XAXIS: Quaternion
    YAXIS: Quaternion
    ZAXIS: Quaternion
    ZERO: Quaternion
    def __mul__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Quaternion: ...
    def __rmul__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Quaternion: ...
    def __truediv__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Quaternion: ...
    @staticmethod
    def as_quaternion(arg: Any, *, recursive: bool = ...) -> Quaternion: ...
    def conj(self, *, recursive: bool = ...) -> Quaternion: ...
    @staticmethod
    def from_euler(ai: ScalarLike, aj: ScalarLike, ak: ScalarLike, axes: str = ...
        ) -> Quaternion: ...
    @staticmethod
    def from_euler_via_matrix(ai: ScalarLike, aj: ScalarLike, ak: ScalarLike,
        axes: str = ...) -> Quaternion: ...
    @staticmethod
    def from_matrix3(matrix: Matrix3Like, *, recursive: bool = ...) -> Quaternion: ...
    @staticmethod
    def from_parts(scalar: ScalarLike | None, vector: Vector3Like | None, *,
        recursive: bool = ...) -> Quaternion: ...
    @staticmethod
    def from_rotation(angle: ScalarLike, vector: Vector3Like, *, recursive: bool = ...
        ) -> Quaternion: ...
    def identity(self) -> Quaternion: ...  # type: ignore[override]
    def reciprocal(self, *, recursive: bool = ...  # type: ignore[override]
        ) -> Quaternion: ...
    def to_euler(self, axes: str = ...) -> tuple[Scalar, Scalar, Scalar]: ...
    def to_matrix3(self, *, recursive: bool = ..., partials: bool = ...
        ) -> Matrix3 | tuple[Matrix3, Matrix]: ...
    def to_parts(self, *, recursive: bool = ...) -> None: ...
    def to_rotation(self, *, recursive: bool = ...) -> None: ...

##########################################################################################
