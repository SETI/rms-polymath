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

from typing import Any

from polymath.matrix import Matrix
from polymath.qube import Qube
from polymath.quaternion import Quaternion
from polymath.scalar import Scalar
from polymath.typedefs import QubeLike, ScalarLike, VectorLike


__all__ = ['Matrix3']

class Matrix3(Matrix):
    IDENTITY: Matrix3
    MASKED: Matrix3
    def __add__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __getstate__(self) -> dict: ...  # type: ignore[type-arg]
    def __iadd__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __imul__(self, arg: Any) -> Matrix3: ...  # type: ignore[misc, override]
    def __isub__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __mul__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Qube: ...
    def __neg__(self) -> None: ...  # type: ignore[override]
    def __radd__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __rmul__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Qube: ...
    def __rsub__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def __sub__(self, arg: Any) -> None: ...  # type: ignore[override]
    @staticmethod
    def as_matrix3(arg: Any, *, recursive: bool = ...) -> Matrix3: ...
    @staticmethod
    def axis_rotation(angle: ScalarLike, axis: int = ..., *, recursive: bool = ...
        ) -> Matrix3: ...
    @staticmethod
    def from_euler(ai: ScalarLike, aj: ScalarLike, ak: ScalarLike, axes: str = ...
        ) -> Matrix3: ...
    def mean(self, axis: int | tuple[int, ...] | None = ..., *, recursive: bool = ...,  # type: ignore[override]
        builtins: bool | None = ..., dtype: Any | None = ..., out: Any | None = ...
        ) -> None: ...
    @staticmethod
    def pole_rotation(ra: ScalarLike, dec: ScalarLike) -> Matrix3: ...
    def reciprocal(self, *, recursive: bool = ..., nozeros: bool = ...  # type: ignore[override]
        ) -> Matrix3: ...
    def rotate(self, arg: QubeLike, *, recursive: bool = ...) -> Qube: ...
    def sum(self, axis: int | tuple[int, ...] | None = ..., *, recursive: bool = ...,  # type: ignore[override]
        builtins: bool | None = ..., out: Any | None = ...) -> None: ...
    def to_euler(self, axes: str = ...) -> tuple[Scalar, Scalar, Scalar]: ...
    def to_quaternion(self, recursive: bool = ...) -> Quaternion: ...
    @staticmethod
    def twovec(vector1: VectorLike, axis1: int, vector2: VectorLike, axis2: int, *,
        recursive: bool = ...) -> Matrix3: ...
    def unrotate(self, arg: QubeLike, *, recursive: bool = ...) -> Qube: ...
    @staticmethod
    def x_rotation(angle: ScalarLike, *, recursive: bool = ...) -> Matrix3: ...
    @staticmethod
    def y_rotation(angle: ScalarLike, *, recursive: bool = ...) -> Matrix3: ...
    @staticmethod
    def z_rotation(angle: ScalarLike, *, recursive: bool = ...) -> Matrix3: ...

##########################################################################################
