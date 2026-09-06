##########################################################################################
# polymath/matrix.pyi
##########################################################################################
"""Type stub for :mod:`polymath.matrix`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. These stubs describe the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
taken from the docstrings wherever those state one unambiguously, and are left as `Any`
where they do not, rather than guessed at.
"""

from typing import Any

from polymath.qube import Qube
from polymath.boolean import Boolean
from polymath.matrix3 import Matrix3
from polymath.scalar import Scalar
from polymath.vector import Vector


__all__ = ['Matrix']

class Matrix(Qube):
    IDENTITY2: Matrix
    IDENTITY3: Matrix
    MASKED2: Matrix
    MASKED3: Matrix
    @property
    def T(self) -> Matrix: ...  # noqa: N802
    UNIT33: Matrix
    XAXIS_COL: Matrix
    XAXIS_ROW: Matrix
    YAXIS_COL: Matrix
    YAXIS_ROW: Matrix
    ZAXIS_COL: Matrix
    ZAXIS_ROW: Matrix
    ZERO33: Matrix
    ZERO3_COL: Matrix
    ZERO3_ROW: Matrix
    def __abs__(self) -> None: ...  # type: ignore[override]
    def __floordiv__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __ifloordiv__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __imod__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __mod__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __rfloordiv__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __rmod__(self, arg: Any) -> None: ...  # type: ignore[override]
    @staticmethod
    def as_matrix(arg: Any, *, recursive: bool = ...) -> Matrix: ...
    def column_vector(self, column: int, *, recursive: bool = ...,
        classes: type | list[type] | tuple[type, ...] = ...) -> Qube: ...
    def column_vectors(self, recursive: bool = ...,
        classes: type | list[type] | tuple[type, ...] = ...) -> tuple[Qube, ...]: ...
    @staticmethod
    def from_scalars(*args: Any, recursive: bool = ...,  # type: ignore[override]
        shape: tuple[int, ...] | None = ...,
        classes: type | list[type] | tuple[type, ...] = ...) -> Matrix: ...
    def identity(self) -> Matrix: ...  # type: ignore[override]
    def inverse(self, *, recursive: bool = ..., nozeros: bool = ...) -> Matrix: ...
    def is_diagonal(self, *, delta: float = ...) -> Boolean: ...
    def reciprocal(self, *, recursive: bool = ..., nozeros: bool = ...  # type: ignore[override]
        ) -> Matrix: ...
    def row_vector(self, row: int, *, recursive: bool = ...,
        classes: type | list[type] | tuple[type, ...] = ...) -> Qube: ...
    def row_vectors(self, *, recursive: bool = ...,
        classes: type | list[type] | tuple[type, ...] = ...) -> tuple[Qube, ...]: ...
    def solve(self, arg: Vector | list | tuple, *, recursive: bool = ...,  # type: ignore[type-arg]
        nozeros: bool = ...) -> Vector: ...
    def to_scalar(self, indx0: int, indx1: int, *, recursive: bool = ...) -> Scalar: ...
    def to_vector(self, axis: int, indx: int, *, recursive: bool = ...,
        classes: type | list[type] | tuple[type, ...] = ...) -> Qube: ...
    def transpose(self, *, recursive: bool = ...) -> Matrix: ...
    def unitary(self) -> Matrix3: ...

##########################################################################################
