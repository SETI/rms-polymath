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

import builtins
from typing import Any

from polymath.qube import Qube, _Arraylike, _ShapeOrTuple

__all__ = ['Matrix']

class Matrix(Qube):
    IDENTITY2: Matrix
    IDENTITY3: Matrix
    MASKED2: Matrix
    MASKED3: Matrix
    @property
    def T(self) -> _Arraylike: ...  # noqa: N802
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
    def __abs__(self) -> Any: ...  # type: ignore[override]
    def __floordiv__(self, arg: Any) -> Any: ...
    def __ifloordiv__(self, arg: Any) -> Any: ...
    def __imod__(self, arg: Any) -> Any: ...  # type: ignore[override]
    def __mod__(self, arg: Any) -> Any: ...  # type: ignore[override]
    def __rfloordiv__(self, arg: Any) -> Any: ...
    def __rmod__(self, arg: Any) -> Any: ...  # type: ignore[override]
    @staticmethod
    def as_matrix(arg: Any, *, recursive: bool = ...) -> _Arraylike: ...
    def column_vector(self, column: Any, *, recursive: bool = ...,
        classes: type | tuple[type, ...] | list[type] = ...) -> _Arraylike: ...
    def column_vectors(self, recursive: bool = ...,
        classes: type | tuple[type, ...] | list[type] = ...) -> _ShapeOrTuple: ...
    @staticmethod
    def from_scalars(*args: Any, recursive: bool = ...,  # type: ignore[override]
        shape: _ShapeOrTuple | None = ...,
        classes: type | tuple[type, ...] | list[type] = ...) -> _Arraylike: ...
    def identity(self) -> Any: ...
    def inverse(self, *, recursive: bool = ..., nozeros: bool = ...) -> _Arraylike: ...
    def is_diagonal(self, *, delta: float = ...) -> _Arraylike: ...
    def reciprocal(self, *, recursive: bool = ..., nozeros: bool = ...) -> _Arraylike: ...
    def row_vector(self, row: Any, *, recursive: bool = ...,
        classes: type | tuple[type, ...] | list[type] = ...) -> _Arraylike: ...
    def row_vectors(self, *, recursive: bool = ...,
        classes: type | tuple[type, ...] | list[type] = ...) -> _ShapeOrTuple: ...
    def solve(self, arg: _Arraylike, *, recursive: bool = ...,
        nozeros: bool = ...) -> _Arraylike: ...
    def to_scalar(self, indx0: builtins.int, indx1: builtins.int, *,
        recursive: bool = ...) -> _Arraylike: ...
    def to_vector(self, axis: Any, indx: Any, *, recursive: bool = ...,
        classes: type | tuple[type, ...] | list[type] = ...) -> _Arraylike: ...
    def transpose(self, *, recursive: bool = ...) -> _Arraylike: ...
    def unitary(self) -> _Arraylike: ...

##########################################################################################
