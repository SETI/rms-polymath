##########################################################################################
# polymath/boolean.pyi
##########################################################################################
"""Type stub for :mod:`polymath.boolean`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. These stubs describe the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
taken from the docstrings wherever those state one unambiguously, and are left as `Any`
where they do not, rather than guessed at.
"""

from typing import Any

from numpy.typing import NDArray

from polymath.qube import _Arraylike
from polymath.scalar import Scalar

__all__ = ['Boolean']

class Boolean(Scalar):
    FALSE: Boolean
    MASKED: Boolean
    TRUE: Boolean
    def __abs__(self, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    def __add__(self, arg: _Arraylike, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    def __floordiv__(self, arg: _Arraylike) -> _Arraylike: ...  # type: ignore[override]
    def __ge__(self, arg: Any, *,  # type: ignore[override]
        builtins: bool = ...) -> _Arraylike | bool: ...
    def __gt__(self, arg: Any, *,  # type: ignore[override]
        builtins: bool = ...) -> _Arraylike | bool: ...
    def __iadd__(self, arg: _Arraylike) -> Any: ...  # type: ignore[misc, override]
    def __ifloordiv__(self, arg: _Arraylike) -> Any: ...
    def __imod__(self, arg: _Arraylike) -> Any: ...  # type: ignore[override]
    def __imul__(self, arg: _Arraylike) -> Any: ...  # type: ignore[misc, override]
    def __ipow__(self, arg: _Arraylike) -> Any: ...  # type: ignore[override]
    def __isub__(self, arg: _Arraylike) -> Any: ...  # type: ignore[misc, override]
    def __itruediv__(self, arg: _Arraylike) -> Any: ...  # type: ignore[misc, override]
    def __le__(self, arg: Any, *,  # type: ignore[override]
        builtins: bool = ...) -> _Arraylike | bool: ...
    def __lt__(self, arg: Any, *,  # type: ignore[override]
        builtins: bool = ...) -> _Arraylike | bool: ...
    def __mod__(self, arg: _Arraylike) -> _Arraylike: ...  # type: ignore[override]
    def __mul__(self, arg: _Arraylike, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    def __neg__(self, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    def __pos__(self, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    def __pow__(self, arg: _Arraylike) -> _Arraylike: ...  # type: ignore[override]
    def __radd__(self, arg: _Arraylike, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[misc, override]
    def __rfloordiv__(self, arg: _Arraylike) -> _Arraylike: ...  # type: ignore[override]
    def __rmod__(self, arg: _Arraylike) -> _Arraylike: ...  # type: ignore[override]
    def __rmul__(self, arg: _Arraylike, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[misc, override]
    def __rsub__(self, arg: _Arraylike, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    def __rtruediv__(self, arg: _Arraylike, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    def __sub__(self, arg: _Arraylike, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    def __truediv__(self, arg: _Arraylike, *, recursive: bool = ...) -> _Arraylike: ...  # type: ignore[override]
    @staticmethod
    def as_boolean(arg: Any, *, recursive: bool = ...) -> _Arraylike: ...
    def as_index(self) -> NDArray[Any]: ...  # type: ignore[override]
    def identity(self) -> _Arraylike: ...
    def sum(self, axis: Any = ..., *, value: bool = ..., builtins: bool | None = ...,
        recursive: bool = ..., masked: bool | None = ...,
        out: Any = ...) -> _Arraylike: ...

##########################################################################################
