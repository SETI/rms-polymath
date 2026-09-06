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


from polymath.scalar import Scalar
import numpy as np


__all__ = ['Boolean']

class Boolean(Scalar):
    FALSE: Boolean
    MASKED: Boolean
    TRUE: Boolean
    def __abs__(self, *, recursive: bool = ...) -> Scalar: ...  # type: ignore[override]
    def __add__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Scalar: ...
    def __floordiv__(self, arg: Any) -> Scalar: ...
    def __ge__(self, arg: Any, *, builtins: bool = ...  # type: ignore[override]
        ) -> Boolean | bool: ...
    def __gt__(self, arg: Any, *, builtins: bool = ...  # type: ignore[override]
        ) -> Boolean | bool: ...
    def __iadd__(self, arg: Any) -> None: ...  # type: ignore[misc, override]
    def __ifloordiv__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __imod__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __imul__(self, arg: Any) -> None: ...  # type: ignore[misc, override]
    def __ipow__(self, arg: Any) -> None: ...  # type: ignore[override]
    def __isub__(self, arg: Any) -> None: ...  # type: ignore[misc, override]
    def __itruediv__(self, arg: Any) -> None: ...  # type: ignore[misc, override]
    def __le__(self, arg: Any, *, builtins: bool = ...  # type: ignore[override]
        ) -> Boolean | bool: ...
    def __lt__(self, arg: Any, *, builtins: bool = ...  # type: ignore[override]
        ) -> Boolean | bool: ...
    def __mod__(self, arg: Any) -> Scalar: ...  # type: ignore[override]
    def __mul__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Scalar: ...
    def __neg__(self, *, recursive: bool = ...) -> Scalar: ...  # type: ignore[override]
    def __pos__(self, *, recursive: bool = ...) -> Scalar: ...  # type: ignore[override]
    def __pow__(self, arg: Any) -> Scalar: ...  # type: ignore[override]
    def __radd__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Scalar: ...
    def __rfloordiv__(self, arg: Any) -> Scalar: ...
    def __rmod__(self, arg: Any) -> Scalar: ...  # type: ignore[override]
    def __rmul__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Scalar: ...
    def __rsub__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Scalar: ...
    def __rtruediv__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Scalar: ...
    def __sub__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Scalar: ...
    def __truediv__(self, arg: Any, *, recursive: bool = ...  # type: ignore[override]
        ) -> Scalar: ...
    @staticmethod
    def as_boolean(arg: Any, *, recursive: bool = ...) -> Boolean: ...
    def as_index(self) -> np.ndarray | bool: ...  # type: ignore[override]
    def identity(self) -> Boolean: ...  # type: ignore[override]
    def sum(self, axis: int | tuple[int, ...] | None = ..., *, value: bool = ...,
        builtins: bool | None = ..., recursive: bool = ..., masked: bool | None = ...,
        out: Any | None = ...) -> Scalar: ...

##########################################################################################
