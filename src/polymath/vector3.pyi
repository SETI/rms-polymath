##########################################################################################
# polymath/vector3.pyi
##########################################################################################
"""Type stub for :mod:`polymath.vector3`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. These stubs describe the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
taken from the docstrings wherever those state one unambiguously, and are left as `Any`
where they do not, rather than guessed at.
"""

from typing import Any

from polymath.qube import _Arraylike, _ShapeOrTuple
from polymath.vector import Vector

__all__ = ['Vector3']

class Vector3(Vector):
    AXES: tuple[Any, ...]
    IDENTITY: Vector3
    MASKED: Vector3
    ONES: Vector3
    XAXIS: Vector3
    YAXIS: Vector3
    ZAXIS: Vector3
    ZERO: Vector3
    ZERO_POS_VEL: Vector3
    @staticmethod
    def as_vector3(arg: Any, *, recursive: bool = ...) -> _Arraylike: ...
    @staticmethod
    def from_cylindrical(radius: _Arraylike, longitude: _Arraylike, z: _Arraylike = ...,
        *, recursive: bool = ...) -> _Arraylike: ...
    @staticmethod
    def from_ra_dec_length(ra: _Arraylike, dec: _Arraylike, length: _Arraylike = ..., *,
        recursive: bool = ...) -> _Arraylike: ...
    @staticmethod
    def from_scalars(x: Any, y: Any, z: Any, *, recursive: bool = ...,  # type: ignore[override]
        readonly: bool = ...) -> _Arraylike: ...
    def latitude(self, *, recursive: bool = ...) -> _Arraylike: ...
    def longitude(self, *, recursive: bool = ...) -> _Arraylike: ...
    def offset_angles(self, vector: _Arraylike, *,
        recursive: bool = ...) -> _ShapeOrTuple: ...
    def spin(self, pole: _Arraylike, angle: _Arraylike | None = ..., *,
        recursive: bool = ...) -> _Arraylike: ...
    def to_cylindrical(self, *, recursive: bool = ...) -> _ShapeOrTuple: ...
    def to_ra_dec_length(self, *, recursive: bool = ...) -> _ShapeOrTuple: ...

##########################################################################################
