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

from polymath.vector import Vector
from polymath.scalar import Scalar
from polymath.typedefs import ScalarLike, Vector3Like


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
    def as_vector3(arg: Any, *, recursive: bool = ...) -> Vector3: ...
    @staticmethod
    def from_cylindrical(radius: ScalarLike, longitude: ScalarLike, z: ScalarLike = ...,
        *, recursive: bool = ...) -> Vector3: ...
    @staticmethod
    def from_ra_dec_length(ra: ScalarLike, dec: ScalarLike, length: ScalarLike = ..., *,
        recursive: bool = ...) -> Vector3: ...
    @staticmethod
    def from_scalars(x: ScalarLike, y: ScalarLike, z: ScalarLike, *,  # type: ignore[override]
        recursive: bool = ..., readonly: bool = ...) -> Vector3: ...
    def latitude(self, *, recursive: bool = ...) -> Scalar: ...
    def longitude(self, *, recursive: bool = ...) -> Scalar: ...
    def offset_angles(self, vector: Vector3Like, *, recursive: bool = ...
        ) -> tuple[Scalar, Scalar]: ...
    def spin(self, pole: Vector3Like, angle: ScalarLike | None = ..., *,
        recursive: bool = ...) -> Vector3: ...
    def to_cylindrical(self, *, recursive: bool = ...
        ) -> tuple[Scalar, Scalar, Scalar]: ...
    def to_ra_dec_length(self, *, recursive: bool = ...
        ) -> tuple[Scalar, Scalar, Scalar]: ...

##########################################################################################
