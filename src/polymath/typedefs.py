##########################################################################################
# polymath/typedefs.py
##########################################################################################
"""Public type aliases naming the values that the PolyMath constructors accept.

These aliases exist for documentation and for downstream code that uses type annotations.
Each alias is an ordinary runtime object, so it can be imported and used in an annotation
anywhere::

    from polymath import Scalar, Vector3
    from polymath.typedefs import Vector3Like

    def speed(velocity: Vector3Like) -> Scalar:
        return Vector3.as_vector3(velocity).norm()

Each alias names what the corresponding class converts, which is broader than the class
itself: any :class:`~polymath.Qube` subclass qualifies, because the constructors re-wrap
any of them. Annotating with an alias therefore documents intent and rules out unrelated
types such as dictionaries and strings, but it does not restrict an argument to objects of
one PolyMath class alone.
"""

from collections.abc import Iterator
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np

from polymath.qube import Qube

__all__ = ['BooleanLike', 'IntValsType', 'MaskType', 'Matrix3Like', 'MatrixLike',
           'PairLike', 'QuaternionLike', 'QubeLike', 'ScalarLike', 'ValsType',
           'Vector3Like', 'VectorLike']

# Float arrays with the specified lower limit on dimensions and/or trailing axes
_Array  : TypeAlias = np.ndarray[tuple[int, ...],
                                 np.dtype[np.number[Any] | np.bool_]]
_Array1D: TypeAlias = np.ndarray[tuple[int, *tuple[int, ...]],
                                 np.dtype[np.number[Any] | np.bool_]]
_Array2D: TypeAlias = np.ndarray[tuple[int, int, *tuple[int, ...]],
                                 np.dtype[np.number[Any] | np.bool_]]
_Array2 : TypeAlias = np.ndarray[tuple[*tuple[int, ...], Literal[2]],
                                 np.dtype[np.number[Any] | np.bool_]]
_Array3 : TypeAlias = np.ndarray[tuple[*tuple[int, ...], Literal[3]],
                                 np.dtype[np.number[Any] | np.bool_]]
_Array4 : TypeAlias = np.ndarray[tuple[*tuple[int, ...], Literal[4]],
                                 np.dtype[np.number[Any] | np.bool_]]
_Array33: TypeAlias = np.ndarray[tuple[*tuple[int, ...], Literal[3], Literal[3]],
                                 np.dtype[np.number[Any] | np.bool_]]

# Typed arrays
_BoolArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
_IntArray : TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.integer[Any]]]

# Numeric ArrayLike type, described by a protocol rather than by list and tuple because
# both are invariant in their member type: a list[float] does not match a list whose
# member type is the union below. Matching a protocol is structural instead, so a sequence
# qualifies at any depth of nesting, however deep. A str does not qualify, because it
# defines no __reversed__ and its __getitem__ returns another str. A member may be an
# array or a PolyMath object as well as a number, because np.asarray() stacks a sequence
# of them into one array of higher rank.
_Scalar: TypeAlias = float | int | bool | np.bool_


class _NestedSequence(Protocol):
    """A sequence of numbers, arrays or PolyMath objects, nested to any depth."""

    def __len__(self, /) -> int: ...
    def __getitem__(self, index: int, /) -> '_SeqMember': ...
    def __contains__(self, x: object, /) -> bool: ...
    def __iter__(self, /) -> 'Iterator[_SeqMember]': ...
    def __reversed__(self, /) -> 'Iterator[_SeqMember]': ...
    def count(self, value: Any, /) -> int: ...
    def index(self, value: Any, /) -> int: ...


# Named after the class so that it can name the class in turn.
_SeqMember: TypeAlias = Qube | _Scalar | _Array | _NestedSequence
_ArrayLike: TypeAlias = _NestedSequence

BooleanLike: TypeAlias = Qube | _Array | _ArrayLike | _Scalar
"""Any value convertible to a :class:`~polymath.Boolean`: a PolyMath object, a numeric
array, a nested sequence of numbers, or a single number."""

ScalarLike: TypeAlias = Qube | _Array | _ArrayLike | _Scalar
"""Any value convertible to a :class:`~polymath.Scalar`: a PolyMath object, a numeric
array, a nested sequence of numbers, or a single number."""

PairLike: TypeAlias = Qube | _Array2 | _ArrayLike | float | int
"""Any value convertible to a :class:`~polymath.Pair`: a PolyMath object, a numeric array
whose last axis has length two, or a nested sequence of numbers. As a special case, a
single value becomes a Pair with the value repeated."""

VectorLike: TypeAlias = Qube | _Array1D | _ArrayLike
"""Any value convertible to a :class:`~polymath.Vector`: a PolyMath object, a numeric
array of one or more axes, or a nested sequence of numbers."""

Vector3Like: TypeAlias = Qube | _Array3 | _ArrayLike
"""Any value convertible to a :class:`~polymath.Vector3`: a PolyMath object, a numeric
array whose last axis has length three, or a nested sequence of numbers."""

MatrixLike: TypeAlias = Qube | _Array2D | _ArrayLike
"""Any value convertible to a :class:`~polymath.Matrix`: a PolyMath object, a numeric
array of two or more axes, or a nested sequence of numbers."""

Matrix3Like: TypeAlias = Qube | _Array33 | _ArrayLike
"""Any value convertible to a :class:`~polymath.Matrix3`: a PolyMath object, a numeric
array whose last two axes each have length three, or a nested sequence of numbers."""

QuaternionLike: TypeAlias = Qube | _Array4 | _ArrayLike
"""Any value convertible to a :class:`~polymath.Quaternion`: a PolyMath object, a numeric
array whose last axis has length four, or a nested sequence of numbers."""

QubeLike: TypeAlias = Qube | _Array | _ArrayLike | _Scalar
"""Any value convertible to a :class:`~polymath.Qube` subclass: a PolyMath object, a
numeric array, a nested sequence of numbers, or a single number."""

ValsType: TypeAlias = _Scalar | _Array
"""Any value that might occupy the `.vals` attribute if a Qube."""

MaskType: TypeAlias = bool | np.bool_ | _BoolArray
"""Any value that might occupy the `.mask` attribute if a Qube."""

IntValsType: TypeAlias = int | _IntArray
"""Any value that might occupy the `.vals` attribute if a Qube and must also be integral.
"""

##########################################################################################
