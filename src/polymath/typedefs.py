##########################################################################################
# polymath/typedefs.py
##########################################################################################
"""Public type aliases naming the values that the PolyMath constructors accept.

The `src` tree carries no inline annotations, so these aliases exist for the benefit of
downstream code that does annotate. Each alias is an ordinary runtime object, so it can be
imported and used in an annotation anywhere::

    from polymath import Scalar
    from polymath.typedefs import ScalarLike

    def halve(value: ScalarLike) -> Scalar:
        return Scalar.as_scalar(value) / 2

Each alias names what the corresponding class converts, which is broader than the class
itself: any :class:`~polymath.Qube` subclass qualifies, because the constructors re-wrap
any of them. Annotating with an alias therefore documents intent and rules out unrelated
types such as dictionaries and strings, but it does not restrict an argument to objects of
one PolyMath class alone.
"""

from typing import Any, Literal, TypeAlias

import numpy as np

from polymath.qube import Qube

__all__ = ['BooleanLike', 'MaskType', 'Matrix3Like', 'MatrixLike', 'PairLike',
           'QuaternionLike', 'QubeLike', 'ScalarLike', 'ValsType', 'Vector3Like',
           'VectorLike']

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

# A boolean array
_BoolArray : TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]

# Numeric ArrayLike type. The nested sequence is recursive, so its members are written as
# forward references; a non-quoted alias cannot refer to itself before it is defined.
_Scalar: TypeAlias = float | int | bool | np.bool_
_ArrayLike: TypeAlias = (list['_Scalar | _ArrayLike'] |
                         tuple['_Scalar | _ArrayLike', ...])

BooleanLike: TypeAlias = Qube | _Array | _ArrayLike | _Scalar
"""Any value convertible to a :class:`~polymath.Boolean`: a PolyMath object, a numeric
array, a nested sequence of numbers, or a single number."""

ScalarLike: TypeAlias = Qube | _Array | _ArrayLike | _Scalar
"""Any value convertible to a :class:`~polymath.Scalar`: a PolyMath object, a numeric
array, a nested sequence of numbers, or a single number."""

PairLike: TypeAlias = Qube | _Array2 | _ArrayLike
"""Any value convertible to a :class:`~polymath.Pair`: a PolyMath object, a numeric array
whose last axis has length two, or a nested sequence of numbers."""

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

##########################################################################################
