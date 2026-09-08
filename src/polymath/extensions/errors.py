##########################################################################################
# polymath/extensions/errors.py: Error message support
##########################################################################################
"""Construction of the error messages raised by PolyMath operations.

The functions here raise the exceptions that operations share, so that a message naming
the operation, the classes involved, and the offending shapes is phrased the same way
everywhere. They also perform the checks that many operations begin with, such as
requiring an object to have no denominator.
"""

import numpy as np
from polymath.qube import Qube

__all__ = []


def _opstr(self, /, op):
    """An operation string to use in an error message for this class.

    Parameters:
        op (str): Name of the operation.

    Returns:
        str: The class name followed by the operation, updated for an error message.
    """

    name = self.__name__ if isinstance(self, type) else type(self).__name__

    if not op:
        return name

    if op[0].isalpha():
        return name + '.' + op

    return name + ' "' + op + '"'


def _disallow_denom(self, op):
    """Raise ValueError if this object has a denominator.

    Parameters:
        op (str): Name of the operation to appear in the error message.

    Raises:
        ValueError: If this object has a denominator.
    """

    if self._drank:
        raise ValueError(self._opstr(op) + ' does not support denominators')


def _require_scalar(self, op):
    """Raise ValueError if this object has a numerator rank greater than zero.

    Parameters:
        op (str): Name of the operation to appear in the error message.

    Raises:
        ValueError: If this object has a numerator rank greater than zero.
    """

    if self._nrank:
        raise ValueError(self._opstr(op) + ' requires scalar items')


def _require_axis_in_range(self, axis, rank, op, name='axis'):
    """Raise ValueError if a given axis index is out of range.

    Parameters:
        axis (int): Axis index, positive or negative.
        rank (int): Rank of an array for indexing.
        op (str): Name of the operation to appear in the error message.
        name (str, optional): Name of the axis variable, for the error message.

    Raises:
        ValueError: If `axis` is less than -`rank` or greater than or equal to `rank`.
    """

    if axis < -rank or axis >= rank:
        opstr = self._opstr(op)
        raise ValueError(f'{opstr} {name} is out of range ({-rank},{rank}): {axis}')


def _raise_unsupported_op(op, /, obj1, obj2=None):
    """Raise a TypeError or ValueError for an unsupported operation.

    Parameters:
        op (str): Name of the operation to appear in the error message.
        obj1 (Qube): The left operand of the operation.
        obj2 (QubeLike | None, optional): The right operand of the operation. If None, the
            operation is reported as unsupported for `obj1` alone.

    Raises:
        TypeError: If `obj2` is None or its type is unsupported.
        ValueError: If either operand is a list, tuple or NumPy array, in which case the
            item shapes are reported as incompatible.
    """

    opstr = obj1._opstr(op)

    if obj2 is None:
        raise TypeError(f'{opstr} operation is not supported')

    if (isinstance(obj1, (list, tuple, np.ndarray)) or
            isinstance(obj2, (list, tuple, np.ndarray))):

        if isinstance(obj1, Qube):
            shape1 = obj1._numer
        else:
            shape1 = np.shape(obj1)

        if isinstance(obj2, Qube):
            shape2 = obj2._numer
        else:
            shape2 = np.shape(obj2)

        raise ValueError(f'unsupported operand item for {opstr}: {shape1}, {shape2}')

    raise TypeError(f'unsupported operand type for {opstr}: {type(obj2)}')


def _raise_incompatible_shape(op, /, obj1, obj2):
    """Raise a ValueError for incompatible object shapes.

    Parameters:
        op (str): Name of the operation to appear in the error message.
        obj1 (Qube): The left operand of the operation.
        obj2 (Qube): The right operand of the operation.

    Raises:
        ValueError: Always, quoting the shape of each operand.
    """

    opstr = obj1._opstr(op)
    raise ValueError(f'incompatible object shapes for {opstr}: '
                     f'{obj1._shape}, {obj2._shape}')


def _raise_incompatible_numers(op, /, obj1, obj2):
    """Raise a ValueError for incompatible numerators in an operation.

    Parameters:
        op (str): Name of the operation to appear in the error message.
        obj1 (Qube): The left operand of the operation.
        obj2 (Qube): The right operand of the operation.

    Raises:
        ValueError: Always, quoting the numerator shape of each operand.
    """

    opstr = obj1._opstr(op)
    raise ValueError(f'incompatible numerator shapes for {opstr}: '
                     f'{obj1._numer}, {obj2._numer}')


def _raise_incompatible_denoms(op, /, obj1, obj2):
    """Raise a ValueError for incompatible denominators in an operation.

    Parameters:
        op (str): Name of the operation to appear in the error message.
        obj1 (Qube): The left operand of the operation.
        obj2 (Qube): The right operand of the operation.

    Raises:
        ValueError: Always, quoting the denominator shape of each operand.
    """

    opstr = obj1._opstr(op)
    raise ValueError(f'incompatible denominator shapes for {opstr}: '
                     f'{obj1._denom}, {obj2._denom}')


def _raise_dual_denoms(op, /, obj1, obj2):
    """Raise a ValueError for denominators on both operands.

    Parameters:
        op (str): Name of the operation to appear in the error message.
        obj1 (Qube): The left operand of the operation.
        obj2 (Qube): The right operand of the operation.

    Raises:
        ValueError: Always, because only one operand may have a denominator.
    """

    opstr = obj1._opstr(op)
    raise ValueError(f'only one operand of {opstr} can have a denominator')

##########################################################################################
