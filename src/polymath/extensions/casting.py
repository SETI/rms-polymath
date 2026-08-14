##########################################################################################
# polymath/extensions/casting.py: Value tests and conversions between Qube subclasses
##########################################################################################

import numpy as np
import numbers
from polymath.qube import Qube, _NUMERIC_TYPES

__all__ = ['as_all_constant', 'as_one_bool', 'as_size_zero', 'as_this_type', 'cast',
           'is_one_false', 'is_one_true']


@staticmethod
def as_one_bool(value):
    """Convert a single value to a bool; leave other values unchanged."""

    if not isinstance(value, np.ndarray):
        return bool(value)

    return value


@staticmethod
def is_one_true(value):
    """True if the value is a single boolean True."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return False


@staticmethod
def is_one_false(value):
    """True if the value is a single boolean False."""

    if isinstance(value, (bool, np.bool_)):
        return not bool(value)

    return False


@staticmethod
def _is_one_value(value):
    """True if the value is a Python numeric or a NumPy numeric scalar."""

    if isinstance(value, _NUMERIC_TYPES):
        return True

    # The types that dominate the negative answer, kept off the ABC as well
    if isinstance(value, (Qube, np.ndarray, list, tuple)):
        return False

    return isinstance(value, numbers.Real)


def as_this_type(self, arg, *, recursive=True, coerce=True, op=''):
    """The argument converted to this class and data type.

    If the object is already of the correct class and type, it is returned unchanged.

    Parameters:
        arg (array-like, float, int, or bool): The object to the class of this object.
            If the argument is a scalar or NumPy ndarray, a new instance of this
            object's class is created.
        recursive (bool, optional): True to convert the derivatives as well.
        coerce (bool, optional): True to coerce the data type silently; False to leave
            the data type unchanged.
        op (str, optional): Name of operator to use in an error message.

    Returns:
        Qube: The argument converted to the type of this object.
    """

    # If the classes already match, we might return the argument as is
    if type(arg) is type(self):
        obj = arg
    else:
        obj = None

    # Initialize the new values and mask; track other attributes
    if not isinstance(arg, Qube):
        arg = Qube(arg, example=self, op=op)

    if arg._nrank != self._nrank:
        Qube._raise_incompatible_numers(op, self, arg)

    new_vals = arg._values
    new_mask = arg._mask
    new_unit = arg._unit
    has_derivs = bool(arg._derivs)
    is_readonly = arg._readonly

    # Convert the value types if necessary
    changed = False
    if coerce:
        casted = Qube._casted_to_dtype(new_vals, Qube._dtype(self._values))
        changed = casted is not new_vals
        new_vals = casted

    # Convert the unit if necessary
    if new_unit and not self._UNITS_OK:
        new_unit = None
        changed = True

    # Validate derivs
    if has_derivs and not self._DERIVS_OK:  # pragma: no cover
        # This should never happen because creating Qube with derivs when
        # _DERIVS_OK is False raises an error earlier
        changed = True
    if has_derivs and not recursive:
        changed = True

    # Construct the new object if necessary
    if changed or obj is None:
        obj = Qube.__new__(type(self))
        obj.__init__(new_vals, new_mask, unit=new_unit, drank=arg._drank,
                     example=self)
        is_readonly = False

    # Update the derivatives if necessary
    if recursive and has_derivs:
        derivs_changed = False
        new_derivs = {}
        for key, deriv in arg._derivs.items():
            new_deriv = self.as_this_type(deriv, recursive=False, coerce=False, op=op)
            if new_deriv is not deriv:
                derivs_changed = True
            new_derivs[key] = new_deriv

        if derivs_changed or (arg is not obj):
            if is_readonly:
                obj = obj.copy(recursive=False)
            obj.insert_derivs(new_derivs)

    return obj


@staticmethod
def _deriv_classes(classes):
    """The candidate classes to use when constructing a derivative.

    A derivative does not necessarily satisfy the constraint that defines the class of
    the object it belongs to. The derivative of a rotation matrix is not a rotation
    matrix: it is not orthogonal, and two of them can be added where two rotation
    matrices cannot. Any class defined by such a constraint names a more general
    substitute, which replaces it here.

    Parameters:
        classes (type, list, or tuple): One class or a list of candidate classes, as
            :meth:`cast` accepts.

    Returns:
        tuple: The candidate classes for a derivative, in the same order.
    """

    if isinstance(classes, type):
        classes = (classes,)

    return tuple(cls._DERIV_CLASS or cls for cls in classes)


def _castable_to(self, cls):
    """True if this object's content already satisfies every restriction of a class.

    This answers whether :meth:`cast` can build the new object without the validation
    that the constructor performs. It does not consider the numerator shape or rank,
    which :meth:`cast` checks for itself.

    Parameters:
        cls (type): The Qube subclass to test.

    Returns:
        bool: True if the data type, unit, denominator and derivatives of this object
        are all permitted by `cls`.
    """

    if not cls._DERIVS_OK and (self._derivs or self._drank):
        return False

    if not cls._UNITS_OK and self._unit is not None:
        return False

    dtype = Qube._dtype(self._values)
    if dtype == 'float':
        return cls._FLOATS_OK
    if dtype == 'int':
        return cls._INTS_OK

    return cls._BOOLS_OK


def cast(self, classes):
    """A shallow copy of this object casted to another Qube subclass.

    Parameters:
        classes (type or list): A Qube subclass or list of subclasses. The object
            will be casted to the first suitable class in the list.

    Returns:
        Qube: A shallow copy of this object. If the object is already of the selected
        class or if no suitable class is found, it is returned without modification.
    """

    # Convert a single class to a tuple
    if isinstance(classes, type):
        classes = (classes,)

    # For each class in the list...
    for cls in classes:

        # If this is already the class of this object, return it as is
        if cls is type(self):
            return self

        # Exclude the class if it is incompatible
        if cls._NUMER is not None and self._numer != cls._NUMER:
            continue
        if cls._NRANK is not None and self._nrank != cls._NRANK:
            continue

        # Construct the new object. The values, mask, unit and derivatives are
        # already valid, so the fast constructor suffices whenever the new class
        # imposes no restriction that the validating constructor would enforce.
        if self._castable_to(cls):
            obj = cls._new_from_parts(self._values, self._mask, nrank=self._nrank,
                                      drank=self._drank, unit=self._unit,
                                      example=self)
            if self._derivs:
                obj.insert_derivs(self._derivs)
            return obj

        obj = Qube.__new__(cls)
        obj.__init__(self._values, self._mask, derivs=self._derivs,
                     example=self)
        return obj

    # If no suitable class was found, return this object unmodified
    return self


def as_all_constant(self, constant=None, *, recursive=True):
    """A shallow, read-only copy of this object with constant values.

    Derivatives are all set to zero. The mask is unchanged.

    Parameters:
        constant (array-like, float, int, or bool, optional): The constant value for
            each item. This must have the same shape as this object's items. Use None
            for values of zero appropriate to the Qube subclass.

    Returns:
        Qube: A shallow copy of this object with constant values.
    """

    if constant is None:
        constant = self.zero()

    constant = self.as_this_type(constant, recursive=False)

    obj = self._clone_new_values(recursive=False)
    obj._set_values(Qube.broadcast(constant, obj)[0]._values)
    obj.as_readonly()

    if recursive:
        for key, deriv in self._derivs.items():
            obj.insert_deriv(key, deriv.as_all_constant(recursive=False))

    return obj


def as_size_zero(self, axis=0, *, recursive=True):
    """A shallow, read-only copy of this object with size zero.

    Parameters:
        axis (int, optional): The axis index (positive or negative) to collapse to
            length zero; the other axes are left unchanged. Use None for an object of
            shape (0,).

    Returns:
        Qube: A shallow copy of this object with size zero.

    Raises:
        ValueError: If `axis` is out of range.
    """

    obj = Qube.__new__(type(self))

    if self._shape == ():
        new_values = np.array([self._values])[:0]
        new_mask = np.array([self._mask])[:0]
    elif axis is None:
        new_values = self._values.ravel()[:0]
        new_mask = np.asarray(self._mask).ravel()[:0]
    else:
        self._require_axis_in_range(axis, self._ndims, 'as_size_zero()')

        # Leading axes are sliced in full; the trailing axes, including any item axes,
        # are left implicit
        a1 = axis % self._ndims
        indx = a1 * (slice(None),) + (slice(0, 0),)

        new_values = self._values[indx]

        if np.shape(self._mask):
            new_mask = self._mask[indx]
        else:
            # For scalar mask, create array matching new_values shape
            new_mask = np.full(new_values.shape[:len(new_values.shape) - self._rank],
                               self._mask, dtype=np.bool_)

    obj.__init__(new_values, new_mask, example=self)

    if recursive:
        for key, deriv in self._derivs.items():
            obj.insert_deriv(key, deriv.as_size_zero(axis=axis, recursive=False))

    return obj

##########################################################################################
