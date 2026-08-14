##########################################################################################
# polymath/extensions/readonly_ops.py: Read-only/read-write and copying operations
##########################################################################################

import numpy as np
from polymath.qube import Qube

__all__ = ['as_readonly', 'copy', 'match_readonly', 'require_writable',
           'require_writeable']


@staticmethod
def _array_is_readonly(arg):
    """True if the argument is a read-only NumPy ndarray.

    False means that it is either a writable array or a scalar.
    """

    if not isinstance(arg, np.ndarray):
        return False

    return (not arg.flags['WRITEABLE'])


@staticmethod
def _array_to_readonly(arg):
    """Make the given argument read-only if it is a NumPy ndarray; then return it."""

    if not isinstance(arg, np.ndarray):
        return arg

    arg.flags['WRITEABLE'] = False
    return arg


def as_readonly(self, *, recursive=True):
    """Convert this object to read-only. It is modified in place and returned.

    If this object is already read-only, it is returned as is. Otherwise, the internal
    _values and _mask arrays are modified as necessary. Once this happens, the
    internal arrays will also cease to be writable in any other object that shares
    them.

    Note that `as_readonly()` cannot be undone. Use `copy()` to create a writable copy
    of a readonly object.

    Parameters:
        recursive (bool, optional): True also to convert the derivatives to read-only;
            False to strip the derivatives.

    Returns:
        Qube: This object, converted to read-only if necessary.
    """

    # If it is already read-only, return
    if self._readonly:
        return self

    # Update the value if it is an array
    Qube._array_to_readonly(self._values)
    Qube._array_to_readonly(self._mask)
    self._readonly = True

    # Update anything cached
    if not Qube._DISABLE_CACHE:
        # Snapshot: the loop replaces entries, and a cached object can reach back into
        # this same dictionary
        for key, value in list(self._cache.items()):
            if isinstance(value, Qube):
                self._cache[key] = value.as_readonly(recursive=recursive)

    # Update the derivatives
    if recursive:
        for key in self._derivs:
            self._derivs[key].as_readonly()

    return self


def match_readonly(self, arg):
    """Convert the read-only status of this object equal to that of another.

    Parameters:
        arg (Qube): An existing Qube subclass.

    Returns:
        Qube: This object converted to read-only.

    Raises:
        ValueError: If this object is read-only but the `arg` is not.
    """

    if arg._readonly:
        return self.as_readonly()
    elif self._readonly:
        raise ValueError(f'{type(self).__name__} object is read-only')

    return self


def require_writeable(self, force=False):
    """Ensure that this object is writeable.

    Parameters:
        force (bool, optional): True to return a new copy if this object is read-only;
            otherwise, if this object is not writeable, raise a ValueError.

    Returns:
        Qube: This object if already writeable; otherwise a new writeable copy.

    Raises:
        ValueError: If this object is read-only but `force` is False.
    """

    if self._readonly:
        if force:
            return self.copy(recursive=True, readonly=True)
        raise ValueError(f'{type(self).__name__} object is read-only')

    # Sometimes the array is writeable but a shared mask is not
    if np.shape(self._mask) and not self._mask.flags['WRITEABLE']:
        self.remask(self._mask.copy())

    # It's possible that a derivative is read-only
    for key, deriv in self._derivs.items():
        if deriv._readonly:
            self._derivs[key] = deriv.copy(recursive=False, readonly=False)

    return self


def require_writable(self, force=False):
    """Ensure that this object is writeable.

    DEPRECATED NAME; use require_writeable().

    Parameters:
        force (bool, optional): True to return a new copy if this object is read-only;
            otherwise, if this object is not writeable, raise a ValueError.

    Returns:
        Qube: This object if already writeable; otherwise a new writeable copy.

    Raises:
        ValueError: If this object is read-only but `force` is False.
    """

    return self.require_writeable(force=force)


def copy(self, *, recursive=True, readonly=False):
    """Deep copy operation with additional options.

    Parameters:
        recursive (bool, optional): True to copy the derivatives; False, to return an
            object without derivatives.
        readonly (bool, optional): True to return a read-only copy, or this object if
            it is already read-only. Otherwise, this return is guaranteed to be an
            entirely new copy, independent of this object and suitable for
            modification.

    Returns:
        Qube: A copy of this object.
    """

    # Create a shallow copy
    obj = self.clone(recursive=False)

    # Copying a readonly object is easy
    if self._readonly and readonly:
        return obj

    # Copy the values
    if self._is_array:
        obj._values = self._values.copy()
    else:
        obj._values = self._values

    # Copy the mask
    if isinstance(self._mask, np.ndarray):
        obj._mask = self._mask.copy()
    else:
        obj._mask = self._mask

    obj._cache = {}

    # Set the read-only state
    if readonly:
        obj.as_readonly()
    else:
        obj._readonly = False

    # Make the derivatives read-only if necessary
    if recursive:
        for key, deriv in self._derivs.items():
            obj.insert_deriv(key, deriv.copy(recursive=False, readonly=readonly))

    return obj


# Python-standard copy function
def __copy__(self):
    """An independent, writeable copy of this object."""

    return self.copy(recursive=True, readonly=False)

##########################################################################################
