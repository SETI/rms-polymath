##########################################################################################
# polymath/extensions/readonly_ops.py: Read-only/read-write and copying operations
##########################################################################################
"""Read-only and read-write state, and the copying of PolyMath objects.

A read-only object is protected from modification as far as Python allows, which makes it
safe to share the memory underlying it. These functions convert an object to read-only,
copy an object into a writable one, and assert that an object may be modified.
"""

import numpy as np
from polymath.qube import Qube

__all__ = ['as_readonly', 'copy', 'match_readonly', 'require_writable',
           'require_writeable']


@staticmethod
def _array_is_readonly(arg):
    """True if the argument is a read-only NumPy ndarray.

    False means that it is either a writable array or a scalar.

    Parameters:
        arg (Any): The object to test.

    Returns:
        bool: True if `arg` is a NumPy array that is not writable.
    """

    if not isinstance(arg, np.ndarray):
        return False

    return (not arg.flags['WRITEABLE'])


@staticmethod
def _array_to_readonly(arg):
    """Make the given argument read-only if it is a NumPy ndarray; then return it.

    Parameters:
        arg (Any): The object to make read-only.

    Returns:
        Any: `arg`, with its writable flag cleared if it is a NumPy array.
    """

    if not isinstance(arg, np.ndarray):
        return arg

    arg.flags['WRITEABLE'] = False
    return arg


def as_readonly(self, *, recursive=True):
    """Convert this object to read-only. It is modified in place and returned.

    If this object is already read-only, it is returned as is. Otherwise, the internal
    value and mask arrays are modified as necessary. Once this happens, the internal
    arrays will also cease to be writable in any other object that shares them.

    Note that :meth:`~polymath.Qube.as_readonly` cannot be undone. Use
    :meth:`~polymath.Qube.copy` to create a writable copy of a read-only object.

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
    """Make the read-only status of this object match that of another.

    Parameters:
        arg (Qube): The object whose read-only status is to be matched.

    Returns:
        Qube: This object, converted to read-only if `arg` is read-only.

    Raises:
        ValueError: If this object is read-only but `arg` is not.
    """

    if arg._readonly:
        return self.as_readonly()
    elif self._readonly:
        raise ValueError(f'{type(self).__name__} object is read-only')

    return self


def require_writeable(self, force=False):
    """Ensure that this object is writable.

    :meth:`~polymath.Qube.require_writable` is an alternative name for this method.

    Parameters:
        force (bool, optional): True to return a new copy if this object is read-only;
            otherwise, if this object is not writable, raise a ValueError.

    Returns:
        Qube: This object if already writable; otherwise a new writable copy.

    Raises:
        ValueError: If this object is read-only but `force` is False.
    """

    if self._readonly:
        if force:
            return self.copy(recursive=True, readonly=False)
        raise ValueError(f'{type(self).__name__} object is read-only')

    # Sometimes the array is writable but a shared mask is not
    if np.shape(self._mask) and not self._mask.flags['WRITEABLE']:
        self.remask(self._mask.copy())

    # It's possible that a derivative is read-only
    for key, deriv in self._derivs.items():
        if deriv._readonly:
            self._derivs[key] = deriv.copy(recursive=False, readonly=False)

    return self


def require_writable(self, force=False):
    """Ensure that this object is writable.

    This is an alternative name for :meth:`~polymath.Qube.require_writeable`.

    Parameters:
        force (bool, optional): True to return a new copy if this object is read-only;
            otherwise, if this object is not writable, raise a ValueError.

    Returns:
        Qube: This object if already writable; otherwise a new writable copy.

    Raises:
        ValueError: If this object is read-only but `force` is False.
    """

    return self.require_writeable(force=force)


def copy(self, *, recursive=True, readonly=False):
    """Deep copy operation with additional options.

    Parameters:
        recursive (bool, optional): True to copy the derivatives; False to return an
            object without derivatives.
        readonly (bool, optional): True to return a read-only copy; if this object is
            already read-only, the return is a shallow copy, which shares this object's
            arrays rather than duplicating them. Otherwise, this return is guaranteed to
            be an entirely new copy, independent of this object and suitable for
            modification.

    Returns:
        Qube: A copy of this object.
    """

    # Copying a readonly object is easy, because nothing in it can be modified
    if self._readonly and readonly:
        return self.clone(recursive=recursive)

    # Create a shallow copy
    obj = self.clone(recursive=False)

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
    """An independent, writable copy of this object.

    Returns:
        Qube: A deep copy of this object, writable.
    """

    return self.copy(recursive=True, readonly=False)

##########################################################################################
