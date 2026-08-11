##########################################################################################
# polymath/extensions/masking.py: Mask construction and object mask operations
##########################################################################################

import numpy as np
import numbers
from polymath.qube import Qube

__all__ = ['and_', 'as_all_masked', 'as_mask_where_nonzero',
           'as_mask_where_nonzero_or_masked', 'as_mask_where_zero',
           'as_mask_where_zero_or_masked', 'as_one_masked', 'collapse_mask',
           'count_masked', 'count_unmasked', 'expand_mask', 'is_all_masked',
           'masked_single', 'or_', 'remask', 'remask_or', 'without_mask']

##########################################################################################
# Mask construction
##########################################################################################


@staticmethod
def _as_mask(arg, *, invert=False, masked_value=True, opstr=''):
    """This argument converted to a scalar bool or boolean Numpy array.

    Parameters:
        arg: The object to convert to a mask.
        invert (bool, optional): True to return the logical not of the mask.
        masked_value (bool, optional): The value to use where the input argument is
           masked. This value is used _after_ `invert` is applied.
        opstr (str, optional): Name of operation to include in any error message.

    Returns:
        (bool or NumPy.ndarray): bool or boolean array suitable for us as a mask.

    Raises:
        TypeError: If the data type of `arg` is invalid for a mask.
    """

    # Handle most common cases first
    if isinstance(arg, (numbers.Real, np.bool_, type(None))):
        return bool(arg) != invert

    if type(arg) is np.ndarray:     # exact type, not a subclass
        if arg.dtype.kind == 'b' and not invert:
            return arg
        elif invert:
            return arg == 0
        else:
            return arg != 0

    # Convert a list or tuple to something else
    if isinstance(arg, (list, tuple)):
        if Qube._has_qube(arg):
            arg = Qube.stack(*arg)
        elif Qube._has_masked_array(arg):
            arg = np.ma.stack(arg)
        else:
            arg = np.array(arg)
            return Qube._as_mask(arg, invert=invert,  masked_value=masked_value,
                                 opstr=opstr)

    # Handle an object with a possible mask
    if isinstance(arg, Qube):
        mask = arg._mask
        arg = arg._values
    elif isinstance(arg, np.ma.MaskedArray):
        mask = arg.mask
        arg = arg.data
    else:
        _opstr = ' ' + opstr if opstr else ''
        raise TypeError(f'invalid{_opstr} mask type: {type(arg).__name__}')

    # Handle a shapeless mask
    if isinstance(mask, (bool, np.bool_)):
        if mask:                        # entirely masked
            return bool(masked_value)
        else:                           # entirely unmasked
            return Qube._as_mask(arg, invert=invert, masked_value=masked_value,
                                 opstr=opstr)

    # Copy the arg and merge the mask
    if invert:
        merged = (arg == 0)
    else:
        merged = (arg != 0)

    merged[mask] = masked_value
    return merged


@staticmethod
def _suitable_mask(arg, shape, *, collapse=False, broadcast=False, invert=False,
                   masked_value=True, check=False, opstr=''):
    """This argument converted to a scalar bool or boolean Numpy array of suitable
    shape to use as a mask.

    Parameters:
        arg: The object to convert to a mask.
        shape (tuple): Shape of the required mask.
        collapse (bool, optional): True to merge the extraneous axes of a mask if its
            rank is greater than that of the given shape.
        broadcast (bool, optional): True to broadcast this mask if its rank is less
            than that of the given shape.
        invert (bool, optional): True to return the logical not of the mask.
        masked_value (bool, optional): The value to use where the input argument is
           nmasked. This value is used _after_ `invert` is applied.
        check (bool, optional): True to check for an array containing all False
            values, and if so, replace it with a single value of False.
        opstr (str, optional): Name of operation to include in any error message.

    Returns:
        (bool or NumPy.ndarray): bool or boolean mask array.

    Raises:
        TypeError: If the data type of `arg` is invalid for a mask.
        ValueError: If the mask is incompatible with the specified `shape`.
    """

    mask = Qube._as_mask(arg, invert=invert, masked_value=masked_value, opstr=opstr)

    if isinstance(mask, bool):
        return mask

    if mask.shape == shape:
        if check and not np.any(mask):
            return False
        return mask

    new_rank = len(shape)
    if collapse and mask.ndim > new_rank:
        axes = tuple(range(new_rank, mask.ndim))
        mask = np.any(mask, axis=axes)
        if not isinstance(mask, np.ndarray):
            return bool(mask)
        if mask.shape == shape:
            return mask

    if broadcast:
        try:
            mask = np.broadcast_to(mask, shape)
        except ValueError:
            pass
        else:
            Qube._array_to_readonly(mask)
            return mask

    opstr_ = opstr + ' ' if opstr else ''
    raise ValueError(f'{opstr_}object and mask shape mismatch: '
                     f'{shape}, {mask.shape}')

##########################################################################################
# Mask combination
##########################################################################################


@staticmethod
def or_(*masks):
    """The logical "or" of two or more masks, avoiding array operations if possible.

    Parameters:
        *masks (array-like or bool): One or more boolean masks.

    Returns:
        (np.ndarray or bool): New mask array or bool.
    """

    # Two inputs is most common
    if len(masks) == 2:
        mask0 = masks[0]
        mask1 = masks[1]

        if isinstance(mask0, (bool, np.bool_)):
            if mask0:
                return True
            else:
                return mask1

        if isinstance(mask1, (bool, np.bool_)):
            if mask1:
                return True
            else:
                return mask0

        if mask0 is mask1:          # can happen when objects share masks
            return mask0

        return mask0 | mask1

    # Handle one input
    if len(masks) == 1:
        return masks[0]

    # Three or more: a single True settles it, and the rest combine in one pass
    arrays = []
    for mask in masks:
        if isinstance(mask, (bool, np.bool_)):
            if mask:
                return True
        else:
            arrays.append(mask)

    if not arrays:
        return False

    result = arrays[0]
    for mask in arrays[1:]:
        if mask is not result:      # can happen when objects share masks
            result = result | mask

    return result


@staticmethod
def and_(*masks):
    """The logical "and" of two or more masks, avoiding array operations if possible.

    Parameters:
        *masks (array-like or bool): One or more boolean masks.

    Returns:
        (np.ndarray or bool): New mask array or bool.
    """

    # Two inputs is most common
    if len(masks) == 2:
        mask0 = masks[0]
        mask1 = masks[1]

        if isinstance(mask0, (bool, np.bool_)):
            if mask0:
                return mask1
            else:
                return False

        if isinstance(mask1, (bool, np.bool_)):
            if mask1:
                return mask0
            else:
                return False

        if mask0 is mask1:          # can happen when objects share masks
            return mask0

        return mask0 & mask1

    # Handle one input
    if len(masks) == 1:
        return masks[0]

    # Three or more: a single False settles it, and the rest combine in one pass
    arrays = []
    for mask in masks:
        if isinstance(mask, (bool, np.bool_)):
            if not mask:
                return False
        else:
            arrays.append(mask)

    if not arrays:
        return True

    result = arrays[0]
    for mask in arrays[1:]:
        if mask is not result:      # can happen when objects share masks
            result = result & mask

    return result

##########################################################################################
# Object mask operations
##########################################################################################


def is_all_masked(self):
    """True if this object is entirely masked."""

    return np.all(self._mask)


def count_masked(self):
    """The number of masked items in this object."""

    if isinstance(self._mask, np.ndarray):
        return np.sum(self._mask)

    return self._size if self._mask else 0


def count_unmasked(self):
    """The number of unmasked items in this object."""

    if isinstance(self._mask, np.ndarray):
        return self._size - np.sum(self._mask)

    return 0 if self._mask else self._size


def masked_single(self, *, recursive=True):
    """An object of this subclass containing one masked value."""

    if not self._rank:
        new_value = self._default
    else:
        new_value = self._default.copy()

    obj = Qube.__new__(type(self))
    obj.__init__(new_value, True, example=self)

    if recursive and self._derivs:
        for key, value in self._derivs.items():
            obj.insert_deriv(key, value.masked_single(recursive=False))

    obj.as_readonly()
    return obj


def without_mask(self, *, recursive=True):
    """A shallow copy of this object without its mask. Note that masked values will be
    revealed.

    Parameters:
        recursive (bool, optional): True to unmask any derivatives; False to strip
            derivatives.

    Returns:
        Qube: This object without a mask.
    """

    obj = self.clone(recursive=recursive)
    obj._set_mask(False)

    if recursive:
        for key, deriv in self._derivs.items():
            obj.insert_deriv(key, deriv.without_mask())

    return obj


def as_all_masked(self, *, recursive=True):
    """A shallow copy of this object with everything masked.

    Parameters:
        recursive (bool, optional): True to mask any derivatives; False to strip
            derivatives.

    Returns:
        Qube: This object but fully masked.
    """

    obj = self.clone(recursive=recursive)
    obj._set_mask(True)

    if recursive:
        for key, deriv in self._derivs.items():
            obj.insert_deriv(key, deriv.as_all_masked(recursive=False))

    return obj


def as_one_masked(self, *, recursive=True):
    """This object reduced to shape () and masked.

    Parameters:
        recursive (bool, optional): True to mask any derivatives; False to strip
            derivatives.

    Returns:
        Qube: This object but fully masked and with shape ()
    """

    return self.flatten()[0].as_all_masked()


def remask(self, mask, *, recursive=True, check=True):
    """A shallow copy of this object with a replaced mask.

    This is much quicker than masked_where(), for cases where only the mask of this
    object is changing.

    Parameters:
        mask (array-like or bool): The new mask to be applied to the object.
        recursive (bool, optional): True to apply the same mask to any derivatives.
        check (bool, optional): True to check for an array containing all False
            values, and if so, replace it with a single value of False.

    Returns:
        Qube: A shallow copy of this object with a new mask.

    Raises:
        TypeError: If the data type of `mask` is invalid.
        ValueError: If the mask is incompatible with the required shape.
    """

    mask = Qube._suitable_mask(mask, self._shape, check=check)

    # Construct the new object
    obj = self.clone(recursive=False)
    obj._set_mask(mask)

    if recursive:
        for key, deriv in self._derivs.items():
            obj.insert_deriv(key, deriv.remask(mask, recursive=False, check=False))

    return obj


def remask_or(self, mask, *, recursive=True, check=True):
    """A shallow copy of this object, in which the current mask is "or-ed" with the
    given mask.

    This is much quicker than masked_where(), for cases where only the mask is
    changing.

    Parameters:
        mask (array-like or bool): The new mask to be applied to the object.
        recursive (bool, optional): True to apply the same mask to any derivatives.
        check (bool, optional): True to check for an array containing all False
            values, and if so, replace it with a single value of False.

    Returns:
        Qube: A shallow copy of this object with a new mask.

    Raises:
        TypeError: If the data type of `mask` is invalid for a mask.
        ValueError: If the mask is incompatible with the required shape.
    """

    mask = Qube._suitable_mask(mask, self._shape, check=check)

    # Construct the new object
    obj = self.clone(recursive=False)
    obj._set_mask(Qube.or_(self._mask, mask))

    if recursive:
        for key, deriv in self._derivs.items():
            obj.insert_deriv(key, deriv.remask(mask, recursive=False, check=False))

    return obj


def expand_mask(self, *, recursive=True):
    """A shallow copy where a single mask value of True or False is converted to an
    array.

    If the object's mask is already an array, it is returned unchanged.

    Parameters:
        recursive (bool, optional): True to expand the mask of any derivatives.

    Returns:
        Qube: A shallow copy of this object with an expanded mask.
    """

    if np.shape(self._mask) and not (recursive and self._derivs):
        return self

    # Clone the object only if necessary
    obj = None
    if not isinstance(self._mask, np.ndarray):
        obj = self.clone(recursive=True)
        if obj._mask:
            obj._set_mask(np.ones(self._shape, dtype=np.bool_))
        else:
            obj._set_mask(np.zeros(self._shape, dtype=np.bool_))

    # Clone any derivs only if necessary
    new_derivs = {}
    if recursive:
        for key, deriv in self._derivs.items():
            mask_before = deriv._mask
            new_deriv = deriv.expand_mask(recursive=False)
            if mask_before is not new_deriv._mask:
                new_derivs[key] = new_deriv

    # If nothing has changed, return self
    if obj is None and not new_derivs:
        return self

    # Return the modified object
    if obj is None:
        obj = self.clone(recursive=True)

    for key, deriv in new_derivs.items():
        obj.insert_deriv(key, deriv, override=True)

    return obj


def collapse_mask(self, *, recursive=True):
    """A shallow copy where a mask entirely containing either True or False is
    converted to a single boolean.

    Parameters:
        recursive (bool, optional): True to collapse the mask of any derivatives.

    Returns:
        Qube: A shallow copy of this object with a collapsed mask.
    """

    if not isinstance(self._mask, np.ndarray) and not (recursive and self._derivs):
        return self

    # Clone the object only if necessary
    obj = None
    if np.shape(self._mask):
        if not np.any(self._mask):
            obj = self.clone(recursive=True)
            obj._set_mask(False)
        elif np.all(self._mask):
            obj = self.clone(recursive=True)
            obj._set_mask(True)

    # Clone any derivs only if necessary
    new_derivs = {}
    if recursive:
        for key, deriv in self._derivs.items():
            mask_before = deriv._mask
            new_deriv = deriv.collapse_mask(recursive=False)
            if mask_before is not new_deriv._mask:
                new_derivs[key] = new_deriv

    # If nothing has changed, return self
    if obj is None and not new_derivs:
        return self

    # Return the modified object
    if obj is None:
        obj = self.clone(recursive=True)

    for key, deriv in new_derivs.items():
        obj.insert_deriv(key, deriv, override=True)

    return obj


def as_mask_where_nonzero(self):
    """A boolean scalar or NumPy ndarray where values are nonzero and unmasked."""

    return (self._values != 0) & self.antimask


def as_mask_where_zero(self):
    """A boolean scalar or NumPy ndarray where values are zero and unmasked."""

    return (self._values == 0) & self.antimask


def as_mask_where_nonzero_or_masked(self):
    """A boolean scalar or NumPy ndarray where values are nonzero or masked."""

    return (self._values != 0) | self._mask


def as_mask_where_zero_or_masked(self):
    """A boolean scalar or NumPy ndarray where values are zero or masked."""

    return (self._values == 0) | self._mask

##########################################################################################
