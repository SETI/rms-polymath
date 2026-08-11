#########################################################################################
# polymath/extensions/mask_ops.py: masking operations
#########################################################################################

import numbers

import numpy as np

from polymath.qube import Qube

__all__ = ['clip', 'is_above', 'is_below', 'is_inside', 'is_outside', 'mask_where',
           'mask_where_between', 'mask_where_eq', 'mask_where_ge', 'mask_where_gt',
           'mask_where_le', 'mask_where_lt', 'mask_where_ne', 'mask_where_outside']


def mask_where(self, mask, replace=None, *, remask=True, recursive=True):
    """Return a copy of this object after a mask has been applied.

    If the mask is empty, this object is returned unchanged.

    Parameters:
        mask (array-like): The mask to apply as a boolean array.
        replace (object, optional): A single replacement value or an object of the same
            shape and class as this object, containing replacement values. These are
            inserted into returned object at every masked location. Use None to leave
            values unchanged.
        remask (bool, optional): True to leave the new values masked; False to replace
            the values but leave them unmasked.
        recursive (bool, optional): True to include and mask the derivatives as well;
            False to exclude derivatives from the returned object.

    Returns:
        Qube: A copy of this object with the mask applied.

    Raises:
        ValueError: If the replacement shape is incompatible with the object shape.
    """

    if replace is None and not remask:      # nothing to do
        return self

    # Convert to boolean array if necessary
    mask = Qube._suitable_mask(mask, self._shape)

    # If the mask is empty, return the object as is
    if not np.any(mask):
        return self

    # Get the replacement value as this type
    if replace is not None:

        # A number replacing the items of an object whose items are single elements needs
        # only to be cast to this object's data type, which is all that constructing an
        # object for it would accomplish, at a small fraction of the cost
        if (self._is_array and self._rank == 0 and isinstance(mask, np.ndarray)
                and isinstance(replace, numbers.Real)):
            values = Qube._casted_to_dtype(replace, Qube._dtype(self._values))
            return _replace_where(self, values, mask, remask=remask,
                                  recursive=recursive)

        replace = self.as_this_type(replace, recursive=True)
        if replace._shape not in ((), self._shape):
            raise ValueError(f'{type(self).__name__}.mask_where() replacement has '
                             f'incompatible shape: {replace._shape}, {self._shape}')

    # Shapeless case
    if self._is_scalar:
        if replace is None:
            obj = self.copy(recursive=recursive)
        else:
            obj = replace.copy(recursive=recursive)

        if remask:
            obj = obj.remask(True, recursive=recursive)

        return obj

    # Case with no replacement
    if replace is None:
        # Note that the new mask must be a copy
        obj = self.remask_or(mask, recursive=recursive)
        return obj

    # A replacement that is a single unmasked item without derivatives of its own can be
    # written into the value arrays directly. This is the common case, and it avoids the
    # copy and the __setitem__ machinery below, which cost the same whether one item is
    # replaced or all of them.
    if (isinstance(mask, np.ndarray) and not replace._shape and not replace._derivs
            and Qube.is_one_false(replace._mask)):
        return _replace_where(self, replace._values, mask, remask=remask,
                              recursive=recursive)

    # If replacement is an array or single Qube...

    # We need a mask to apply to the given replacement value.
    # If the replacement value has shape, use the existing mask; otherwise,
    # use True, which will allow the replacement to broadcast as needed.
    rep_mask = mask if replace._shape else True

    obj = self.copy(recursive=recursive)
    obj[mask] = replace[rep_mask]   # handles derivatives too!

    if remask:
        obj = obj.remask_or(mask, recursive=recursive)

    return obj


def _replace_where(self, replace_values, mask, *, remask, recursive):
    """A copy of this object with one value substituted wherever a mask is True.

    This is the fast path of mask_where() for a replacement value that is a single item,
    unmasked, and without derivatives. The result matches the general path: the replaced
    items take the new value, their derivatives are set to zero, and their mask is set if
    `remask` is True and cleared otherwise.

    Parameters:
        self (Qube): The object to copy.
        replace_values (numpy.ndarray, float, int, or bool): The values of one item to
            substitute, already cast to this object's data type.
        mask (numpy.ndarray): Boolean mask of the items to replace, already validated
            against the shape of this object and known to contain at least one True.
        remask (bool): True to mask the replaced items; False to unmask them.
        recursive (bool): True to carry the derivatives into the returned object.

    Returns:
        Qube: A new read-writable object with the replacements applied.
    """

    values = self._values.copy()
    values[mask] = replace_values

    obj = type(self)._new_from_parts(values, _replaced_mask(self._mask, mask, remask),
                                     nrank=self._nrank, drank=self._drank,
                                     unit=self._unit, example=self)

    if recursive and self._derivs:
        new_derivs = {}
        for key, deriv in self._derivs.items():
            deriv_values = deriv._values.copy()
            deriv_values[mask] = 0
            new_derivs[key] = type(deriv)._new_from_parts(
                deriv_values,
                _replaced_mask(deriv._mask, mask, remask),
                nrank=deriv._nrank, drank=deriv._drank,
                unit=deriv._unit, example=deriv)

        obj.insert_derivs(new_derivs)

    return obj


def _replaced_mask(old_mask, mask, remask):
    """The mask of an object after unmasked values have been substituted into it.

    Parameters:
        old_mask (numpy.ndarray or bool): The mask before the substitution.
        mask (numpy.ndarray): Boolean mask of the items that were replaced.
        remask (bool): True to mask the replaced items; False to unmask them.

    Returns:
        (numpy.ndarray or bool): The mask after the substitution.
    """

    if remask:
        return Qube.or_(old_mask, mask)

    if Qube.is_one_false(old_mask):
        return False

    return old_mask & np.logical_not(mask)


def mask_where_eq(self, match, replace=None, *, remask=True):
    """Return a copy of this object with items equal to a value masked.

    Instead of or in addition to masking the items, the values can be
    replaced. If no items need to be masked, this object is returned
    unchanged.

    Parameters:
        match (object): The item value to match.
        replace (object, optional): A single replacement value or an object of the same
            shape and class as this object, containing replacement values. These are
            inserted into returned object at every masked location. Use None to leave
            values unchanged.
        remask (bool, optional): True to leave the new values masked; False to replace
            the values but leave them unmasked.

    Returns:
        Qube: A copy of this object with matching items masked.
    """

    return self.mask_where(_mask_where_match(self, match, np.equal),
                           replace=replace, remask=remask)


def mask_where_ne(self, match, replace=None, *, remask=True):
    """Return a copy of this object with items not equal to a value masked.

    Instead of or in addition to masking the items, the values can be replaced.
    If no items need to be masked, this object is returned unchanged.

    Parameters:
        match (object): The item value to match.
        replace (object, optional): A single replacement value or an object of the same
            shape and class as this object, containing replacement values. These are
            inserted into returned object at every masked location. Use None to leave
            values unchanged.
        remask (bool, optional): True to leave the new values masked; False to replace
            the values but leave them unmasked.

    Returns:
        Qube: A copy of this object with non-matching items masked.
    """

    return self.mask_where(_mask_where_match(self, match, np.not_equal),
                           replace=replace, remask=remask)


def _mask_where_match(self, match, comparison):
    """The mask of the items of this object that satisfy a comparison against a value.

    An item qualifies only if every element of the item satisfies the comparison.

    Parameters:
        self (Qube): The object whose items are to be compared.
        match (Qube, array-like, float, int, or bool): The item value to match. A value
            that is not already an object of this class is converted to one, which
            coerces it to this object's data type.
        comparison (function): The NumPy comparison to apply, one of numpy.equal or
            numpy.not_equal.

    Returns:
        (numpy.ndarray or bool): True for each item that satisfies the comparison.
    """

    # An object whose items are single elements can compare directly against a number
    # once that number has been coerced to its data type, which is all that constructing
    # an object for the number would accomplish, at a small fraction of the cost
    if self._rank == 0:
        if isinstance(match, numbers.Real):
            match_values = Qube._casted_to_dtype(match, Qube._dtype(self._values))
        else:
            match_values = self.as_this_type(match, recursive=False)._values

        return comparison(self._values, match_values)

    match = self.as_this_type(match, recursive=False)
    axes = tuple(range(-self._rank, 0))

    return np.all(comparison(self._values, match._values), axis=axes)


def mask_where_le(self, limit, replace=None, *, remask=True):
    """Return a copy of this object with items <= a limit value masked.

    Instead of or in addition to masking the items, the values can be replaced.
    If no items need to be masked, this object is returned unchanged.

    Parameters:
        limit (object): The limiting value.
        replace (object, optional): A single replacement value or an object of the same
            shape and class as this object, containing replacement values. These are
            inserted into returned object at every masked location. Use None to leave
            values unchanged.
        remask (bool, optional): True to leave the new values masked; False to replace
            the values but leave them unmasked.

    Returns:
        Qube: A copy of this object with items <= limit masked.

    Raises:
        ValueError: If this object has denominators or item rank > 0.
    """

    self._disallow_denom('mask_where_le()')
    self._require_scalar('mask_where_le()')
    limit = _limit_from_qube(self, limit, masked=-np.inf, op='mask_where_le()')
    return self.mask_where(self._values <= limit, replace=replace, remask=remask)


def mask_where_ge(self, limit, replace=None, *, remask=True):
    """Return a copy of this object with items >= a limit value masked.

    Instead of or in addition to masking the items, the values can be replaced.
    If no items need to be masked, this object is returned unchanged.

    Parameters:
        limit (object): The limiting value.
        replace (object, optional): A single replacement value or an object of the same
            shape and class as this object, containing replacement values. These are
            inserted into returned object at every masked location. Use None to leave
            values unchanged.
        remask (bool, optional): True to leave the new values masked; False to replace
            the values but leave them unmasked.

    Returns:
        Qube: A copy of this object with items >= limit masked.

    Raises:
        ValueError: If this object has denominators or item rank > 0.
    """

    self._disallow_denom('mask_where_ge()')
    self._require_scalar('mask_where_ge()')
    limit = _limit_from_qube(self, limit, masked=np.inf, op='mask_where_ge()')
    return self.mask_where(self._values >= limit, replace=replace, remask=remask)


def mask_where_lt(self, limit, replace=None, *, remask=True):
    """Return a copy with items less than a limit value masked.

    Instead of or in addition to masking the items, the values can be
    replaced. If no items need to be masked, this object is returned
    unchanged.

    Parameters:
        limit (object): The limiting value.
        replace (object, optional): A single replacement value or an object of the same
            shape and class as this object, containing replacement values. These are
            inserted into returned object at every masked location. Use None to leave
            values unchanged.
        remask (bool, optional): True to leave the new values masked; False to replace
            the values but leave them unmasked.

    Returns:
        Qube: A copy of this object with items < limit masked.

    Raises:
        ValueError: If this object has denominators or item rank > 0.
    """

    self._disallow_denom('mask_where_lt()')
    self._require_scalar('mask_where_lt()')
    limit = _limit_from_qube(self, limit, masked=-np.inf, op='mask_where_lt()')
    return self.mask_where(self._values < limit, replace=replace, remask=remask)


def mask_where_gt(self, limit, replace=None, *, remask=True):
    """Return a copy with items greater than a limit value masked.

    Instead of or in addition to masking the items, the values can be replaced.
    If no items need to be masked, this object is returned unchanged.

    Parameters:
        limit (object): The limiting value.
        replace (object, optional): A single replacement value or an object of the same
            shape and class as this object, containing replacement values. These are
            inserted into returned object at every masked location. Use None to leave
            values unchanged.
        remask (bool, optional): True to leave the new values masked; False to replace
            the values but leave them unmasked.

    Returns:
        Qube: A copy of this object with items > limit masked.

    Raises:
        ValueError: If this object has denominators or item rank > 0.
    """

    self._disallow_denom('mask_where_gt()')
    self._require_scalar('mask_where_gt()')
    limit = _limit_from_qube(self, limit, masked=np.inf, op='mask_where_gt()')
    return self.mask_where(self._values > limit, replace=replace, remask=remask)


def mask_where_between(self, lower, upper, *, mask_endpoints=False, replace=None,
                       remask=True):
    """Return a copy with values between two limits masked.

    Instead of or in addition to masking the items, the values can be replaced. If no
    items need to be masked, this object is returned unchanged.

    Parameters:
        lower (Qube, array-like, float, or int): The lower limit as a number or an object
            that can be broadcasted to the shape of this object's values (including its
            item shape). Masked limits are ignored.
        upper (Qube, array-like, float, or int): The upper limit as a number or an object
            that can be broadcasted to the shape of this object's values (including its
            item shape). Masked limits are ignored.
        mask_endpoints (bool or tuple, optional): True to mask the endpoints, where values
            are equal to the lower or upper limits; False to exclude the endpoints. Use a
            tuple of two values to handle the endpoints differently.
        replace (Qube, array-like, float, or int, optional): A single replacement value or
            an object that can be broadcasted to the shape of this object's values
            (including its item shape). Masked replacements become masked. Use None to
            leave values unchanged.
        remask (bool, optional): True to leave the new values masked; False to replace
            the values but leave them unmasked.

    Returns:
        Qube: A copy with values between the specified limits masked.

    Raises:
        ValueError: If this object has denominators or item rank > 0.
    """

    self._disallow_denom('mask_where_between()')
    self._require_scalar('mask_where_between()')
    lower = _limit_from_qube(self, lower, masked=+np.inf, op='mask_where_between()')
    upper = _limit_from_qube(self, upper, masked=-np.inf, op='mask_where_between()')

    # To minimize the number of array operations, identify the options first
    if not isinstance(mask_endpoints, (tuple, list)):
        mask_endpoints = (mask_endpoints, mask_endpoints)

    if mask_endpoints[0]:               # lower point included in the mask
        op0 = self._values.__ge__
    else:                               # lower point excluded from the mask
        op0 = self._values.__gt__

    if mask_endpoints[1]:               # upper point included in the mask
        op1 = self._values.__le__
    else:                               # upper point excluded from the mask
        op1 = self._values.__lt__

    mask = op0(lower) & op1(upper)

    return self.mask_where(mask, replace=replace, remask=remask)


def mask_where_outside(self, lower, upper, *, mask_endpoints=False, replace=None,
                       remask=True):
    """Return a copy with values outside two limits masked.

    Instead of or in addition to masking the items, the values can be replaced. If no
    items need to be masked, this object is returned unchanged.

    Parameters:
        lower (Qube, array-like, float, or int): The lower limit as a number or an object
            that can be broadcasted to the shape of this object's values (including its
            item shape). Masked limits are ignored.
        upper (Qube, array-like, float, or int): The upper limit as a number or an object
            that can be broadcasted to the shape of this object's values (including its
            item shape). Masked limits are ignored.
        mask_endpoints (bool or tuple, optional): True to mask the endpoints, where values
            are equal to the lower or upper limits; False to exclude the endpoints. Use a
            tuple of two values to handle the endpoints differently.
        replace (Qube, array-like, float, or int, optional): A single replacement value or
            an object that can be broadcasted to the shape of this object's values
            (including its item shape). Masked replacements become masked. Use None to
            leave values unchanged.
        remask (bool, optional): True to leave the new values masked; False to replace
            the values but leave them unmasked.

    Returns:
        Qube: A copy with values outside the specified limits masked.

    Raises:
        ValueError: If this object has denominators or item rank > 0.
    """

    self._disallow_denom('mask_where_outside()')
    self._require_scalar('mask_where_outside()')
    lower = _limit_from_qube(self, lower, masked=-np.inf, op='mask_where_outside()')
    upper = _limit_from_qube(self, upper, masked=+np.inf, op='mask_where_outside()')

    # To minimize the number of array operations, identify the options first
    if not isinstance(mask_endpoints, (tuple, list)):
        mask_endpoints = (mask_endpoints, mask_endpoints)

    if mask_endpoints[0]:               # end points are included in the mask
        op0 = self._values.__le__
    else:                               # end points are excluded from the mask
        op0 = self._values.__lt__

    if mask_endpoints[1]:               # end points are included in the mask
        op1 = self._values.__ge__
    else:                               # end points are excluded from the mask
        op1 = self._values.__gt__

    mask = op0(lower) | op1(upper)

    return self.mask_where(mask, replace=replace, remask=remask)


def clip(self, lower, upper, *, remask=True, inclusive=True):
    """Return a copy with values clipped to fall within a pair of limits.

    Values below the lower limit become equal to the lower limit; values above
    the upper limit become equal to the upper limit.

    Parameters:
        lower (object, optional): The numerical lower limit or an object of the same shape
            and type as this, containing lower limits. None or masked values to ignore.
        upper (object, optional): The numerical upper limit or an object of the same shape
            and type as this, containing upper limits. None or masked values to ignore.
        remask (bool, optional): True to leave the new values masked; False to replace
            the values but leave them unmasked.
        inclusive (bool, optional): True to leave values that exactly match the upper
            limit unmasked; False to mask them.

    Returns:
        Qube: A copy with values clipped to the specified limits.

    Raises:
        ValueError: If this object has denominators or item rank > 0.
    """

    self._disallow_denom('clip()')
    self._require_scalar('clip()')

    # Easy case...
    if np.isscalar(lower) and np.isscalar(upper):
        new_values = np.clip(self._values, lower, upper)
        if remask:
            outside = Qube.is_outside(self._values, lower, upper, inclusive)
            mask = Qube.or_(self._mask, outside)
        else:
            mask = self._mask

        # Without remasking, derivatives out of range are now all zero
        if self._derivs and not remask:
            new_derivs = {}
            outside = Qube.is_outside(self._values, lower, upper, inclusive)
            for key, deriv in self._derivs.items():
                new_deriv = deriv.copy()
                new_deriv[outside] = deriv.zero()
                new_derivs[key] = new_deriv
        else:
            new_derivs = self._derivs

        result = Qube.__new__(type(self))
        result.__init__(new_values, mask, derivs=new_derivs, example=self)
        return result

    result = self

    if lower is not None:
        result = result.mask_where(result._values < lower, replace=lower, remask=remask)

    if upper is not None:
        if inclusive:
            result = result.mask_where(result._values > upper, replace=upper,
                                       remask=remask)
        else:
            result = result.mask_where(result._values >= upper, replace=upper,
                                       remask=remask)

    return result


def _limit_from_qube(self, limit, masked, op):
    """Interpret the limit.

    Parameters:
        self (Qube): The object for which the limit is to be applied.
        limit (Qube, array-like, float, or int): Limit value. If it is array-like or a
            Qube, it must be broadcastable to self.shape. Also, if it is a Qube, the shape
            of its numerator must be either () or self.numer.
        masked (Qube, array-like, float or int): The value(s) to use where `limit` is
            masked if it is a Qube. If it is array-like or a Qube, it must be
            broadcastable to self.shape. Also, if it is a Qube, the shape of its numerator
            must be either () or self.numer.
        op (str, optional): Operation name to appear in an error message.

    Returns:
        (numpy.ndarray, float, or int): The value of `limit` as an array or scalar. If it
            is an array, its shape will match that of the array `self.values`.
    """

    if isinstance(limit, np.ndarray):
        if self._rank:  # pragma: no cover
            # Limits apply to items overall, not to individual components
            # Note: For Scalars, _rank is always 0, so this line cannot be reached with
            # Scalars
            # This line would require a Qube type with _rank > 0, but such types don't
            # have mask_where_le/clip because they require scalar items
            limit = np.reshape(limit, self._rank * (1,))
        return limit

    if not isinstance(limit, Qube):
        return limit

    if not isinstance(limit._mask, np.ndarray) and limit._mask:     # if fully masked
        return masked

    if limit._drank:
        raise ValueError(self._opstr(op) + ' limit does not support denominators')
    tail = self._drank * (1,)

    if limit._numer:
        if limit._numer != self._numer:
            raise ValueError(self._opstr(op) + ' limit item does not match object: '
                             f'{limit._numer}, {self._numer}')
        # This requires both self and limit to have non-empty _numer that match
        # Scalars have _numer = (), so we can't test this with Scalars
        # But Scalars always have _numer = (), so this line cannot be reached with Scalars
        tail = limit._numer + tail  # pragma: no cover
    elif self._numer:
        # This requires self._numer to be truthy (non-empty) and limit._numer to be falsy
        # (empty)
        # Scalars have _numer = (), so we can't test this with Scalars
        tail = self._nrank * (1,) + tail  # pragma: no cover

    vals = np.broadcast_to(limit._values, self._shape + tail)

    if not np.any(limit._mask):
        return vals

    mask = np.reshape(limit._mask, limit._mask.shape + self._rank * (1,))
    mask = np.broadcast_to(mask, vals.shape)
    vals = vals.copy()
    vals[mask] = masked
    return vals

#########################################################################################
# Convenience methods for range masks and clipping
#########################################################################################

@staticmethod
def is_below(arg, high, inclusive=True):
    """Check if arg is inside a range with upper end at high.

    Parameters:
        arg (object): The value to check.
        high (object): The upper limit of the range.
        inclusive (bool, optional): True to include the upper limit in the range; False to
            exclude it.

    Returns:
        bool: True if arg is inside the range with upper end at high.
    """

    if inclusive:
        return arg <= high
    else:
        return arg < high


@staticmethod
def is_above(arg, high, inclusive=True):
    """Check if arg is outside a range with upper end at high.

    Parameters:
        arg (object): The value to check.
        high (object): The upper limit of the range.
        inclusive (bool, optional): True to include the upper limit in the range; False to
            exclude it.

    Returns:
        bool: True if arg is outside the range with upper end at high.
    """

    if inclusive:
        return arg > high
    else:
        return arg >= high


@staticmethod
def is_outside(arg, low, high, inclusive=True):
    """Check if arg is outside the range low to high.

    Parameters:
        arg (object): The value to check.
        low (object): The lower limit of the range.
        high (object): The upper limit of the range.
        inclusive (bool, optional): True to include the upper limit in the range; False to
            exclude it.

    Returns:
        bool: True if arg is outside the range low to high.
    """

    if inclusive:
        return (arg < low) | (arg > high)
    else:
        return (arg < low) | (arg >= high)


@staticmethod
def is_inside(arg, low, high, inclusive=True):
    """Check if arg is inside the range low to high.

    Parameters:
        arg (object): The value to check.
        low (object): The lower limit of the range.
        high (object): The upper limit of the range.
        inclusive (bool, optional): True to include the upper limit in the range; False to
            exclude it.

    Returns:
        bool: True if arg is inside the range low to high.
    """

    if inclusive:
        return (arg >= low) & (arg <= high)
    else:
        return (arg >= low) & (arg < high)

#########################################################################################
