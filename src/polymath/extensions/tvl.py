################################################################################
# polymath/extensions/tvl.py: Three-valued logic operations
################################################################################

import numpy as np
from polymath.qube import Qube

__all__ = ['tvl_all', 'tvl_and', 'tvl_any', 'tvl_eq', 'tvl_ge', 'tvl_gt', 'tvl_le',
           'tvl_lt', 'tvl_ne', 'tvl_or']


def tvl_and(self, arg, builtins=None, masked=None):
    """The three-valued logic "and" operator result.

    Masked values are treated as indeterminate rather than being ignored. These are the
    rules:

        * False (unmasked) and anything = False (even if the other operand is masked)
        * True and True = True
        * True and Masked = Masked
        * Masked and Masked = Masked

    Parameters:
        arg (Qube or bool): The right-hand operand for the AND operation.
        builtins (bool, optional): If True and the result is a single unmasked scalar, the
            result is returned as a Python boolean instead of as an instance of Boolean.
            Default is to use the global setting defined by Qube.prefer_builtins().
        masked (bool, optional): The value to return if builtins is True but the returned
            value is masked. Default is to return a masked value instead of a builtin
            type.

    Returns:
        (Boolean or bool): The result of the three-valued logic "and" operation. When the
        result is masked, the underlying boolean value may be either True or False, and
        the mask indicates indeterminacy.
    """

    # Truth table...
    #           False       Masked      True
    # False     False       False       False
    # Masked    False       Masked      Masked
    # True      False       Masked      True

    self = Qube._BOOLEAN_CLASS.as_boolean(self)
    arg = Qube._BOOLEAN_CLASS.as_boolean(arg)

    # Determine if each operand is False (unmasked)
    # False (unmasked) and anything = False, so we need to detect unmasked False values
    if Qube.is_one_false(self._mask):
        self_is_false_unmasked = np.logical_not(self._values)
    else:
        # False if value is False and unmasked
        self_is_false_unmasked = np.logical_not(self._values) & self.antimask

    if Qube.is_one_false(arg._mask):
        arg_is_false_unmasked = np.logical_not(arg._values)
    else:
        # False if value is False and unmasked
        arg_is_false_unmasked = np.logical_not(arg._values) & arg.antimask

    # If either operand is False (unmasked), result is False and unmasked
    result_is_false = self_is_false_unmasked | arg_is_false_unmasked

    # For non-False cases, determine truth values
    # True and Masked = Masked, so we need to check if either is masked
    if Qube.is_one_false(self._mask):
        self_is_true_unmasked = self._values
        self_is_masked = False
    else:
        self_is_true_unmasked = self._values & self.antimask
        # Masked if mask is True and value is not False (unmasked)
        self_is_masked = self._mask & np.logical_not(self_is_false_unmasked)

    if Qube.is_one_false(arg._mask):
        arg_is_true_unmasked = arg._values
        arg_is_masked = False
    else:
        arg_is_true_unmasked = arg._values & arg.antimask
        # Masked if mask is True and value is not False (unmasked)
        arg_is_masked = arg._mask & np.logical_not(arg_is_false_unmasked)

    # Result is True only if both are True and unmasked
    result_is_true = self_is_true_unmasked & arg_is_true_unmasked

    # Result is masked if:
    # - Both are masked (and neither is False unmasked), OR
    # - One is True (unmasked) and the other is masked (True and Masked = Masked)
    result_is_masked = ((self_is_masked & arg_is_masked) |
                        (self_is_true_unmasked & arg_is_masked) |
                        (self_is_masked & arg_is_true_unmasked))

    # Override: if result is False, it's never masked
    result_is_masked = result_is_masked & np.logical_not(result_is_false)

    result = Qube._BOOLEAN_CLASS(result_is_true, result_is_masked)

    # Convert result to a Python bool if necessary
    if builtins is None:
        builtins = Qube.prefer_builtins()

    if builtins:
        return result.as_builtin(masked=masked)

    return result


def tvl_or(self, arg, builtins=None, masked=None):
    """The three-valued logic "or" operator result.

    Masked values are treated as indeterminate rather than being ignored. These are the
    rules:

        * True (unmasked) or anything = True (even if the other operand is masked)
        * False or False = False
        * False or Masked = Masked
        * Masked or Masked = Masked

    Parameters:
        arg (Qube or bool): The right-hand operand for the OR operation.
        builtins (bool, optional): If True and the result is a single unmasked scalar, the
            result is returned as a Python boolean instead of as an instance of Boolean.
            Default is to use the global setting defined by Qube.prefer_builtins().
        masked (bool, optional): The value to return if builtins is True but the returned
            value is masked. Default is to return a masked value instead of a builtin
            type.

    Returns:
        (Boolean or bool): The result of the three-valued logic "or" operation. When the
        result is masked, the underlying boolean value may be either True or False, and
        the mask indicates indeterminacy.
    """

    # Truth table...
    #           False       Masked      True
    # False     False       Masked      True
    # Masked    Masked      Masked      True
    # True      True        True        True

    self = Qube._BOOLEAN_CLASS.as_boolean(self)
    arg = Qube._BOOLEAN_CLASS.as_boolean(arg)

    # Determine if each operand is True (unmasked)
    # True (unmasked) or anything = True, so we need to detect unmasked True values
    if Qube.is_one_false(self._mask):
        self_is_true_unmasked = self._values
    else:
        # True if value is True and unmasked
        self_is_true_unmasked = self._values & self.antimask

    if Qube.is_one_false(arg._mask):
        arg_is_true_unmasked = arg._values
    else:
        # True if value is True and unmasked
        arg_is_true_unmasked = arg._values & arg.antimask

    # If either operand is True (unmasked), result is True and unmasked
    result_is_true = self_is_true_unmasked | arg_is_true_unmasked

    # For non-True cases, determine false/masked values
    # False or Masked = Masked, so we need to check if either is masked
    if Qube.is_one_false(self._mask):
        self_is_false_unmasked = np.logical_not(self._values)
        self_is_masked = False
    else:
        self_is_false_unmasked = np.logical_not(self._values) & self.antimask
        # Masked if mask is True and value is not True (unmasked)
        self_is_masked = self._mask & np.logical_not(self_is_true_unmasked)

    if Qube.is_one_false(arg._mask):
        arg_is_false_unmasked = np.logical_not(arg._values)
        arg_is_masked = False
    else:
        arg_is_false_unmasked = np.logical_not(arg._values) & arg.antimask
        # Masked if mask is True and value is not True (unmasked)
        arg_is_masked = arg._mask & np.logical_not(arg_is_true_unmasked)

    # Result is masked if:
    # - Both are masked (and neither is True unmasked), OR
    # - One is False (unmasked) and the other is masked (False or Masked = Masked)
    result_is_masked = ((self_is_masked & arg_is_masked) |
                        (self_is_false_unmasked & arg_is_masked) |
                        (self_is_masked & arg_is_false_unmasked))

    # Override: if result is True, it's never masked
    result_is_masked = result_is_masked & np.logical_not(result_is_true)

    result = Qube._BOOLEAN_CLASS(result_is_true, result_is_masked)

    # Convert result to a Python bool if necessary
    if builtins is None:
        builtins = Qube.prefer_builtins()

    if builtins:
        return result.as_builtin(masked=masked)

    return result


def tvl_any(self, axis=None, builtins=None, masked=None):
    """True if any unmasked value is True using three-valued logic.

    Masked values are treated as indeterminate rather than being ignored. These are the
    rules:

        * True if any unmasked value is True;
        * False if and only if all the items are False and unmasked;
        * otherwise, Masked.

    Parameters:
        axis (int or tuple, optional): An integer axis or a tuple of axes. The
            any operation is performed across these axes, leaving any remaining
            axes in the returned value. If None (the default), then the any
            operation is performed across all axes of the object, reducing to a
            scalar result. When axis is specified, the result shape is the original
            shape with the specified axes removed.
        builtins (bool, optional): If True and the result is a single unmasked scalar, the
            result is returned as a Python boolean instead of as an instance of Boolean.
            Default is to use the global setting defined by Qube.prefer_builtins().
        masked (bool, optional): The value to return if builtins is True but the returned
            value is masked. Default is to return a masked value instead of a builtin
            type.

    Returns:
        (Boolean or bool): The result of the three-valued logic "any" operation. The
        result is masked if any values along the specified axes are masked, unless
        an unmasked True value is found.

    Examples:
        >>> a = Boolean([True, False, True])
        >>> a.tvl_any()  # Returns True
        >>> a = Boolean([False, False, False], mask=[False, True, False])
        >>> a.tvl_any()  # Returns Masked (indeterminate due to masked False)
    """

    self = Qube._BOOLEAN_CLASS.as_boolean(self)

    # Construct the input args to Boolean()
    if self._is_scalar:
        args = (self,)
    elif isinstance(self._mask, (bool, np.bool_)):
        args = (np.any(self._values, axis=axis), self._mask)
    else:
        # True where any value is True AND its antimask is True
        new_values = np.any(self._values & self.antimask, axis=axis)

        # Masked if any value is masked unless new_values is True
        masked_found = np.any(self._mask, axis=axis)
        new_mask = np.logical_not(new_values) & masked_found

        args = (new_values, new_mask)

    result = Qube._BOOLEAN_CLASS(*args)

    # Convert result to a Python bool if necessary
    if builtins is None:
        builtins = Qube.prefer_builtins()

    if builtins:
        return result.as_builtin(masked=masked)

    return result


def tvl_all(self, axis=None, builtins=None, masked=None):
    """True if all unmasked values are True using three-valued logic.

    Masked values are treated as indeterminate rather than being ignored. These are the
    rules:

        * True if and only if all the items are True and unmasked.
        * False if any unmasked value is False.
        * otherwise, Masked.

    Parameters:
        axis (int or tuple, optional): An integer axis or a tuple of axes. The
            all operation is performed across these axes, leaving any remaining
            axes in the returned value. If None (the default), then the all
            operation is performed across all axes of the object, reducing to a
            scalar result. When axis is specified, the result shape is the original
            shape with the specified axes removed.
        builtins (bool, optional): If True and the result is a single unmasked scalar, the
            result is returned as a Python boolean instead of as an instance of Boolean.
            Default is to use the global setting defined by Qube.prefer_builtins().
        masked (bool, optional): The value to return if builtins is True but the returned
            value is masked. Default is to return a masked value instead of a builtin
            type.

    Returns:
        (Boolean or bool): The result of the three-valued logic "all" operation. The
        result is masked if any values along the specified axes are masked, unless
        an unmasked False value is found.

    Examples:
        >>> a = Boolean([True, True, True])
        >>> a.tvl_all()  # Returns True
        >>> a = Boolean([True, True, True], mask=[False, True, False])
        >>> a.tvl_all()  # Returns Masked (indeterminate due to masked True)
    """

    self = Qube._BOOLEAN_CLASS.as_boolean(self)

    # Construct the input args to Boolean()
    if self._is_scalar:
        args = (self,)
    elif isinstance(self._mask, (bool, np.bool_)):
        args = (np.all(self._values, axis=axis), self._mask)
    else:
        # False where any value is False AND its antimask is True
        # Therefore, True where every value is True OR its mask is True
        new_values = np.all(self._values | self._mask, axis=axis)

        # Masked where any value is masked unless new_values is False
        mask_found = np.any(self._mask, axis=axis)
        new_mask = new_values & mask_found

        args = (new_values, new_mask)

    result = Qube._BOOLEAN_CLASS(*args)

    # Convert result to a Python bool if necessary
    if builtins is None:
        builtins = Qube.prefer_builtins()

    if builtins:
        return result.as_builtin(masked=masked)

    return result


def tvl_eq(self, arg, builtins=None):
    """The three-valued logic "equals" operator result.

    Masked values are treated as indeterminate, so if either value is masked, the returned
    value is masked.

    Parameters:
        arg (Qube or bool): The right-hand operand for the equality comparison.
        builtins (bool, optional): If True and the result is a single unmasked scalar, the
            result is returned as a Python boolean instead of as an instance of Boolean.
            Default is to use the global setting defined by Qube.prefer_builtins().

    Returns:
        (Boolean or bool): The result of the three-valued logic equality comparison.
        When the result is masked, the underlying boolean value may be either True or
        False, and the mask indicates indeterminacy. The `builtins` parameter affects
        the return type but not the masking behavior.
    """

    return self._tvl_op(arg, (self == arg), builtins=builtins)


def tvl_ne(self, arg, builtins=None):
    """The three-valued logic "not equal" operator result.

    Masked values are treated as indeterminate, so if either value is masked, the returned
    value is masked.

    Parameters:
        arg (Qube or bool): The right-hand operand for the inequality comparison.
        builtins (bool, optional): If True and the result is a single unmasked scalar, the
            result is returned as a Python boolean instead of as an instance of Boolean.
            Default is to use the global setting defined by Qube.prefer_builtins().

    Returns:
        (Boolean or bool): The result of the three-valued logic inequality comparison.
        When the result is masked, the underlying boolean value may be either True or
        False, and the mask indicates indeterminacy. The `builtins` parameter affects
        the return type but not the masking behavior.
    """

    return self._tvl_op(arg, (self != arg), builtins=builtins)


def tvl_lt(self, arg, builtins=None):
    """The three-valued logic "less than" operator result.

    Masked values are treated as indeterminate, so if either value is masked, the returned
    value is masked.

    Parameters:
        arg (Qube or number): The right-hand operand for the comparison.
        builtins (bool, optional): If True and the result is a single unmasked scalar, the
            result is returned as a Python boolean instead of as an instance of Boolean.
            Default is to use the global setting defined by Qube.prefer_builtins().

    Returns:
        (Boolean or bool): The result of the three-valued logic "less than" comparison.
        When the result is masked, the underlying boolean value may be either True or
        False, and the mask indicates indeterminacy. The `builtins` parameter affects
        the return type but not the masking behavior.
    """

    return self._tvl_op(arg, (self < arg), builtins=builtins)


def tvl_gt(self, arg, builtins=None):
    """The three-valued logic "greater than" operator result.

    Masked values are treated as indeterminate, so if either value is masked, the returned
    value is masked.

    Parameters:
        arg (Qube or number): The right-hand operand for the comparison.
        builtins (bool, optional): If True and the result is a single unmasked scalar, the
            result is returned as a Python boolean instead of as an instance of Boolean.
            Default is to use the global setting defined by Qube.prefer_builtins().

    Returns:
        (Boolean or bool): The result of the three-valued logic "greater than"
        comparison. When the result is masked, the underlying boolean value may be
        either True or False, and the mask indicates indeterminacy. The `builtins`
        parameter affects the return type but not the masking behavior.
    """

    return self._tvl_op(arg, (self > arg), builtins=builtins)


def tvl_le(self, arg, builtins=None):
    """The three-valued logic "less than or equal to" operator result.

    Masked values are treated as indeterminate, so if either value is masked, the returned
    value is masked.

    Parameters:
        arg (Qube or number): The right-hand operand for the comparison.
        builtins (bool, optional): If True and the result is a single unmasked scalar, the
            result is returned as a Python boolean instead of as an instance of Boolean.
            Default is to use the global setting defined by Qube.prefer_builtins().

    Returns:
        (Boolean or bool): The result of the three-valued logic "less than or equal
        to" comparison. When the result is masked, the underlying boolean value may be
        either True or False, and the mask indicates indeterminacy. The `builtins`
        parameter affects the return type but not the masking behavior.
    """

    return self._tvl_op(arg, (self <= arg), builtins=builtins)


def tvl_ge(self, arg, builtins=None):
    """The three-valued logic "greater than or equal to" operator result.

    Masked values are treated as indeterminate, so if either value is masked, the returned
    value is masked.

    Parameters:
        arg (Qube or number): The right-hand operand for the comparison.
        builtins (bool, optional): If True and the result is a single unmasked scalar, the
            result is returned as a Python boolean instead of as an instance of Boolean.
            Default is to use the global setting defined by Qube.prefer_builtins().

    Returns:
        (Boolean or bool): The result of the three-valued logic "greater than or
        equal to" comparison. When the result is masked, the underlying boolean value
        may be either True or False, and the mask indicates indeterminacy. The
        `builtins` parameter affects the return type but not the masking behavior.
    """

    return self._tvl_op(arg, (self >= arg), builtins=builtins)


def _tvl_op(self, arg, comparison, builtins=None):
    """The three-valued logic version of any boolean operator.

    Masked values are treated as indeterminate, so if either value is masked, the returned
    value is masked.

    Parameters:
        arg (Qube or number): The right-hand operand for the operation.
        comparison (Qube or bool): The result of the boolean comparison.
        builtins (bool, optional): If True and the result is a single unmasked scalar, the
            result is returned as a Python boolean instead of as an instance of Boolean.
            Default is to use the global setting defined by Qube.prefer_builtins().

    Returns:
        (Boolean or bool): The result of the three-valued logic operation.
    """

    # Return a Python bool if appropriate
    if isinstance(comparison, bool):
        if builtins is None:
            builtins = Qube.prefer_builtins()
        if builtins:
            return comparison

        comparison = Qube._BOOLEAN_CLASS(comparison)

    # Determine arg_mask, if any
    if isinstance(arg, Qube):
        arg_mask = arg._mask
    elif isinstance(arg, np.ma.MaskedArray):
        arg_mask = arg.mask
    else:
        arg_mask = False

    comparison._set_mask(Qube.or_(self._mask, arg_mask))
    return comparison

################################################################################
