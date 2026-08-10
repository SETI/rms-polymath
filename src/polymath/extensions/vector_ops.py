##########################################################################################
# polymath/extensions/vector_ops.py: vector operations
##########################################################################################

import math
import numpy as np
import numbers
from polymath.qube import Qube
from polymath.unit import Unit

__all__ = ['as_diagonal', 'cross', 'dot', 'norm', 'norm_sq', 'outer', 'rms']


def _mean_or_sum(arg, axis=None, *, recursive=True, _combine_as_mean=False):
    """Calculate the mean or sum of the unmasked values.

    Internal method for computing mean or sum operations.

    Parameters:
        arg (Qube): The object for which to calculate the mean or sum.
        axis (int or tuple, optional): An integer axis or a tuple of axes. The mean is
            determined across these axes, leaving any remaining axes in the returned
            value. If None (the default), then the mean is performed across all axes of
            the object.
        recursive (bool, optional): True to include derivatives in the returned object.
        _combine_as_mean (bool, optional): True to combine as a mean; False to combine as
            a sum.

    Returns:
        Qube: The mean or sum of the unmasked values.
    """

    arg._check_axis(axis, 'mean()' if _combine_as_mean else 'sum()')

    if arg._size == 0:
        return arg._zero_sized_result(axis=axis)

    # Select the NumPy function
    if _combine_as_mean:
        func = np.mean
    else:
        func = np.sum

    # Create the new axis, which is valid regardless of items
    if isinstance(axis, numbers.Integral):
        new_axis = axis % arg._ndims
    elif axis is None:
        new_axis = tuple(range(arg._ndims))
    else:
        new_axis = tuple(a % arg._ndims for a in axis)

    # If there's no mask, this is easy
    if not np.any(arg._mask):
        obj = Qube(func(arg._values, axis=new_axis), False, example=arg)

    # Handle a fully masked object
    elif np.all(arg._mask):
        obj = Qube(func(arg._values, axis=new_axis), True, example=arg)

    # If we are averaging over all axes, this is fairly easy
    elif axis is None:
        if arg._shape:
            obj = Qube(func(arg._values[arg.antimask], axis=0), False, example=arg)
        else:  # pragma: no cover
            # This is unreachable because if arg._shape is (), then the mask is boolean
            # and either mask=False (line 50 hits) or mask=True (line 54 hits).
            raise RuntimeError('This should be unreachable')

    # At this point, we have handled the cases mask==True and mask==False, so the mask
    # must be an array. Also, there must be at least one unmasked value.
    else:
        # Set masked items to zero, then sum across axes. np.where() does this in one
        # pass, where a copy followed by an indexed assignment takes several.
        if arg._rank:
            mask = arg._mask.reshape(arg._shape + arg._rank * (1,))
        else:
            mask = arg._mask

        new_values = np.sum(np.where(mask, 0, arg._values), axis=new_axis)

        # Count the unmasked items, summed across axes. Counting the masked ones and
        # subtracting avoids materializing the antimask.
        if isinstance(new_axis, tuple):
            length = math.prod([arg._shape[a] for a in new_axis])
        else:
            length = arg._shape[new_axis]

        count = length - np.count_nonzero(arg._mask, axis=new_axis)

        # Convert to a mask and a mean
        new_mask = (count == 0)
        if _combine_as_mean:
            count_reshaped = count.reshape(count.shape + arg._rank * (1,))
            denom = np.maximum(count_reshaped, 1)
            new_values = new_values / denom

        # Fill in masked values with the default
        if np.any(new_mask):
            new_values[(new_mask,) +
                       arg._rank * (slice(None),)] = arg._default
        else:
            new_mask = False

        obj = Qube(new_values, new_mask, example=arg)

    # Cast to the proper class
    obj = obj.cast(type(arg))

    # Handle derivatives
    if recursive and arg._derivs:
        new_derivs = {}
        for key, deriv in arg._derivs.items():
            new_derivs[key] = _mean_or_sum(deriv, axis, recursive=False,
                                           _combine_as_mean=_combine_as_mean)

        obj.insert_derivs(new_derivs)

    return obj


def _check_axis(arg, axis, op):
    """Validate the axis as None, an int, or a tuple of ints.

    Parameters:
        arg (Qube): The object to check the axis for.
        axis: The axis to validate.
        op (str): The operation name for error messages.

    Raises:
        IndexError: If the axis is out of range or duplicated.
    """

    if axis is None:    # can't be a problem
        return

    # Fix up the axis argument
    if isinstance(axis, tuple):
        axis_for_show = axis
    elif isinstance(axis, list):
        axis_for_show = tuple(axis)
    else:
        axis_for_show = axis
        axis = (axis,)

    # Check for duplicates
    # Check for in-range values
    selections = arg._ndims * [False]
    for i in axis:
        try:
            _ = selections[i]
        except IndexError as e:
            raise IndexError(f'axis is out of range ({-arg._ndims},{arg._ndims}) in '
                             f'{type(arg)}.{op}: {i}') from e

        if selections[i]:
            raise IndexError(f'duplicated axis in {type(arg)}.{op}: {axis_for_show}')

        selections[i] = True


def _zero_sized_result(self, axis):
    """Return a zero-sized result obtained by collapsing one or more axes.

    Parameters:
        axis (int or tuple, optional): The axis or axes to collapse.

    Returns:
        Qube: A zero-sized result with the specified axes collapsed.
    """

    if axis is None:
        return self.flatten().as_size_zero()

    # Construct an index to obtain the correct shape
    indx = self._ndims * [slice(None)]
    if isinstance(axis, (list, tuple)):
        for i in axis:
            indx[i] = 0
    else:
        indx[axis] = 0

    return self[tuple(indx)]


@staticmethod
def dot(arg1, arg2, axis1=-1, axis2=0, *, classes=(), recursive=True):
    """Calculate the dot product of two objects.

    The axes must be in the numerator, and only one of the objects can have a denominator
    (which makes this suitable for first derivatives but not second derivatives).

    Note: When one object has a denominator, the dot product is computed along numerator
    axes. The denominator dimensions are preserved in the result.

    Note: This is a static method. Call it as Qube.dot(arg1, arg2, ...) rather than
    arg1.dot(arg2, ...).

    Parameters:
        arg1 (Qube): The first operand as a subclass of Qube.
        arg2 (Qube): The second operand as a subclass of Qube.
        axis1 (int, optional): The item axis of arg1 for the dot product. Default is -1.
        axis2 (int, optional): The item axis of arg2 for the dot product. Default is 0.
        classes (class, list, or tuple, optional): The class of the object returned. If
            a list is provided, the object will be an instance of the first suitable class
            in the list. Otherwise, a generic Qube object will be returned.
        recursive (bool, optional): True to include derivatives in the returned object.

    Returns:
        Qube: The dot product of the two objects.

    Raises:
        ValueError: If both objects have denominators or if axes are out of range.
    """

    # At most one object can have a denominator.
    if arg1._drank and arg2._drank:
        Qube._raise_dual_denoms('dot()', arg1, arg2)

    # Position axis1 from left
    if axis1 >= 0:
        a1 = axis1
    else:
        a1 = axis1 + arg1._nrank
    if a1 < 0 or a1 >= arg1._nrank:
        raise ValueError(f'first axis is out of range ({-arg1._nrank},{arg1._nrank}) '
                         f'in type(arg1).dot(): {axis1}')
    k1 = a1 + arg1._ndims

    # Position axis2 from item left
    if axis2 >= 0:
        a2 = axis2
    else:
        a2 = axis2 + arg2._nrank
    if a2 < 0 or a2 >= arg2._nrank:
        raise ValueError(f'second axis is out of range ({-arg2._nrank},{arg2._nrank}) '
                         f'in type(arg2).dot(): {axis2}')
    k2 = a2 + arg2._ndims

    # Confirm that the axis lengths are compatible
    if arg1._numer[a1] != arg2._numer[a2]:
        raise ValueError(f'{type(arg1)}.dot() axes have different lengths: '
                         f'{arg1._numer[a1]}, {arg2._numer[a2]}')

    # Re-shape the value arrays (shape, numer1, numer2, denom1, denom2)
    shape1 = (arg1._shape + arg1._numer + (arg2._nrank - 1) * (1,) +
              arg1._denom + arg2._drank * (1,))
    array1 = arg1._values.reshape(shape1)

    shape2 = (arg2._shape + (arg1._nrank - 1) * (1,) + arg2._numer +
              arg1._drank * (1,) + arg2._denom)
    array2 = arg2._values.reshape(shape2)
    k2 += arg1._nrank - 1

    # Roll both array axes to the right
    array1 = np.moveaxis(array1, k1, -1)
    array2 = np.moveaxis(array2, k2, -1)

    # Construct the dot product. einsum contracts the last axis without materializing the
    # elementwise product, which matters for the large arrays this is used on. It also
    # reads strided input directly, so the operands need not be made contiguous first.
    new_values = np.einsum('...i,...i->...', array1, array2)

    # Construct the object and cast
    new_nrank = arg1._nrank + arg2._nrank - 2
    new_drank = arg1._drank + arg2._drank

    obj = Qube._new_from_parts(new_values, Qube.or_(arg1._mask, arg2._mask),
                               nrank=new_nrank, drank=new_drank,
                               unit=Unit.mul_units(arg1._unit, arg2._unit),
                               example=arg1)
    obj = obj.cast(classes)

    # Insert derivatives if necessary
    if recursive and (arg1._derivs or arg2._derivs):
        new_derivs = {}

        if arg1._derivs:
            arg2_wod = arg2.wod
            for key, arg1_deriv in arg1._derivs.items():
                new_derivs[key] = Qube.dot(arg1_deriv, arg2_wod, a1, a2, classes=classes,
                                           recursive=False)

        if arg2._derivs:
            arg1_wod = arg1.wod
            for key, arg2_deriv in arg2._derivs.items():
                term = Qube.dot(arg1_wod, arg2_deriv, a1, a2, classes=classes,
                                recursive=False)
                if key in new_derivs:
                    new_derivs[key] += term
                else:
                    new_derivs[key] = term

        obj.insert_derivs(new_derivs)

    return obj


@staticmethod
def norm(arg, axis=-1, *, classes=(), recursive=True):
    """Calculate the norm of an object along one axis.

    The axes must be in the numerator. The denominator must have zero rank.

    Note: This is a static method. Call it as Qube.norm(arg, ...) rather than
    arg.norm(...).

    Parameters:
        arg (Qube): The object for which to calculate the norm.
        axis (int, optional): The numerator axis for the norm. Defaults to -1.
        classes (class, list, or tuple, optional): The class of the object returned. If
            a list is provided, the object will be an instance of the first suitable class
            in the list. Otherwise, a generic Qube object will be returned.
        recursive (bool, optional): True to include derivatives in the returned object.

    Returns:
        Qube: The norm of the object along the specified axis.

    Raises:
        ValueError: If the object has denominators or if the axis is out of
            range.

    Examples:
        For a Vector with shape (2, 3) and numer (2,):
        - axis=-1 (default) → result shape (2, 3), numer ()
        - axis=0 → result shape (2, 3), numer ()
    """

    arg._disallow_denom('norm()')

    # Position axis from left
    if axis >= 0:
        a1 = axis
    else:
        a1 = axis + arg._nrank
    if a1 < 0 or a1 >= arg._nrank:
        raise ValueError(f'axis is out of range ({-arg._nrank},{arg._nrank}) in '
                         f'{type(arg)}.norm(): {axis}')
    k1 = a1 + arg._ndims

    # Evaluate the norm
    new_values = np.sqrt(np.sum(arg._values**2, axis=k1))

    # Construct the object and cast
    obj = Qube._new_from_parts(new_values, arg._mask, nrank=arg._nrank-1,
                               drank=arg._drank, unit=arg._unit, example=arg)
    obj = obj.cast(classes)

    # Insert derivatives if necessary
    if recursive and arg._derivs:
        factor = arg.wod / obj
        for key, arg_deriv in arg._derivs.items():
            obj.insert_deriv(key, Qube.dot(factor, arg_deriv, a1, a1, classes=classes,
                                           recursive=False))

    return obj


@staticmethod
def norm_sq(arg, axis=-1, *, classes=(), recursive=True):
    """Calculate the square of the norm of an object along one axis.

    The axes must be in the numerator. The denominator must have zero rank.

    Note: This is a static method. Call it as Qube.norm_sq(arg, ...) rather than
    arg.norm_sq(...).

    Parameters:
        arg: The object for which to calculate the norm-squared.
        axis (int, optional): The item axis for the norm. Default is -1.
        classes (class, list, or tuple, optional): The class of the object returned. If
            a list is provided, the object will be an instance of the first suitable class
            in the list. Otherwise, a generic Qube object will be returned.
        recursive (bool, optional): True to include derivatives in the returned object.

    Returns:
        Qube: The square of the norm of the object along the specified axis.

    Raises:
        ValueError: If the object has denominators or if the axis is out of range.

    Examples:
        For a Vector with shape (2, 3) and numer (2,):
        - axis=-1 (default) → result shape (2, 3), numer ()
        - axis=0 → result shape (2, 3), numer ()
    """

    arg._disallow_denom('norm_sq()')

    # Position axis from left
    if axis >= 0:
        a1 = axis
    else:
        a1 = axis + arg._nrank
    if a1 < 0 or a1 >= arg._nrank:
        raise ValueError(f'axis is out of range ({-arg._nrank},{arg._nrank}) in '
                         f'{type(arg)}.norm_sq(): {axis}')
    k1 = a1 + arg._ndims

    # Evaluate the norm
    new_values = np.sum(arg._values**2, axis=k1)

    # Construct the object and cast
    obj = Qube._new_from_parts(new_values, arg._mask, nrank=arg._nrank-1,
                               drank=arg._drank,
                               unit=Unit.mul_units(arg._unit, arg._unit), example=arg)
    obj = obj.cast(classes)

    # Insert derivatives if necessary
    if recursive and arg._derivs:
        factor = 2. * arg.wod
        for key, arg_deriv in arg._derivs.items():
            obj.insert_deriv(key, Qube.dot(factor, arg_deriv, a1, a1, classes=classes,
                                           recursive=False))

    return obj


@staticmethod
def cross(arg1, arg2, axis1=-1, axis2=0, *, classes=(), recursive=True):
    """Calculate the cross product of two objects.

    Axis lengths must be either two or three, and must be equal. At least one of the
    objects must be lacking a denominator.

    Note: This is a static method. Call it as Qube.cross(arg1, arg2, ...) rather than
    arg1.cross(arg2, ...).

    Parameters:
        arg1 (Qube): The first operand.
        arg2 (Qube): The second operand.
        axis1 (int, optional): The item axis of the first object. Defaults to -1.
        axis2 (int, optional): The item axis of the second object. Defaults to 0.
        classes (class, list, or tuple, optional): The class of the object returned. If
            a list is provided, the object will be an instance of the first suitable class
            in the list. Otherwise, a generic Qube object will be returned.
        recursive (bool, optional): True to include derivatives in the returned object.

    Returns:
        Qube: The cross product of the two objects.

    Raises:
        ValueError: If both objects have denominators, if axes are out of range,
            or if axis lengths are incompatible.
    """

    # At most one object can have a denominator.
    if arg1._drank and arg2._drank:
        Qube._raise_dual_denoms('cross()', arg1, arg2)

    # Position axis1 from left
    if axis1 >= 0:
        a1 = axis1
    else:
        a1 = axis1 + arg1._nrank
    if a1 < 0 or a1 >= arg1._nrank:
        raise ValueError(f'first axis is out of range ({-arg1._nrank},{arg1._nrank}) '
                         f'in {type(arg1)}.cross(): {axis1}')
    k1 = a1 + arg1._ndims

    # Position axis2 from item left
    if axis2 >= 0:
        a2 = axis2
    else:
        a2 = axis2 + arg2._nrank
    if a2 < 0 or a2 >= arg2._nrank:
        raise ValueError(f'second axis is out of range ({-arg2._nrank},{arg2._nrank}) '
                         f'in {type(arg2)}.cross(): {axis2}')
    k2 = a2 + arg2._ndims

    # Confirm that the axis lengths are compatible
    if (arg1._numer[a1] != arg2._numer[a2]) or (arg1._numer[a1] not in {2, 3}):
        raise ValueError(f'invalid axis length for {type(arg1)}.cross(): '
                         f'{arg1._numer[a1]}, {arg2._numer[a2]}; must be 2 or 3')

    # Re-shape the value arrays (shape, numer1, numer2, denom1, denom2)
    shape1 = (arg1._shape + arg1._numer + (arg2._nrank - 1) * (1,) +
              arg1._denom + arg2._drank * (1,))
    array1 = arg1._values.reshape(shape1)

    shape2 = (arg2._shape + (arg1._nrank - 1) * (1,) + arg2._numer +
              arg1._drank * (1,) + arg2._denom)
    array2 = arg2._values.reshape(shape2)
    k2 += arg1._nrank - 1

    # Roll both array axes to the right
    array1 = np.moveaxis(array1, k1, -1)
    array2 = np.moveaxis(array2, k2, -1)

    new_drank = arg1._drank + arg2._drank

    # Construct the cross product values
    if arg1._numer[a1] == 3:
        new_values = _cross_3x3(array1, array2)

        # Roll the new axis back to its position in arg1
        new_nrank = arg1._nrank + arg2._nrank - 1
        new_k1 = new_values.ndim - new_drank - new_nrank + a1
        new_values = np.moveaxis(new_values, -1, new_k1)

    else:
        new_values = _cross_2x2(array1, array2)
        new_nrank = arg1._nrank + arg2._nrank - 2

    # Construct the object and cast
    obj = Qube._new_from_parts(new_values, Qube.or_(arg1._mask, arg2._mask),
                               nrank=new_nrank, drank=new_drank,
                               unit=Unit.mul_units(arg1._unit, arg2._unit),
                               example=arg1)
    obj = obj.cast(classes)

    # Insert derivatives if necessary
    if recursive and (arg1._derivs or arg2._derivs):
        new_derivs = {}

        if arg1._derivs:
            arg2_wod = arg2.wod
            for key, arg1_deriv in arg1._derivs.items():
                new_derivs[key] = Qube.cross(arg1_deriv, arg2_wod, a1, a2,
                                             classes=classes, recursive=False)

        if arg2._derivs:
            arg1_wod = arg1.wod
            for key, arg2_deriv in arg2._derivs.items():
                term = Qube.cross(arg1_wod, arg2_deriv, a1, a2, classes=classes,
                                  recursive=False)
                if key in new_derivs:
                    new_derivs[key] += term
                else:
                    new_derivs[key] = term

        obj.insert_derivs(new_derivs)

    return obj


def _cross_3x3(a, b):
    """Calculate the cross product of two 3-vectors.

    Internal helper function for computing cross products. The inputs are NumPy arrays
    representing 3-vectors, and the result is returned as a NumPy array.

    Parameters:
        a (ndarray): First 3-vector array.
        b (ndarray): Second 3-vector array.

    Returns:
        ndarray: The cross product of the two 3-vectors.

    Raises:
        ValueError: If the arrays are not 3-vectors.
    """

    (a, b) = np.broadcast_arrays(a, b)
    if not (a.shape[-1] == b.shape[-1] == 3):
        raise ValueError('_cross_3x3 requires 3-vectors')

    new_values = np.empty(a.shape, dtype=np.result_type(a, b))
    new_values[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    new_values[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    new_values[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    return new_values


def _cross_2x2(a, b):
    """Calculate the cross product of two 2-vectors.

    Internal helper function for computing cross products. The inputs are NumPy arrays
    representing 2-vectors, and the result is returned as a NumPy array.

    Parameters:
        a (ndarray): First 2-vector array.
        b (ndarray): Second 2-vector array.

    Returns:
        ndarray: The cross product of the two 2-vectors.

    Raises:
        ValueError: If the arrays are not 2-vectors.
    """

    (a, b) = np.broadcast_arrays(a, b)
    if not (a.shape[-1] == b.shape[-1] == 2):
        raise ValueError('_cross_2x2 requires 2-vectors')

    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@staticmethod
def outer(arg1, arg2, classes=(), recursive=True):
    """Calculate the outer product of two objects.

    The item shape of the returned object is obtained by concatenating the two
    numerators and then the two denominators, and each element is the product of
    the corresponding elements of the two objects.

    Note: This is a static method. Call it as Qube.outer(arg1, arg2, ...) rather than
    arg1.outer(arg2, ...).

    Parameters:
        arg1 (Qube): The first operand.
        arg2 (Qube): The second operand.
        classes (class, list, or tuple, optional): The class of the object returned. If
            a list is provided, the object will be an instance of the first suitable class
            in the list. Otherwise, a generic Qube object will be returned.
        recursive (bool, optional): True to include derivatives in the returned object.

    Returns:
        Qube: The outer product of the two objects.

    Raises:
        ValueError: If both objects have denominators.
    """

    # At most one object can have a denominator. This is sufficient
    # to track first derivatives
    if arg1._drank and arg2._drank:
        Qube._raise_dual_denoms('outer()', arg1, arg2)

    # Re-shape the value arrays (shape, numer1, numer2, denom1, denom2)
    shape1 = (arg1._shape + arg1._numer + arg2._nrank * (1,) +
              arg1._denom + arg2._drank * (1,))
    array1 = arg1._values.reshape(shape1)

    shape2 = (arg2._shape + arg1._nrank * (1,) + arg2._numer +
              arg1._drank * (1,) + arg2._denom)
    array2 = arg2._values.reshape(shape2)

    # Construct the outer product
    new_values = array1 * array2

    # Construct the object and cast
    new_nrank = arg1._nrank + arg2._nrank
    new_drank = arg1._drank + arg2._drank

    obj = Qube._new_from_parts(new_values, Qube.or_(arg1._mask, arg2._mask),
                               nrank=new_nrank, drank=new_drank,
                               unit=Unit.mul_units(arg1._unit, arg2._unit),
                               example=arg1)
    obj = obj.cast(classes)

    # Insert derivatives if necessary
    if recursive and (arg1._derivs or arg2._derivs):
        new_derivs = {}

        if arg1._derivs:
            arg_wod = arg2.wod
            for key, self_deriv in arg1._derivs.items():
                new_derivs[key] = Qube.outer(self_deriv, arg_wod, classes=classes,
                                             recursive=False)

        if arg2._derivs:
            self_wod = arg1.wod
            for key, arg_deriv in arg2._derivs.items():
                term = Qube.outer(self_wod, arg_deriv, classes=classes, recursive=False)
                if key in new_derivs:
                    new_derivs[key] += term
                else:
                    new_derivs[key] = term

        obj.insert_derivs(new_derivs)

    return obj


@staticmethod
def as_diagonal(arg, axis, classes=(), recursive=True):
    """Return a copy with one axis converted to a diagonal across two.

    Note: This is a static method. Call it as Qube.as_diagonal(arg, axis, ...) rather than
    arg.as_diagonal(axis, ...).

    Parameters:
        arg (Qube): The object to convert.
        axis (int): The item axis to convert to two.
        classes (class, list, or tuple, optional): The class of the object returned. If
            a list is provided, the object will be an instance of the first suitable class
            in the list. Otherwise, a generic Qube object will be returned.
        recursive (bool, optional): True to include derivatives in the returned object.

    Returns:
        Qube: A copy with the specified axis converted to a diagonal.

    Raises:
        ValueError: If the axis is out of range.
    """

    # Position axis from left
    if axis >= 0:
        a1 = axis
    else:
        a1 = axis + arg._nrank
    if a1 < 0 or a1 >= arg._nrank:
        raise ValueError(f'axis is out of range ({-arg._nrank},{arg._nrank}) in '
                         f'{type(arg)}.as_diagonal(): {axis}')

    k1 = a1 + arg._ndims

    # Roll this axis to the end
    rolled = np.moveaxis(arg._values, k1, -1)

    # Create the diagonal array
    new_values = np.zeros(rolled.shape + rolled.shape[-1:], dtype=rolled.dtype)

    # np.einsum('...ii->...i', new_values)[...] = rolled writes the diagonal in one call,
    # but it only overtakes this loop past about five components, and the item shapes here
    # are almost always shorter than that
    for i in range(rolled.shape[-1]):
        new_values[..., i, i] = rolled[..., i]

    # Roll the new axes back
    new_values = np.moveaxis(new_values, -1, k1)
    new_values = np.moveaxis(new_values, -1, k1)

    # Construct and cast
    obj = Qube._new_from_parts(new_values, arg._mask, nrank=arg._nrank + 1,
                               drank=arg._drank, unit=arg._unit, example=arg)
    obj = obj.cast(classes)

    # Diagonalize the derivatives if necessary
    if recursive:
        for key, deriv in arg._derivs.items():
            obj.insert_deriv(key, Qube.as_diagonal(deriv, axis, classes, False))

    return obj


def rms(self):
    """Calculate the root-mean-square values of all items as a Scalar.

    The RMS is computed across all item dimensions (numerator dimensions) for each
    array element. For a Vector with shape (n,) and numer (3,), this computes
    sqrt(sum(vals^2) / 3) for each of the n elements.

    Useful for looking at the overall magnitude of the differences between two objects.

    Returns:
        Scalar: The root-mean-square values of all items.
    """

    # Evaluate the norm
    sum_sq = np.sum(self._values**2, axis=tuple(range(-self._rank, 0)))

    return Qube._SCALAR_CLASS(np.sqrt(sum_sq / self.isize), self._mask)

################################################################################
