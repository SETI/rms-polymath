##########################################################################################
# polymath/extensions/dtypes.py: Data type interpretation and conversion
##########################################################################################

import numpy as np
import numbers
from polymath.qube import Qube, _NUMERIC_TYPES

__all__ = ['as_bool', 'as_float', 'as_int', 'as_numeric', 'dtype', 'is_bool',
           'is_float', 'is_int', 'is_numeric']

##########################################################################################
# Argument inspection and data type interpretation
##########################################################################################

@staticmethod
def _has_qube(arg):
    """True if the given list or tuple contains a Qube somewhere within."""

    if isinstance(arg, (list, tuple)):
        return (any(isinstance(item, Qube) for item in arg) or
                any(Qube._has_qube(item) for item in arg))

    return False


@staticmethod
def _has_masked_array(arg):
    """True if the given list or tuple contains a MaskedArray somewhere within."""

    if isinstance(arg, (list, tuple)):
        return (any(isinstance(item, np.ma.MaskedArray) for item in arg) or
                any(Qube._has_masked_array(item) for item in arg))

    return False


@staticmethod
def _as_values_and_mask(arg, opstr=''):
    """This object converted to a scalar or Numpy array with optional mask.

    Parameters:
        arg: object to convert to a scalar or array.
        opstr (str, optional): Name of operation string to include in any error
            message.

    Returns:
        tuple: (`value`, `mask`) as inferred from `arg`.

    Raises:
        TypeError: If the data type of `arg` is invalid.
    """

    # Ordered by how often each case arises, with the concrete types ahead of the ABC
    if type(arg) is np.ndarray:         # exact type, not a subclass
        return (arg, False)

    if isinstance(arg, Qube):
        return (arg._values, arg._mask)

    if isinstance(arg, _NUMERIC_TYPES):
        return (arg, False)

    if isinstance(arg, np.ma.MaskedArray):
        return (arg.data, arg.mask)

    if isinstance(arg, np.ndarray):
        return (arg, False)

    if isinstance(arg, Qube):
        return (arg._values, arg._mask)

    if isinstance(arg, (list, tuple)):
        if Qube._has_qube(arg):
            merged = Qube.stack(*arg)
            return (merged._values, merged._mask)
        elif Qube._has_masked_array(arg):
            merged = np.ma.stack(arg)
            return (merged.data, merged.mask)
        else:
            merged = np.array(arg)
            return (merged, False)

    if isinstance(arg, np.bool_):
        return (bool(arg), False)

    if isinstance(arg, numbers.Real):   # a numeric type registered with the ABC
        return (arg, False)

    _opstr = ' ' + opstr if opstr else ''
    raise TypeError(f'invalid{_opstr} data type: {type(arg)}')


@staticmethod
def _dtype_and_value(arg, masked_value=0, opstr=''):
    """Tuple (dtype, value), where dtype is one of "float", "int", or "bool".

    The value is converted to a builtin type if it is scalar; otherwise it is returned
    as an array with its original dtype.

    Parameters:
        arg (Qube, array-like, float, int, or bool): Object to interpret.
        masked_value (float, int, or bool): Value to use where `arg` is masked.
        opstr (str, optional): Name of operation to include in any error message.

    Returns:
        tuple: (`dtype`, `value`), where `dtype` is one of "float", "int", or "bool",
            and `value` is the result of converting `arg` to a NumPy.ndarray, float,
            int, or bool.

    Raises:
        TypeError: If the type of `arg` is invalid.
    """

    # Handle the easy and common cases first. A plain array is the most frequent
    # input by far, so it is recognized by its exact type before anything else.
    if type(arg) is np.ndarray:
        return Qube._array_dtype_and_value(arg, opstr=opstr)

    # Concrete scalar types, tested ahead of the ABCs
    if isinstance(arg, (bool, np.bool_)):
        return ('bool', bool(arg))

    if isinstance(arg, (int, np.integer)):
        return ('int', int(arg))

    if isinstance(arg, (float, np.floating)):
        return ('float', float(arg))

    # Any other ndarray subclass. Note that a MaskedArray is caught here and returned
    # with its mask intact, rather than by the masked-object handling further down.
    if isinstance(arg, np.ndarray):
        return Qube._array_dtype_and_value(arg, opstr=opstr)

    # A numeric type registered with an ABC but not listed above
    if isinstance(arg, numbers.Integral):
        return ('int', int(arg))

    if isinstance(arg, numbers.Real):
        return ('float', float(arg))

    # Convert a list or tuple to something else
    if isinstance(arg, (list, tuple)):
        if Qube._has_qube(arg):
            arg = Qube.stack(*arg)
        elif Qube._has_masked_array(arg):
            arg = np.ma.stack(arg)
        else:
            arg = np.array(arg)
            return Qube._dtype_and_value(arg, opstr=opstr)

    # Handle an object with a possible mask
    if isinstance(arg, Qube):
        mask = arg._mask
        arg = arg._values
    elif isinstance(arg, np.ma.MaskedArray):
        mask = arg.mask
        arg = arg.data
    else:
        _opstr = ' ' + opstr if opstr else ''
        raise TypeError(f'unsupported{_opstr} data type: {type(arg)}')

    # Interpret the argument ignoring its mask
    (dtype, arg) = Qube._dtype_and_value(arg, opstr=opstr)

    # Handle a shapeless mask
    if isinstance(mask, (bool, np.bool_)):
        if mask:                        # entirely masked
            return (dtype, Qube._casted_to_dtype(masked_value, dtype))
        else:                           # entirely unmasked
            return (dtype, arg)

    # Mask an array value
    arg = arg.copy()
    arg[mask] = masked_value
    return (dtype, arg)


@staticmethod
def _array_dtype_and_value(arg, opstr=''):
    """Tuple (dtype, value) for a NumPy array, where dtype is "float", "int", or
    "bool".

    Parameters:
        arg (numpy.ndarray): Array to interpret. It must not be a MaskedArray.
        opstr (str, optional): Name of operation to include in any error message.

    Returns:
        tuple: (`dtype`, `value`), where `dtype` is one of "float", "int", or "bool".
        A shapeless array is returned as a Python scalar.

    Raises:
        ValueError: If the dtype of `arg` is unsupported.
    """

    if arg.shape == ():             # shapeless array
        return Qube._dtype_and_value(arg[()], opstr=opstr)

    kind = arg.dtype.kind
    if kind == 'f':
        return ('float', arg)

    if kind in ('i', 'u'):
        return ('int', arg)

    if kind == 'b':
        return ('bool', arg)

    _opstr = ' ' + opstr if opstr else ''
    raise ValueError(f'unsupported{_opstr} dtype: {arg.dtype}')


@staticmethod
def _dtype(arg):
    """dtype of the given argument, one of "float", "int", or "bool"."""

    return Qube._dtype_and_value(arg)[0]


@staticmethod
def _casted_to_dtype(arg, dtype, masked_value=0):
    """This value casted to the specified dtype, one of "float", "int", or "bool".

    An object that is already of the requested type is returned unchanged.

    Note that converting floats to ints is always a "floor" operation, so -1.5 -> -2.

    Parameters:
        arg (Qube, array-like, float, int, or bool): Object to cast
        dtype (str): dtype to cast to, one of float", "int", or "bool".
        masked_value (float, int, or bool): Value to assign to a masked item in the
            case where the input argument is a Qube or MaskedArray.

    Returns:
        (numpy.ndarray, float, int, or bool): The result of the cast.
    """

    if isinstance(arg, (list, tuple)):
        arg = np.array(arg)

    if isinstance(arg, Qube):
        if arg._mask is False:
            arg = arg._values
        else:
            mask = arg._mask
            arg = arg.without_mask(recursive=False).copy()
            arg[mask] = masked_value
            arg = arg._values

    elif isinstance(arg, np.ma.MaskedArray):
        if arg.mask is False:
            arg = arg.data
        else:
            mask = arg.mask
            arg = arg.data.copy()
            arg[mask] = masked_value

    if isinstance(arg, np.ndarray):
        if arg.shape == ():
            return Qube._casted_to_dtype(arg[()], dtype)

        if dtype == 'float':
            if arg.dtype.kind == 'f':
                return arg
            return np.asarray(arg, dtype=np.double)

        if dtype == 'int':
            if arg.dtype.kind in ('i', 'u'):
                return arg
            return (arg // 1).astype('int')

        # must be bool
        if arg.dtype.kind == 'b':
            return arg

        return (arg != 0)

    # Handle shapeless
    if dtype == 'float':
        return float(arg)

    if dtype == 'int':
        if isinstance(arg, numbers.Integral):
            return int(arg)
        return int(arg // 1)

    # bool case
    if isinstance(arg, (bool, np.bool_)):
        return bool(arg)

    return (arg != 0)


def _suitable_dtype(cls, dtype='float', opstr=''):
    """The dtype for this Qube subclass closest to a given dtype.

    Parameters:
        cls (class): Qube subclass.
        dtype (str, optional): Default dtype, one of "float", "int", or "bool", to
            return if it is compatible with the subclass.
        opstr (str, optional): Name of the operation to include in any error message.

    Returns:
        str: One of "float", "int", or "bool".

    Raises:
        ValueError: If a suitable dtype cannot be determined.
    """

    if dtype == 'float':
        if cls._FLOATS_OK:
            return 'float'
        elif cls._INTS_OK:
            return 'int'
        else:
            return 'bool'

    elif dtype == 'int':
        if cls._INTS_OK:
            return 'int'
        elif cls._FLOATS_OK:
            return 'float'
        else:
            return 'bool'

    elif dtype == 'bool':
        if cls._BOOLS_OK:
            return 'bool'
        elif cls._INTS_OK:
            return 'int'
        else:
            return 'float'

    # Handle a NumPy dtype
    try:
        kind = np.dtype(dtype).kind
    except (TypeError, ValueError):
        pass
    else:
        if kind == 'f':
            return cls._suitable_dtype('float', opstr=opstr)
        if kind in ('i', 'u'):
            return cls._suitable_dtype('int', opstr=opstr)
        if kind == 'b':  # pragma: no cover
            return cls._suitable_dtype('bool', opstr=opstr)

    _in_opstr = ' in ' + opstr if opstr else ''
    raise ValueError(f'invalid dtype{_in_opstr}: "{dtype}"')


def _suitable_numer(cls, numer=None, opstr=''):
    """The given numerator made suitable for this class; ValueError otherwise.

    Parameters:
        cls (class): Qube subclass.
        numer (tuple, optional): Numerator shape to make suitable for use; None to
            return the default numerator shape for this Qube subclass.
        opstr (str, optional): Name of operation to include in any error message.

    Returns:
        tuple: Numerator shape.

    Raises:
        ValueError: If `numer` is unspecified and `cls` does not have a default.
        ValueError: If `numer` is incompatible with `cls`.
    """

    if numer is None:
        if cls._NUMER is not None:
            return cls._NUMER

        if not cls._NRANK:
            return ()

        _in_opstr = ' in ' + opstr if opstr else ''
        raise ValueError(f'class {cls} does not have a default numerator{_in_opstr}')

    numer = tuple(numer)

    opstr = opstr or cls.__name__
    if ((cls._NUMER is not None and numer != cls._NUMER) or
            (cls._NRANK is not None and len(numer) != cls._NRANK)):
        raise ValueError(f'invalid {opstr} numerator shape {numer}; '
                         f'must be {cls._NUMER}')

    return numer


def _suitable_value(cls, arg, *, numer=None, denom=(), expand=True, opstr=''):
    """This argument converted to a suitable value for this class.

    Parameters:
        cls (class): Qube subclass.
        arg (Qube, array-like, float, int, or bool): Object to be made suitable.
        numer (tuple, optional): Numerator shape; None for class default.
        denom (tuple, optional): Denominator shape.
        expand (bool, optional): True to expand the shape of the returned argument to
            the minimum required for the class; False to leave it with its original
            shape.
        opstr (str, optional): Name of operation to include in any error message.

    Returns:
        (numpy.ndarray, float, int, or bool): The value made suitable for `cls`.

    Raises:
        ValueError: If `arg` is incompatible with `cls`.
    """

    # Convert arg to a valid dtype
    (old_dtype, arg) = Qube._dtype_and_value(arg, opstr=opstr)
    new_dtype = cls._suitable_dtype(old_dtype, opstr=opstr)
    if new_dtype != old_dtype:
        arg = Qube._casted_to_dtype(arg, new_dtype)

    # Without expansion, we're done
    if not expand:
        return arg

    # Get the valid numerator
    numer = cls._suitable_numer(numer, opstr=opstr)

    # Expand the arg shape if necessary
    item = numer + denom
    if len(np.shape(arg)) < len(item):
        temp = np.empty(item, dtype=new_dtype)
        temp[...] = arg
        arg = temp

    return arg


##########################################################################################
# Data type conversions
##########################################################################################

def dtype(self):
    """One of "float", "int", or "bool", depending this object's value."""

    return Qube._dtype(self._values)


def is_numeric(self):
    """True if this object contains numbers; False if boolean."""

    if isinstance(self._values, (bool, np.bool_)):
        return False
    return not (isinstance(self._values, np.ndarray)
                and self._values.dtype.kind == 'b')


def as_numeric(self, *, recursive=True):
    """A numeric version of this object.

    Booleans are converted to Scalars.

    Parameters:
        recursive (bool, optional): True to include any derivatives; False to remove
            them.

    Returns:
        Qube: This object if it is already numeric; a Boolean is converted to a
        Scalar.
    """

    if self.is_numeric():
        return self if recursive else self.wod

    values = int(self._values) if self._is_scalar else self._values.astype(np.int8)
    return Qube._SCALAR_CLASS(values, self._mask, example=self, op='as_numeric()')


def is_float(self):
    """True if this object contains floats; False if ints or booleans."""

    if isinstance(self._values, np.ndarray):
        return self._values.dtype.kind == 'f'
    return isinstance(self._values, float)


def as_float(self, *, recursive=True, copy=False, builtins=False):
    """A floating-point version of this object.

    Booleans are converted to Scalars.

    Parameters:
        recursive (bool, optional): True to include any derivatives; False to remove
            them.
        copy (bool, optional): True to ensure that a new object with an independent
            copy of the values is returned.
        builtins (bool, optional): True to return a Python float if the returned value
            has shape (), is unmasked, and has no derivatives.

    Returns:
        Qube: The result.

    Raises:
        TypeError: If this object cannot contain floats.
    """

    if (builtins and self._is_scalar and not self._mask
            and not (recursive and self._derivs)):
        return float(self._values)

    if isinstance(self._values, np.ndarray) and self._values.dtype.kind == 'f':
        if copy:
            return self.copy(recursive=recursive)
        return self if recursive else self.wod

    cls = type(self)
    if cls is Qube._BOOLEAN_CLASS:
        cls = Qube._SCALAR_CLASS

    if not cls._FLOATS_OK:
        raise TypeError(f'{cls.__name__} object cannot contain floats')

    if self._is_scalar:
        values = float(self._values)
    else:
        values = self._values.astype(np.float64)
    derivs = self._derivs if recursive else {}

    obj = Qube.__new__(cls)
    obj.__init__(values, self._mask, derivs=derivs, example=self, op='as_float()')
    return obj


def is_int(self):
    """True if this object contains ints; False if floats or booleans."""

    if isinstance(self._values, np.ndarray):
        return self._values.dtype.kind in 'iu'
    if isinstance(self._values, bool):
        return False
    return isinstance(self._values, int)


def as_int(self, *, copy=False, builtins=False):
    """An integer version of this object.

    Booleans are converted to Scalars.

    Parameters:
        copy (bool, optional): True to ensure that a new object with an independent
            copy of the values is returned.
        builtins (bool, optional): True to return a Python float if the returned value
            has shape (), is unmasked, and has no derivatives.

    Returns:
        Qube or int: The result.

    Raises:
        TypeError: If this object cannot contain integers.
   """

    if builtins and self._is_scalar and not self._mask:
        return int(self._values)

    if isinstance(self._values, np.ndarray) and self._values.dtype.kind in 'iu':
        return self.__copy__() if copy else self

    cls = type(self)
    if cls is Qube._BOOLEAN_CLASS:
        cls = Qube._SCALAR_CLASS

    if not cls._INTS_OK:
        raise TypeError(f'{cls.__name__} object cannot contain ints')

    if self._is_scalar:
        values = int(self._values // 1)
    elif self._values.dtype.kind == 'b':
        values = self._values.astype(np.int8)
    else:
        values = (self._values // 1).astype(np.int64)

    obj = Qube.__new__(cls)
    obj.__init__(values, self._mask, example=self, op='as_int()')
    return obj


def is_bool(self):
    """True if this object contains booleans; False otherwise."""

    if isinstance(self._values, np.ndarray):
        return self._values.dtype.kind == 'b'
    return isinstance(self._values, bool)


def as_bool(self, *, copy=False, builtins=False):
    """A boolean version of this object.

    Scalars are converted to Booleans.

    Parameters:
        copy (bool, optional): True to ensure that a new object with an independent
            copy of the values is returned.
        builtins (bool, optional): True to return a Python float if the returned value
            has shape (), is unmasked, and has no derivatives.

    Returns:
        Qube: A copy of object converted to bools; if the values are already bools and
            `copy` is False, this object is returned unchanged.

    Raises:
        TypeError: If this object cannot contain bools.
    """

    if builtins and self._is_scalar and not self._mask:
        return bool(self._values)

    if isinstance(self._values, np.ndarray) and self._values.dtype.kind == 'b':
        return self.__copy__() if copy else self

    cls = type(self)
    if cls is Qube._SCALAR_CLASS:
        cls = Qube._BOOLEAN_CLASS

    if not cls._INTS_OK:  # pragma: no cover
        # This should never happen
        raise TypeError(f'{cls.__name__} object cannot contain bools')

    values = bool(self._values) if self._is_scalar else self._values.astype(np.bool_)
    obj = Qube.__new__(cls)
    obj.__init__(values, self._mask, example=self, op='as_bool()')
    return obj

##########################################################################################
