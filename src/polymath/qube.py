##########################################################################################
# polymath/qube.py: Base class for all PolyMath subclasses.
##########################################################################################

import math
import numpy as np
import numbers

from polymath.unit import Unit

__all__ = ['Qube']

# Concrete numeric types, tested ahead of the numbers ABCs. An isinstance() check against
# an ABC dispatches through __instancecheck__, several times slower than a check against a
# tuple of classes, and these run on every arithmetic operation. Where the answer must be
# exact, the ABC still has the last word, so that a type registered with it but not listed
# here, such as fractions.Fraction, is still recognized.
_NUMERIC_TYPES = (int, float, np.integer, np.floating)


class Qube:
    """The base class for all PolyMath subclasses.

    The PolyMath subclasses, e.g., Scalar, Vector3, Matrix3, etc., define one or more
    possibly multidimensional items. Unlike NumPy ndarrays, this class makes a clear
    distinction between the dimensions associated with the items and any additional,
    leading dimensions that define an array of such items.

    The "shape" is defined by the leading axes only, so a 2x2 array of 3x3 matrices would
    have shape (2,2,3,3) according to NumPy but has shape (2,2) according to PolyMath.
    Standard NumPy rules of broadcasting apply, but only on the array dimensions, not on
    the item dimensions. In other words, you can multiply a (2,2) array of 3x3 matrices by
    a (5,1,2) array of 3-vectors, yielding a (5,2,2) array of 3-vectors.

    PolyMath objects are designed as lightweight wrappers on NumPy ndarrays. All standard
    mathematical operators and indexing/slicing options are defined. One can generally mix
    PolyMath arithmetic with scalars, NumPy ndarrays, NumPy MaskedArrays, or anything
    array-like.

    In every object, a boolean mask is maintained in order to identify undefined array
    elements. Operations that would otherwise raise errors such as 1/0 and sqrt(-1) are
    masked out so that run-time errors can be avoided. See more about masks below.

    PolyMath objects also support embedded units using the Unit class. However, the
    internal values in a PolyMath object are always held in standard units of kilometers,
    seconds and radians, or arbitrary combinations thereof. The unit is primarily used
    to affect the appearance of numbers during input and output.

    PolyMath objects can be either read-only or read-write. Read-only objects are
    prevented from modification to the extent that Python makes this possible. Operations
    on read-only objects should always return read-only objects.

    PolyMath objects can track associated derivatives and partial derivatives, which are
    represented by other PolyMath objects. Mathematical operations generally carry all
    derivatives along so that, for example, if x.d_dt is the derivative of x with respect
    to t, then x.sin().d_dt will be the derivative of sin(x) with respect to t.

    The denominators of partial derivatives are represented by splitting the item shape
    into a numerator shape plus a denominator shape. As a result, for example, the partial
    derivatives of a Vector3 object (item shape (3,)) with respect to a Pair (item shape
    (2,)) will have overall item shape (3,2).

    The PolyMath subclasses generally do not constrain the shape of the denominator, just
    the numerator. As a result, the aforementioned partial derivatives can still be
    represented by a Vector3 object.

    Properties:
        shape (tuple):
            The leading axes of the object, i.e., those that are not considered part of
            the items.
        rank (int):
            The number of axes belonging to the items.
        nrank (int):
            The number of numerator axes associated with the items.
        drank (int):
            The number of denominator axes associated with the items.
        item (tuple):
            The shape of the individual items.
        numer (tuple):
            The shape of the numerator items.
        denom (tuple):
            The shape of the denominator items.
        values (numpy.ndarray, float, int, or bool):
            The object's data, with shape object.shape + object.item. If the object has a
            unit, then the values are in default units (km, sec, etc.) rather than in the
            specified unit.
        vals (numpy.ndarray, float, int, or bool):
            Alternative name for `values`.
        mask (numpy.ndarray or bool):
            The array's mask. A scalar False means the object is entirely unmasked; a
            scalar True means it is entirely masked. Otherwise, it is a boolean array of
            shape object.shape.
        unit (Unit or None):
            The unit of the array, if any. None indicates no unit.
        derivs (dict):
            A dictionary of the names and values of any derivatives, each represented by
            additional PolyMath object.
        readonly (bool):
            True if the object cannot (or at least should not) be modified. A determined
            user may be able to alter a read-only object, but the API makes this more
            difficult.
        size (int):
            The number of elements in the shape.
        isize (int):
            The number of elements in each item.
        nsize (int):
            The number of elements in the numerator of the items.
        dsize (int):
            The number of elements in the denominator of the items.

    Notes:
        PolyMath objects are not hashable. They compare by value and are mutable, so they
        cannot be used as dictionary keys or placed in sets.

        Nothing here is synchronized. Reading a shared object from several threads is
        safe, but modifying one while another thread reads it is not, and neither is
        changing any of the global settings, such as those of
        :meth:`~Qube.prefer_builtins` and :meth:`~Qube.set_default_pickle_digits`, once
        other threads are running. Confine each object to one thread, or serialize access
        to it yourself.
    """

    # This prevents binary operations of the form:
    #   <np.ndarray> <op> <Qube>
    # from executing the ndarray operation instead of the polymath operation
    __array_priority__ = 1

    # Global attribute to be used for testing
    _DISABLE_CACHE = False

    # If this global is set to True, the shrink/unshrink methods are disabled.
    # Calculations done with and without shrinking should always produce the same results,
    # although they may be slower with shrinking disabled. Used for testing and debugging.
    _DISABLE_SHRINKING = False

    # If this global is set to True, the unshrunk method will ignore any cached value of
    # its un-shrunken equivalent. Used for testing and debugging.
    _IGNORE_UNSHRUNK_AS_CACHED = False

    # Default class constants, to be overridden as needed by subclasses...
    _NRANK = None       # The number of numerator axes; None to leave this unconstrained.
    _NUMER = None       # Shape of the numerator; None to leave unconstrained.
    _FLOATS_OK = True   # True to allow floating-point numbers.
    _INTS_OK = True     # True to allow integers.
    _BOOLS_OK = True    # True to allow booleans.
    _UNITS_OK = True    # True to allow units; False to disallow them.
    _DERIVS_OK = True   # True to allow derivatives and denominators; False to disallow.

    # The class that represents a derivative of this class. A derivative does not satisfy
    # the constraint that defines some classes, so such a class names a more general
    # substitute here. None means that a derivative has the same class as the object.
    _DERIV_CLASS = None

    def __new__(subtype, *values, **keywords):
        """Create a new, un-initialized object given a Qube subclass."""

        return object.__new__(subtype)

    def __init__(self, arg, mask=False, *, derivs={},  # noqa: B006  # {} and None
                 unit=None, nrank=None, drank=None,    # are distinct, documented values
                 example=None, default=None, op=''):
        """Default constructor.

        Parameters:
            arg (Qube, array-like, float, int, or bool): An object to define the numeric
                value(s) of the returned object. If this object is read-only, then the
                returned object will be entirely read-only. Otherwise, the object will be
                read-writable. The values are generally given in standard units of km,
                seconds and radians, regardless of the specified unit.
            mask (Boolean, array-like, or bool, optional): The mask for the object. Use
                None to copy the mask from the example object. False (the default) leaves
                the object un-masked.
            derivs (dict, optional): Derivatives represented as PolyMath objects. Use None
                to make a copy of the derivs attribute of the example object, or {} (the
                default) for no derivatives. All derivatives are broadcasted to the shape
                of the object if necessary.
            unit (Unit, optional): The unit of the object. Use None to infer the unit from
                the example object; use False to suppress the unit.
            nrank (int, optional): The number of numerator axes in the returned object;
                None to derive the rank from the input data and/or the subclass.
            drank (int, optional): The number of denominator axes in the returned object;
                None to derive it from the input data and/or the subclass.
            example (Qube, optional): Another Qube object from which to copy any input
                arguments except derivs that have not been explicitly specified.
            default (array-like, float, int, or bool): Value to use where masked. This is
                typically a constant that will not "break" most arithmetic calculations.
                If it is an array, it must be of the same shape as the items.
            op (str, optional): Name of an operation to include in an error message if
                something goes wrong.

        Raises:
            TypeError: If the data type of `arg` or `mask` is invalid.
            TypeError: If `example` is not an instance of Qube.
            ValueError: If the shape of `mask` is incompatible with object.
            TypeError: If `unit` is specified but is disallowed by the Qube subclass.
            ValueError: If `derivs` are specified but are disallowed by the Qube
                subclass.
            ValueError: If `nrank` is incompatible with the Qube subclass.
            ValueError: If `drank` is specified but the Qube subclass disallows
                derivatives.
            ValueError: If the dimensions of `arg` are incompatible with the subclass.
        """

        opstr = Qube._opstr(self, op)
        nrank_given = nrank is not None

        # Set defaults based on a Qube input
        if isinstance(arg, Qube):

            if derivs is None:
                derivs = arg._derivs.copy()     # shallow copy

            if unit is None:
                unit = arg._unit

            if nrank is None:
                nrank = arg._nrank
            elif nrank != arg._nrank:           # nranks _must_ be compatible
                self._nrank = nrank
                Qube._raise_incompatible_numers(op, self, arg)

            if drank is None:
                drank = arg._drank
            elif drank != arg._drank:           # dranks _must_ be compatible
                self._drank = drank
                Qube._raise_incompatible_denoms(op, self, arg)

            if default is None:
                default = arg._default

        # Set defaults based on an example object
        if example is not None:

            if not isinstance(example, Qube):
                raise TypeError(f'{opstr} example value is not a Qube subclass')

            if mask is None:
                mask = example._mask

            if unit is None and self._UNITS_OK:
                unit = example._unit

            if nrank is None and self._NRANK is None:
                nrank = example._nrank

            if drank is None:
                drank = example._drank

            if default is None:
                default = example._default

        # Validate inputs. An explicitly given numerator rank is honored as it stands,
        # including an explicit zero, and is checked against the subclass below. A rank
        # inherited from `arg` or `example` is only a starting point: the subclass default
        # outranks an inherited zero, which is what lets Matrix(scalar) reinterpret the
        # trailing axes of a rank-0 object as its items.
        if not nrank_given:
            nrank = nrank or self._NRANK or 0
        if drank is None:
            drank = 0
        rank = nrank + drank

        if derivs and not self._DERIVS_OK:
            raise ValueError(f'{opstr} derivatives are disallowed')

        if unit and not self._UNITS_OK:
            raise TypeError(f'{opstr} unit is disallowed: {unit}')

        if self._NRANK is not None and nrank != self._NRANK:
            raise ValueError(f'invalid {opstr} numerator rank: {nrank}')

        if drank and not self._DERIVS_OK:
            raise ValueError(f'{opstr} denominators are disallowed')

        # Get the value and check its shape
        (values, arg_mask) = Qube._as_values_and_mask(arg, opstr=opstr)
        full_shape = np.shape(values)
        if len(full_shape) < rank:
            raise ValueError(f'invalid {opstr} array shape {full_shape}: '
                             f'minimum rank = {nrank} + {drank}')

        dd = len(full_shape) - drank
        nn = dd - nrank
        denom = full_shape[dd:]
        numer = full_shape[nn:dd]
        item  = full_shape[nn:]
        shape = full_shape[:nn]

        # Fill in the values
        self._values = self._suitable_value(values, numer=numer, denom=denom,
                                            opstr=opstr)
        self._is_array = isinstance(self._values, np.ndarray)
        self._is_scalar = not self._is_array

        # Get the mask and check its shape
        mask = Qube.or_(arg_mask, Qube._as_mask(mask, opstr=opstr))
        collapse = isinstance(arg, np.ma.MaskedArray)
        self._mask = Qube._suitable_mask(mask, shape=shape, broadcast=True,
                                         collapse=collapse, check=False, opstr=opstr)

        # Fill in the remaining shape info
        self._shape = shape
        self._ndims = len(shape)
        self._rank  = rank
        self._nrank = nrank
        self._drank = drank
        self._item  = item
        self._numer = numer
        self._denom = denom

        # The example supplies the products of the shape and of the item shape whenever
        # those carried through, exactly as in _new_from_parts()
        if example is not None and example._item == item and example._nrank == nrank:
            self._isize = example._isize
            self._nsize = example._nsize
            self._dsize = example._dsize
        else:
            self._nsize = math.prod(numer)
            self._dsize = math.prod(denom)
            self._isize = self._nsize * self._dsize

        if example is not None and example._shape == shape:
            self._size = example._size
        else:
            self._size = math.prod(shape)

        # Fill in the unit
        self._unit = None if Qube.is_one_false(unit) else unit

        # The object is read-only if the values array is read-only
        self._readonly = Qube._array_is_readonly(self._values)

        if self._readonly:
            Qube._array_to_readonly(self._mask)

        # Used for anything we want to cache in association with an object. This cache
        # will be cleared whenever the object is modified in any way.
        self._cache = {}

        # Install the derivs (converting to read-only if necessary)
        self._derivs = {}
        if derivs:
            self.insert_derivs(derivs)

        # Used only for if clauses; filled in when needed
        self._truth_if_any = False
        self._truth_if_all = False

        # Fill in the default
        dtype = Qube._dtype(self._values)
        if default is not None and np.shape(default) == item:
            self._default = Qube._casted_to_dtype(default, dtype)
        else:
            self._default = type(self)._default_for(item, drank, dtype)

    ######################################################################################
    # Builtin type support
    ######################################################################################

    _PREFER_BUILTIN_TYPES = False

    @staticmethod
    def prefer_builtins(status=None):
        """Set a global flag defining whether certain functions return a Python builtin
        type, rather than a Qube subclass, if possible.

        Parameters:
            status (bool, optional): True to favor Python builtin types; False otherwise.
                Omit this input to leave the global setting unchanged (but return it).

        Returns:
            bool: True if builtins are globally preferred; False otherwise.
        """

        if status is not None:
            Qube._PREFER_BUILTIN_TYPES = status

        return Qube._PREFER_BUILTIN_TYPES

    def as_builtin(self, masked=None):
        """This object as a Python built-in class (float, int, or bool) if the conversion
        can be done without loss of information.

        Parameters:
            masked (float, int, or bool, optional): Value to return if the shape of this
                object is () and it is masked.

        Returns:
            (Qube, float, int, bool, or None): This object's `values` attribute if its
            shape is () and it is unmasked; the value of `masked` if the shape is () and
            it is masked; otherwise, this object.
        """

        values = self._values
        if np.size(values) == 0:
            return self         # previously, erroneously returned `masked`
        if np.shape(values):
            return self

        # Now we know shape is ()
        if self._mask:
            return self if masked is None else masked

        if not self.is_unitless():
            return self

        if isinstance(values, (bool, np.bool_)):
            return bool(values)
        if isinstance(values, numbers.Integral):
            return int(values)
        if isinstance(values, numbers.Real):
            return float(values)

        return self  # pragma: no cover # This shouldn't happen

    ######################################################################################
    # Alternative constructors
    ######################################################################################

    # The attributes that describe every Qube, in the order __init__ assigns them.
    # "_derivs" and "_cache" are excluded because the methods that copy an object always
    # decide what to do with them separately. Keep this in step with __init__.
    _TRANSFERABLE_ATTRS = ('_values', '_mask', '_is_array', '_is_scalar', '_shape',
                           '_ndims', '_rank', '_nrank', '_drank', '_item', '_numer',
                           '_denom', '_size', '_isize', '_nsize', '_dsize', '_unit',
                           '_readonly', '_truth_if_any', '_truth_if_all', '_default')

    # Attributes that an object carries only once something has set them
    _OPTIONAL_ATTRS = ('_pickle_digits', '_pickle_reference')

    # The names of the attributes added by add_attr(). This class-level value is shared by
    # every object that has not added one, so it is never modified in place; add_attr()
    # replaces it with a new frozenset instead.
    _added_attrs = frozenset()

    @staticmethod
    def _transfer_attrs(source, dest, *, added_attrs=True):
        """Copy the descriptive attributes of one object onto another.

        Derivatives and the cache are not copied; the caller decides what those should
        be. The attributes are named explicitly rather than discovered from the instance
        dictionary, because reading __dict__ materializes it and forfeits the inline
        attribute storage that CPython would otherwise give both objects.

        Parameters:
            source (Qube): The object to copy from.
            dest (Qube): The object to copy onto.
            added_attrs (bool, optional): True to copy the attributes added by
                add_attr(), which are transferred by reference; False to omit them.
        """

        for attr in Qube._TRANSFERABLE_ATTRS:
            setattr(dest, attr, getattr(source, attr))

        for attr in Qube._OPTIONAL_ATTRS:
            if hasattr(source, attr):
                setattr(dest, attr, getattr(source, attr))

        added = source._added_attrs if added_attrs else ()
        if added:
            dest._added_attrs = added
            for attr in added:
                setattr(dest, attr, getattr(source, attr))

    def clone(self, *, recursive=True, preserve=(), retain_cache=False):
        """Fast construction of a shallow copy.

        The copy carries any attributes added by add_attr().

        Parameters:
            recursive (bool, optional): True to clone the derivatives of this object;
                False to ignore them.
            preserve (list, optional): Name(s) of derivatives to include even if
                `recursive` is False.
            retain_cache (bool, optional): True to retain cache except "unshrunk" and
                "wod"; False to return clone with an empty cache.

        Returns:
            Qube: The shallow clone.
        """

        return self._clone(recursive=recursive, preserve=preserve,
                           retain_cache=retain_cache, added_attrs=True)

    def _clone_new_values(self, *, recursive=True, retain_cache=False):
        """Fast construction of a shallow copy that is about to be given new values.

        This is the counterpart to clone() for an operation, such as a negation or a
        multiplication, that builds its result by copying this object and then replacing
        the values. Any attributes added by add_attr() describe this object's own values,
        so they are not carried onto a copy whose values are about to become something
        else.

        Parameters:
            recursive (bool, optional): True to clone the derivatives of this object;
                False to ignore them.
            retain_cache (bool, optional): True to retain cache except "unshrunk" and
                "wod"; False to return clone with an empty cache.

        Returns:
            Qube: The shallow clone.
        """

        return self._clone(recursive=recursive, preserve=(),
                           retain_cache=retain_cache, added_attrs=False)

    def _clone(self, *, recursive, preserve, retain_cache, added_attrs):
        """Fast construction of a shallow copy, with or without the added attributes.

        Parameters:
            recursive (bool): True to clone the derivatives of this object; False to
                ignore them.
            preserve (list): Name(s) of derivatives to include even if `recursive` is
                False.
            retain_cache (bool): True to retain cache except "unshrunk" and "wod"; False
                to return clone with an empty cache.
            added_attrs (bool): True to carry the attributes added by add_attr() onto the
                copy; False to omit them.

        Returns:
            Qube: The shallow clone.
        """

        obj = Qube.__new__(type(self))

        # Transfer attributes other than derivatives and cache
        Qube._transfer_attrs(self, obj, added_attrs=added_attrs)
        obj._derivs = {}
        obj._cache = {}

        # Handle derivatives recursively
        if recursive:
            new_keys = set(self._derivs.keys())
        elif preserve:
            if isinstance(preserve, str):
                new_keys = {preserve}
            else:
                new_keys = set(preserve)
        else:
            new_keys = set()

        for key in new_keys:
            deriv = self._derivs[key]
            new_deriv = deriv.clone(recursive=False, retain_cache=retain_cache)
            obj.insert_deriv(key, new_deriv)

        # Handle cache
        if retain_cache:
            obj._cache = self._cache.copy()
            if 'shrunk' in obj._cache:
                del obj._cache['shrunk']
            if 'wod' in obj._cache:
                del obj._cache['wod']
        else:
            obj._cache = {}

        return obj

    @classmethod
    def zeros(cls, shape, dtype='float', *, numer=None, denom=(), mask=False):
        """New object of this class and shape, filled with zeros.

        Parameters:
            shape (tuple): Shape of the object.
            dtype (str, optional): One of "bool", "int", or "float", defining the data
                type. Ignored if `cls` has a default dtype.
            numer (tuple, optional): Numerator shape; None to use default for `cls`.
            denom (tuple, optional): Denominator shape.
            mask (array-like or bool, optional): Mask to apply.

        Returns:
            Qube: The new object.
        """

        dtype = cls._suitable_dtype(dtype)
        numer = cls._suitable_numer(numer)

        obj = Qube.__new__(cls)
        obj.__init__(np.zeros(shape + numer + denom, dtype=dtype),
                     mask=mask, drank=len(denom))
        return obj

    @classmethod
    def ones(cls, shape, dtype='float', *, numer=None, denom=(), mask=False):
        """New object of this class and shape, filled with ones.

        Parameters:
            shape (tuple): Shape of the object.
            dtype (str, optional): One of "bool", "int", or "float", defining the data
                type. Ignored if `cls` has a default dtype.
            numer (tuple, optional): Numerator shape; None to use default for `cls`.
            denom (tuple, optional): Denominator shape.
            mask (array-like or bool, optional): Mask to apply.

        Returns:
            Qube: The new object.
        """

        dtype = cls._suitable_dtype(dtype)
        numer = cls._suitable_numer(numer)

        obj = Qube.__new__(cls)
        obj.__init__(np.ones(shape + numer + denom, dtype=dtype),
                     mask=mask, drank=len(denom))
        return obj

    @classmethod
    def _new_from_parts(cls, values, mask=False, *, nrank, drank=0, unit=None,
                        example=None):
        """Fast construction of an object from parts that are already known to be valid.

        This is the internal counterpart to the constructor. It performs none of the type
        checking, dtype coercion or shape inference that `__init__` performs, so it is
        only suitable for operations that have already computed the result themselves.
        The caller guarantees that:

        * `values` is a NumPy array, or a Python or NumPy scalar, whose dtype is already
          one that `cls` permits;
        * `mask` is a bool or a boolean array broadcastable to the leading shape of
          `values`;
        * `nrank` and `drank` correctly describe the trailing axes of `values`, and are
          consistent with `cls`.

        Derivatives are never carried over; insert them into the returned object instead.

        Parameters:
            values (numpy.ndarray, float, int, or bool): The values of the new object.
            mask (numpy.ndarray or bool, optional): The mask of the new object.
            nrank (int): The number of numerator axes at the end of `values`.
            drank (int, optional): The number of denominator axes at the end of `values`.
            unit (Unit, optional): The unit of the new object; None for unitless.
            example (Qube, optional): An object from which to take the default value when
                its item shape and dtype match those of the new object, and from which to
                take the products of the shape and of the item shape when those match.
                It is used only to avoid repeating work and never changes the result.

        Returns:
            Qube: The new object, without derivatives.
        """

        obj = Qube.__new__(cls)

        is_array = isinstance(values, np.ndarray)
        full_shape = values.shape if is_array else ()

        if is_array and not full_shape:     # a shapeless array is stored as a scalar
            values = values[()]
            is_array = False

        if not is_array and isinstance(values, np.generic):
            values = values.item()          # the constructor reduces NumPy scalars too

        ndims = len(full_shape) - nrank - drank
        shape = full_shape[:ndims]
        item = full_shape[ndims:]

        # The values of two operands may broadcast against each other while their masks do
        # not, so a mask can still be narrower than the values it describes
        if isinstance(mask, np.ndarray) and mask.shape != shape:
            mask = Qube._array_to_readonly(np.broadcast_to(mask, shape))

        obj._values = values
        obj._mask = mask
        obj._is_array = is_array
        obj._is_scalar = not is_array

        obj._shape = shape
        obj._ndims = ndims
        obj._rank  = nrank + drank
        obj._nrank = nrank
        obj._drank = drank
        obj._item  = item
        obj._numer = full_shape[ndims:ndims + nrank]
        obj._denom = full_shape[ndims + nrank:]

        # The item products depend only on the item shape and on the way it divides into
        # a numerator and a denominator, so the example supplies them whenever both of
        # those carried through the operation. Failing that, the numerator and
        # denominator products multiply to give the item product, so the item shape does
        # not need a pass of its own.
        if example is not None and example._item == item and example._nrank == nrank:
            obj._isize = example._isize
            obj._nsize = example._nsize
            obj._dsize = example._dsize
        else:
            obj._nsize = math.prod(obj._numer)
            obj._dsize = math.prod(obj._denom)
            obj._isize = obj._nsize * obj._dsize

        # Likewise, the shape product comes from the example whenever the shape did
        if example is not None and example._shape == shape:
            obj._size = example._size
        else:
            obj._size = math.prod(shape)

        obj._unit = unit
        obj._readonly = is_array and not values.flags['WRITEABLE']
        obj._cache = {}
        obj._derivs = {}
        obj._truth_if_any = False
        obj._truth_if_all = False

        if obj._readonly:
            Qube._array_to_readonly(mask)

        # The default depends only on the item shape and the dtype, so the example
        # supplies it whenever both of those carried through the operation
        if (example is not None and example._item == item
                and is_array == example._is_array
                and (values.dtype == example._values.dtype if is_array
                     else type(values) is type(example._values))):
            obj._default = example._default
        else:
            obj._default = cls._default_for(item, drank, Qube._dtype(values))

        return obj

    @classmethod
    def _default_for(cls, item, drank, dtype):
        """The default value for an object of this class, item shape and dtype.

        Parameters:
            cls (type): Qube subclass.
            item (tuple): Shape of the items.
            drank (int): The number of denominator axes.
            dtype (str): One of "float", "int", or "bool".

        Returns:
            (numpy.ndarray, float, int, or bool): The value to use where masked.
        """

        if hasattr(cls, '_DEFAULT_VALUE') and drank == 0:
            default = cls._DEFAULT_VALUE
        elif item:
            default = np.ones(item)
        else:
            default = 1

        return Qube._casted_to_dtype(default, dtype)

    @classmethod
    def filled(cls, shape, fill=0, *, numer=None, denom=(), mask=False):
        """Internal object of this class and shape, filled with a constant.

        Parameters:
            shape (tuple): Shape of the object.
            fill (array-like, float, int, or bool, optional): The constant value for each
                item. It must be compatible with the item shape of `cls`.
            numer (tuple, optional): Numerator shape; None to use default for `cls`.
            denom (tuple, optional): Denominator shape.
            mask (array-like or bool, optional): Mask to apply.

        Returns:
            Qube: The new object.

        Raises:
            ValueError: If `fill` is not compatible with the `cls`.
        """

        # Create example object with shape == ()
        example = Qube.__new__(cls)
        example.__init__(cls._suitable_value(fill, numer=numer, denom=denom),
                         drank=len(denom))

        # For a shapeless object, return the example
        if not shape:
            if not mask:
                return example
            example = example.remask(mask)
            return example

        # Return the filled object
        vals = np.empty(shape + example._item, dtype=example.dtype())
        vals[...] = example._values

        obj = Qube.__new__(cls)
        obj.__init__(vals, mask=mask, example=example, drank=len(denom))
        return obj

    ######################################################################################
    # Low-level access
    ######################################################################################

    def _set_values(self, values, mask=None, *, antimask=None, retain_cache=False):
        """Low-level method to update the values of an array.

        The read-only status of the object is defined by that of the given value.

        Parameters:
            values (array-like, float, int, or bool): New values.
            mask (array-like or bool, optional): New mask.
            antimask (array-like or bool, optional): If provided, then only the array
                locations associated with the antimask are modified.
            retain_cache (bool, optional): If True, the cache values are retained except
                for "unshrunk".

        Returns:
            Qube: This object, updated.

        Raises:
            TypeError: If the type of `values` or `mask` is invalid.
            ValueError: If the shape of `values`, `mask`, or `antimask` is invalid.
        """

        # Confirm shapes
        shape = np.shape(self._values)
        shape1 = np.shape(values)
        if shape1 != shape:
            raise ValueError(f'value shape mismatch: {shape1}, {shape}')

        if mask is not None:
            mshape = np.shape(mask)
            if mshape and mshape != shape:
                raise ValueError(f'mask shape mismatch: {mshape}, {shape}')

        # Update values
        if antimask is not None:
            ashape = np.shape(antimask)
            if ashape != shape:
                raise ValueError(f'antimask shape mismatch: {ashape}, {shape}')
            self._values[antimask] = values[antimask]
        else:
            if isinstance(values, np.generic):
                if isinstance(values, np.floating):
                    values = float(values)
                elif isinstance(values, np.integer):
                    values = int(values)
                else:
                    values = bool(values)
            self._values = values

        self._readonly = Qube._array_is_readonly(self._values)

        # Update the mask if necessary
        if mask is not None:
            if antimask is None:
                self._mask = mask
            elif isinstance(mask, np.ndarray):
                self._mask[antimask] = mask[antimask]
            else:
                if not isinstance(self._mask, np.ndarray):
                    old_mask = self._mask
                    self._mask = np.empty(self._shape, dtype=np.bool_)
                    self._mask.fill(old_mask)
                self._mask[antimask] = mask

        # Handle the cache
        if retain_cache and mask is None:
            if 'unshrunk' in self._cache:
                del self._cache['unshrunk']
        else:
            self._cache.clear()

        # Set the readonly state based on the values given
        if np.shape(self._mask):
            if self._readonly:
                self._mask = Qube._array_to_readonly(self._mask)
            elif Qube._array_is_readonly(self._mask):
                self._mask = self._mask.copy()

        return self

    def _new_values(self):
        """Low-level method to indicate that values have changed.

        This means "unshrunk" will be deleted from the cache if present.
        """

        if 'unshrunk' in self._cache:
            del self._cache['unshrunk']

    def _set_mask(self, mask, *, antimask=None, check=False):
        """Low-level method to update the mask of an array.

        The read-only status of the object will be preserved.

        Parameters:
            mask (array-like or bool, optional): New mask.
            antimask (array-like or bool, optional): If provided, then only the array
                locations associated with the antimask are modified.
            check (bool, optional): True to check for an array containing all False
                values, and if so, replace it with a single value of False.

        Returns:
            Qube: This object, updated.

        Raises:
            TypeError: If the type of `mask` is invalid.
            ValueError: If the mask is incompatible with the required shape.
        """

        # Cast the mask and confirm the shape
        mask = Qube._suitable_mask(mask, self._shape, check=check)
        is_readonly = self._readonly

        if antimask is None:
            self._mask = mask
        elif isinstance(mask, np.ndarray):
            self._mask[antimask] = mask[antimask]
        else:
            if not isinstance(self._mask, np.ndarray):
                old_mask = self._mask
                self._mask = np.empty(self._shape, dtype=np.bool_)
                self._mask.fill(old_mask)
            self._mask[antimask] = mask

        self._cache.clear()

        if isinstance(self._mask, np.ndarray):
            if is_readonly:
                self._mask = Qube._array_to_readonly(self._mask)

            elif Qube._array_is_readonly(self._mask):
                self._mask = self._mask.copy()

        return self

    ######################################################################################
    # Properties
    ######################################################################################

    @property
    def values(self):
        """The value of this object as a numpy.ndarray, float, int, or bool."""

        return self._values

    @property
    def vals(self):
        """The value of this object as a numpy.ndarray, float, int, or bool."""

        return self._values       # Handy shorthand

    @property
    def mvals(self):
        """This object as a NumPy ma.MaskedArray."""

        # Deal with a scalar
        if self._is_scalar:
            if self._mask:
                return np.ma.masked
            else:
                return np.ma.MaskedArray(self._values)

        # Deal with a scalar mask
        if isinstance(self._mask, (bool, np.bool_)):
            if self._mask:
                return np.ma.MaskedArray(self._values, True)
            else:
                return np.ma.MaskedArray(self._values)

        # For zero rank, the mask is already the right size
        if self._rank == 0:
            return np.ma.MaskedArray(self._values, self._mask)

        # Expand the mask
        mask = self._mask.reshape(self._shape + self._rank * (1,))
        mask = np.broadcast_to(mask, self._values.shape)
        return np.ma.MaskedArray(self._values, mask)

    @property
    def mask(self):
        """The boolean mask of this object as a NumPy.ndarray or bool."""

        return self._mask

    @property
    def antimask(self):
        """The inverse of the mask of this object, True wherever an element is valid."""

        if not Qube._DISABLE_CACHE and 'antimask' in self._cache:
            return self._cache['antimask']

        if isinstance(self._mask, np.ndarray):
            # Read-only, because every caller receives this same array
            antimask = Qube._array_to_readonly(np.logical_not(self._mask))
            self._cache['antimask'] = antimask
            return antimask

        antimask = not self._mask
        self._cache['antimask'] = antimask
        return antimask

    @property
    def default(self):
        """The default element value for this object."""

        return self._default

    @property
    def unit_(self):
        """The Unit of this object."""

        return self._unit

    @property
    def units(self):
        """The Unit of this object; alternative name for `unit_`."""

        return self._unit

    @property
    def derivs(self):
        """The dictionary of derivatives of this object."""

        return self._derivs

    @property
    def shape(self):
        """The shape of this object as a tuple."""

        return self._shape

    @property
    def ndims(self):
        """The number of dimensions in this object (excluding items)."""

        return self._ndims          # alternative name

    @property
    def ndim(self):
        """The number of dimensions in this object (excluding items)."""

        return self._ndims

    @property
    def rank(self):
        """The rank of this object."""

        return self._rank

    @property
    def nrank(self):
        """The rank of the element numerator in this object."""

        return self._nrank

    @property
    def drank(self):
        """The rank of the element denominator in this object."""

        return self._drank

    @property
    def item(self):
        """The shape of the elements in this object as a tuple."""

        return self._item

    @property
    def numer(self):
        """The shape of the element numerator in this object as a tuple."""

        return self._numer

    @property
    def denom(self):
        """The shape of the element denominator in this object as a tuple."""

        return self._denom

    @property
    def size(self):
        """The number of elements in this object's shape."""

        return self._size

    @property
    def isize(self):
        """The number of components in this object's items."""

        return self._isize

    @property
    def nsize(self):
        """The number of numerator components in this object's items."""

        return self._nsize

    @property
    def dsize(self):
        """The number of denominator components in this object's items."""

        return self._dsize

    @property
    def readonly(self):
        """True if this object is read-only; False otherwise."""

        return self._readonly

    ######################################################################################
    # Cache support
    ######################################################################################

    def _clear_cache(self):
        """Clear the cache."""

        self._cache.clear()

    def _find_corners(self):
        """Update the corner indices such that everything outside this defined "hypercube"
        is masked.
        """

        if self._ndims == 0:
            return None

        index0 = self._ndims * (0,)
        if isinstance(self._mask, (bool, np.bool_)):
            if self._mask:
                return (index0, index0)
            else:
                return (index0, self._shape)

        lower = []
        upper = []
        antimask = self.antimask

        for axis in range(self._ndims):
            other_axes = list(range(self._ndims))
            del other_axes[axis]

            occupied = np.any(antimask, tuple(other_axes))
            indices = np.where(occupied)[0]
            if len(indices) == 0:
                return (index0, index0)

            lower.append(indices[0])
            upper.append(indices[-1] + 1)

        return (tuple(lower), tuple(upper))

    @property
    def corners(self):
        """Corners of a "hypercube" that contain all the unmasked array elements.

        Returns:
            (tuple, tuple): The first tuple defines the lower coordinates of the unmasked
            region, and the second tuple defines the upper coordinates.
        """

        if not Qube._DISABLE_CACHE and 'corners' in self._cache:
            return self._cache['corners']

        corners = self._find_corners()
        self._cache['corners'] = corners
        return corners

    @staticmethod
    def _slicer_from_corners(corners):
        """A slice object based on corners specified as a tuple of indices."""

        slice_objects = []
        for axis in range(len(corners[0])):
            slice_objects.append(slice(corners[0][axis], corners[1][axis]))

        return tuple(slice_objects)

    @staticmethod
    def _shape_from_corners(corners):
        """Array shape based on corner indices."""

        shape = []
        for axis in range(len(corners[0])):
            shape.append(corners[1][axis] - corners[0][axis])

        return tuple(shape)

    @property
    def _slicer(self):
        """A slice object containing all the array elements inside the current corners."""

        if not Qube._DISABLE_CACHE and 'slicer' in self._cache:
            return self._cache['slicer']

        slicer = Qube._slicer_from_corners(self.corners)
        self._cache['slicer'] = slicer
        return slicer

    ######################################################################################
    # I/O operations
    ######################################################################################

    def __repr__(self):
        """Express the value as a string.

        The format of the returned string is `Class([value, value, ...], suffixes, ...)`,
        where the quanity inside square brackets is the result of str() applied to a NumPy
        ndarray.

        The suffixes are, in order...

        * "denom=(shape)" if the object has a denominator;
        * "mask" if the object has a mask
        * the name of the unit of the object has a unit
        * the names of all the derivatives in alphabetical order

        Returns:
            str: String representation
        """

        return self.__str__()

    def __str__(self):
        """Express the value as a string.

        The format of the returned string is `Class([value, value, ...], suffixes, ...)`,
        where the quanity inside square brackets is the result of str() applied to a NumPy
        ndarray.

        The suffixes are, in order...

        * "denom=(shape)" if the object has a denominator;
        * "mask" if the object has a mask
        * the name of the unit of the object has a unit
        * the names of all the derivatives in alphabetical order

        Returns:
            str: String representation
        """

        suffix = []

        # Indicate the denominator shape if necessary
        if self._denom != ():
            suffix += ['denom=' + str(self._denom)]

        # Masked objects have a suffix ', mask'
        is_masked = np.any(self._mask)
        if is_masked:
            suffix += ['mask']

        # Objects with a unit include the unit in the suffix
        if not self.is_unitless():
            suffix += [str(self._unit)]

        # Objects with derivatives include a list of the names
        if self._derivs:
            keys = list(self._derivs.keys())
            keys.sort()
            for key in keys:
                suffix += ['d_d' + key]

        # Generate the value string
        scaled = self.into_unit(recursive=False)    # apply the unit
        if self._is_scalar:
            if is_masked:
                string = '--'
            else:
                string = str(scaled)
        elif is_masked:
            temp = Qube(scaled, self._mask, example=self, derivs={})
            string = str(temp.mvals)[1:-1]
        else:
            string = str(scaled)[1:-1]

        # Add an extra set of brackets around derivatives
        if self._denom:
            string = '[' + string + ']'

        # Concatenate the results
        if len(suffix) == 0:
            suffix = ''
        else:
            suffix = '; ' + ', '.join(suffix)

        return type(self).__name__ + '(' + string + suffix + ')'

    ######################################################################################
    # from_scalars() special method
    ######################################################################################

    @classmethod
    def from_scalars(cls, *scalars, recursive=True, readonly=False, classes=()):
        """A new instance constructed from Scalars or arrays given as arguments.

        Defined as a class method so it can also be used to generate instances of any 1-D
        subclass.

        Parameters:
            *scalars (Qube, array-like, float, or int):
                One or more Scalars or objects that can be converted to Scalars.
            recursive (bool, optional):
                True to construct the derivatives as the union of the derivatives of all
                the components' derivatives. False to return an object without
                derivatives.
            readonly (bool, optional):
                True to return a read-only object; False (the default) to return something
                potentially writable.
            classes: (class or list[class]):
                A list defining the preferred class of the returned object. The first
                suitable class in the list will be used; default is [Vector].

        Returns:
            Qube: A new object constructed from the inputs and using the first suitable
            class within `classes`.

        Raises:
            ValueError: If two of the `scalars` have incompatible denominators.
        """

        # Convert to scalars and broadcast to the same shape
        args = []
        for arg in scalars:
            scalar = Qube._SCALAR_CLASS.as_scalar(arg)
            args.append(scalar)

        scalars = Qube.broadcast(*args, recursive=recursive)

        # Tabulate the properties and construct the value array
        new_unit = None
        new_denom = None

        arrays = []
        masks = []
        deriv_dicts = []
        has_derivs = False
        dtype = np.int64
        for scalar in scalars:
            arrays.append(scalar._values)
            masks.append(scalar._mask)

            new_unit = new_unit or scalar._unit
            Unit.require_match(new_unit, scalar._unit)

            if new_denom is None:
                new_denom = scalar._denom
            elif new_denom != scalar._denom:
                raise ValueError(f'incompatible denominators in {cls}.from_scalars(): '
                                 f'{scalar._denom}, {new_denom}')

            deriv_dicts.append(scalar._derivs)
            if len(scalar._derivs):
                has_derivs = True

            # Remember any floats encountered
            if scalar.is_float():
                dtype = np.float64

        # Construct the values array
        new_drank = len(new_denom)
        new_values = np.array(arrays, dtype=dtype)
        new_values = np.moveaxis(new_values, 0, new_values.ndim - new_drank - 1)

        # Construct the mask (scalar or array)
        masks = Qube.broadcast(*masks)
        new_mask = Qube.or_(*masks)

        # Construct the object
        obj = Qube.__new__(cls)
        obj.__init__(new_values, new_mask, unit=new_unit, nrank=scalars[0]._nrank + 1,
                     drank=new_drank)
        obj = obj.cast(classes)

        # Insert derivatives if necessary
        if recursive and has_derivs:
            new_derivs = {}

            # Find one example of each derivative
            examples = {}
            for deriv_dict in deriv_dicts:
                for key, deriv in deriv_dict.items():
                    examples[key] = deriv

            for key, example in examples.items():
                items = []
                if example._item:
                    missing_deriv = Qube(np.zeros(example._item), nrank=example._nrank,
                                         drank=example._drank, op='from_scalars()')
                else:
                    missing_deriv = 0.

                for deriv_dict in deriv_dicts:
                    items.append(deriv_dict.get(key, missing_deriv))

                new_derivs[key] = Qube.from_scalars(*items, recursive=False,
                                                    readonly=readonly, classes=classes)
            obj.insert_derivs(new_derivs)

        return obj

##########################################################################################
