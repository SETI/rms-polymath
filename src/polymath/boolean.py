##########################################################################################
# polymath/boolean.py
##########################################################################################
"""The :class:`~polymath.Boolean` subclass, representing True and False values.

A Boolean is a :class:`~polymath.Scalar` whose values are booleans. Arithmetic on a
Boolean first converts it to an integer Scalar, so ``True`` behaves as one and ``False``
as zero. Masked elements make a Boolean three-valued; see :mod:`polymath.extensions.tvl`
for the operations that treat a masked value as "maybe" rather than as an error.
"""

import numpy as np

from polymath.qube import Qube
from polymath.scalar import Scalar

__all__ = ['Boolean']


class Boolean(Scalar):
    """Represent boolean values in the PolyMath framework.

    This class handles boolean values with masking support. Masked values are considered
    unknown, neither True nor False.
    """

    _NRANK = 0          # The number of numerator axes.
    _NUMER = ()         # Shape of the numerator.
    _FLOATS_OK = False  # True to allow floating-point numbers.
    _INTS_OK = False    # True to allow integers.
    _BOOLS_OK = True    # True to allow booleans.
    _UNITS_OK = False   # True to allow units; False to disallow them.
    _DERIVS_OK = False  # True to allow derivatives and denominators; False to disallow.
    _DEFAULT_VALUE = False

    @staticmethod
    def as_boolean(arg, *, recursive=True):
        """Convert the argument to Boolean if possible.

        Parameters:
            arg (BooleanLike): The object to convert to Boolean.
            recursive (bool, optional): This parameter is ignored for Boolean class but
                included for compatibility.

        Returns:
            Boolean: The converted Boolean object.
        """

        if isinstance(arg, Boolean):
            return arg

        if isinstance(arg, np.bool_):   # np.bool_ is not a subclass of bool
            arg = bool(arg)

        return Boolean(arg, unit=False, derivs={})

    def as_index(self):
        """An object suitable for indexing a NumPy ndarray.

        Returns:
            numpy.ndarray | bool: A boolean or boolean array with False values where
            values are False or masked.
        """

        return (self._values & self.antimask)

    def sum(self, axis=None, *, value=True, builtins=None, recursive=True, masked=None,
            out=None):
        """The sum of the unmasked values along the specified axis or axes.

        This is an override of :meth:`Qube.sum`, adding the `value` option to count False
        values instead of True values.

        Parameters:
            axis (int | tuple[int, ...] | None, optional): An integer axis or a tuple of
                axes. The sum is determined across these axes, leaving any remaining axes
                in the returned value. If None (the default), then the sum is performed
                across all axes of the object.
            value (bool, optional): True to count True values; False to count False
                values.
            builtins (bool | None, optional): If True and the result is a single unmasked
                scalar, the result is returned as a Python int instead of as an instance
                of Scalar. Default is the value specified by
                :meth:`~polymath.Qube.prefer_builtins`.
            recursive (bool, optional): Ignored for class Boolean.
            masked (int | None, optional): The value to return if `builtins` is True but
                the returned value is masked. Default is to return a masked value instead
                of a builtin type.
            out (Any, optional): Ignored. Enables ``np.sum(Boolean)`` to work.

        Returns:
            Scalar | int: The count of matching values (True or False) along the
            specified axis or axes.
        """

        if value:
            return self.as_int().sum(axis=axis, builtins=builtins, masked=masked)
        else:
            return (Scalar.ONE - self.as_int()).sum(axis=axis, builtins=builtins,
                                                    masked=masked)

    def identity(self):
        """An object of this subclass equivalent to the identity.

        This is an override of :meth:`Qube.identity`.

        Returns:
            Boolean: A read-only Boolean with value True.
        """

        return Boolean(True).as_readonly()

    ######################################################################################
    # Arithmetic operators
    ######################################################################################

    def __pos__(self, recursive=True):
        """The integer equivalent of this Boolean.

        This is an override of :meth:`Qube.__pos__`.

        Parameters:
            recursive (bool, optional): Ignored for Boolean.

        Returns:
            Scalar: Ones where this object is True, zeros where False.
        """

        return self.as_int()

    def __neg__(self, recursive=True):
        """The negated integer equivalent of this Boolean.

        This is an override of :meth:`Qube.__neg__`.

        Parameters:
            recursive (bool, optional): Ignored for Boolean.

        Returns:
            Scalar: A negated integer Scalar (-1 where True, 0 where False).
        """

        return -self.as_int()

    def __abs__(self, recursive=True):
        """The absolute value of this Boolean as an integer.

        This is an override of :meth:`Qube.__abs__`.

        Parameters:
            recursive (bool, optional): Ignored for Boolean.

        Returns:
            Scalar: Ones where this object is True, zeros where False.
        """

        return self.as_int()

    def __add__(self, /, arg, *, recursive=True):
        """``self + arg``, element-by-element addition after this Boolean is converted to
        an integer Scalar.

        This is an override of :meth:`Qube.__add__`.

        Parameters:
            arg (Any): The argument.
            recursive (bool, optional): Ignored for Boolean.

        Returns:
            Scalar: The sum.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        return self.as_int() + arg

    def __radd__(self, /, arg, *, recursive=True):
        """``arg + self``, element-by-element addition after this Boolean is converted to
        an integer Scalar.

        This is an override of :meth:`Qube.__radd__`.

        Parameters:
            arg (Any): The argument.
            recursive (bool, optional): Ignored for Boolean.

        Returns:
            Scalar: The sum.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        return self.as_int() + arg

    def __iadd__(self, /, arg):
        """``self += arg``; in-place addition is not supported for Boolean.

        This is an override of :meth:`Qube.__iadd__`.

        Parameters:
            arg (Any): The argument.

        Raises:
            TypeError: Always; in-place addition is not supported for Boolean.
        """

        Qube._raise_unsupported_op('+=', self)

    def __sub__(self, /, arg, *, recursive=True):
        """``self - arg``, element-by-element subtraction after this Boolean is converted
        to an integer Scalar.

        This is an override of :meth:`Qube.__sub__`.

        Parameters:
            arg (Any): The argument.
            recursive (bool, optional): Ignored for Boolean.

        Returns:
            Scalar: The difference.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        return self.as_int() - arg

    def __rsub__(self, /, arg, *, recursive=True):
        """``arg - self``, element-by-element subtraction after this Boolean is converted
        to an integer Scalar.

        This is an override of :meth:`Qube.__rsub__`.

        Parameters:
            arg (Any): The argument.
            recursive (bool, optional): Ignored for Boolean.

        Returns:
            Scalar: The difference.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        return -self.as_int() + arg

    def __isub__(self, /, arg):
        """``self -= arg``; in-place subtraction is not supported for Boolean.

        This is an override of :meth:`Qube.__isub__`.

        Parameters:
            arg (Any): The argument.

        Raises:
            TypeError: Always; in-place subtraction is not supported for Boolean.
        """

        Qube._raise_unsupported_op('-=', self)

    def __mul__(self, /, arg, *, recursive=True):
        """``self * arg``, element-by-element multiplication after this Boolean is
        converted to an integer Scalar.

        This is an override of :meth:`Qube.__mul__`.

        Parameters:
            arg (Any): The argument.
            recursive (bool, optional): Ignored for Boolean.

        Returns:
            Scalar: The product.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        return self.as_int() * arg

    def __rmul__(self, /, arg, *, recursive=True):
        """``arg * self``, element-by-element multiplication after this Boolean is
        converted to an integer Scalar.

        This is an override of :meth:`Qube.__rmul__`.

        Parameters:
            arg (Any): The argument.
            recursive (bool, optional): Ignored for Boolean.

        Returns:
            Scalar: The product.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        return self.as_int() * arg

    def __imul__(self, /, arg):
        """``self *= arg``; in-place multiplication is not supported for Boolean.

        This is an override of :meth:`Qube.__imul__`.

        Parameters:
            arg (Any): The argument.

        Raises:
            TypeError: Always; in-place multiplication is not supported for Boolean.
        """

        Qube._raise_unsupported_op('*=', self)

    def __truediv__(self, /, arg, *, recursive=True):
        """``self / arg``, element-by-element division after this Boolean is converted to
        a floating-point Scalar.

        This is an override of :meth:`Qube.__truediv__`.

        Parameters:
            arg (Any): The argument.
            recursive (bool, optional): Ignored for Boolean.

        Returns:
            Scalar: The quotient.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        return self.as_float() / arg

    def __rtruediv__(self, /, arg, *, recursive=True):
        """``arg / self``, element-by-element division after this Boolean is converted to
        a floating-point Scalar.

        This is an override of :meth:`Qube.__rtruediv__`.

        Parameters:
            arg (Any): The argument.
            recursive (bool, optional): Ignored for Boolean.

        Returns:
            Scalar: The quotient.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        if not isinstance(arg, Qube):
            arg = Scalar(arg)

        return arg / self.as_float()

    def __itruediv__(self, /, arg):
        """``self /= arg``; in-place division is not supported for Boolean.

        This is an override of :meth:`Qube.__itruediv__`.

        Parameters:
            arg (Any): The argument.

        Raises:
            TypeError: Always; in-place division is not supported for Boolean.
        """

        Qube._raise_unsupported_op('/=', self)

    def __floordiv__(self, /, arg):
        """``self // arg``, element-by-element floor division after this Boolean is
        converted to an integer Scalar.

        This is an override of :meth:`Qube.__floordiv__`.

        Parameters:
            arg (Any): The argument.

        Returns:
            Scalar: The result of the floor division.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        return self.as_int() // arg

    def __rfloordiv__(self, /, arg):
        """``arg // self``, element-by-element floor division after this Boolean is
        converted to an integer Scalar.

        This is an override of :meth:`Qube.__rfloordiv__`.

        Parameters:
            arg (Any): The argument.

        Returns:
            Scalar: The result of the floor division.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        if not isinstance(arg, Qube):
            arg = Scalar(arg)

        return arg // self.as_int()

    def __ifloordiv__(self, /, arg):
        """``self //= arg``; in-place floor division is not supported for Boolean.

        This is an override of :meth:`Qube.__ifloordiv__`.

        Parameters:
            arg (Any): The argument.

        Raises:
            TypeError: Always; in-place floor division is not supported for Boolean.
        """

        Qube._raise_unsupported_op('//=', self)

    def __mod__(self, /, arg):
        """``self % arg``, element-by-element modulus after this Boolean is converted to
        an integer Scalar.

        This is an override of :meth:`Qube.__mod__`.

        Parameters:
            arg (Any): The argument.

        Returns:
            Scalar: The remainder.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        return self.as_int() % arg

    def __rmod__(self, /, arg):
        """``arg % self``, element-by-element modulus after this Boolean is converted to
        an integer Scalar.

        This is an override of :meth:`Qube.__rmod__`.

        Parameters:
            arg (Any): The argument.

        Returns:
            Scalar: The remainder.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        if not isinstance(arg, Qube):
            arg = Scalar(arg)

        return arg % self.as_int()

    def __imod__(self, /, arg):
        """``self %= arg``; in-place modulo is not supported for Boolean.

        This is an override of :meth:`Qube.__imod__`.

        Parameters:
            arg (Any): The argument.

        Raises:
            TypeError: Always; in-place modulo is not supported for Boolean.
        """

        Qube._raise_unsupported_op('%=', self)

    def __pow__(self, /, arg):
        """``self ** arg``, element-by-element exponentiation after this Boolean is
        converted to an integer Scalar.

        This is an override of :meth:`Qube.__pow__`.

        Parameters:
            arg (Any): The exponent.

        Returns:
            Scalar: The result of the exponentiation.

        Raises:
            ValueError: If shapes are incompatible or `arg` has denominators.
        """

        arg = Scalar.as_scalar(arg)
        if arg.is_float():
            return self.as_float() ** arg

        self = self.as_int()

        # Result is 1 where self is True or arg == 0. The "|" of an integer with a boolean
        # is already an integer, for arrays and for single values alike.
        vals = self._values | (arg._values == 0)

        # Result is masked where self == 0 and arg < 0 or either item is masked
        invalid = (self._values == 0) & (arg._values < 0)
        return Scalar(vals, Qube.or_(self._mask, arg._mask, invalid))

    def __ipow__(self, /, arg):
        """``self **= arg``; in-place exponentiation is not supported for Boolean.

        This is an override of :meth:`Qube.__ipow__`. Exponentiation returns a Scalar, so
        the result cannot be stored back into a Boolean.

        Parameters:
            arg (Any): The exponent.

        Raises:
            TypeError: Always; in-place exponentiation is not supported for Boolean.
        """

        Qube._raise_unsupported_op('**=', self)

    ######################################################################################
    # Logical operators
    ######################################################################################

    def __le__(self, arg, *, builtins=True):
        """``self <= arg``, element-by-element "less than or equal" after this Boolean is
        converted to an integer Scalar.

        This is an override of :meth:`Qube.__le__`.

        Parameters:
            arg (Any): The object to compare with.
            builtins (bool, optional): If True and the result is a single unmasked scalar,
                return a Python bool instead of a Boolean object.

        Returns:
            Boolean | bool: True where this int value is less than or equal to the
            argument.

        Raises:
            ValueError: If the shapes are incompatible.
        """

        return self.as_int().__le__(arg, builtins=builtins)

    def __lt__(self, arg, *, builtins=True):
        """``self < arg``, element-by-element "less than" after this Boolean is converted
        to an integer Scalar.

        This is an override of :meth:`Qube.__lt__`.

        Parameters:
            arg (Any): The object to compare with.
            builtins (bool, optional): If True and the result is a single unmasked scalar,
                return a Python bool instead of a Boolean object.

        Returns:
            Boolean | bool: True where this int value is less than the argument.

        Raises:
            ValueError: If the shapes are incompatible.
        """

        return self.as_int().__lt__(arg, builtins=builtins)

    def __ge__(self, arg, *, builtins=True):
        """``self >= arg``, element-by-element "greater than or equal" after this Boolean
        is converted to an integer Scalar.

        This is an override of :meth:`Qube.__ge__`.

        Parameters:
            arg (Any): The object to compare with.
            builtins (bool, optional): If True and the result is a single unmasked scalar,
                return a Python bool instead of a Boolean object.

        Returns:
            Boolean | bool: True where this int value is greater than or equal to the
            argument.

        Raises:
            ValueError: If the shapes are incompatible.
        """

        return self.as_int().__ge__(arg, builtins=builtins)

    def __gt__(self, arg, *, builtins=True):
        """``self > arg``, element-by-element "greater than" after this Boolean is
        converted to an integer Scalar.

        This is an override of :meth:`Qube.__gt__`.

        Parameters:
            arg (Any): The object to compare with.
            builtins (bool, optional): If True and the result is a single unmasked scalar,
                return a Python bool instead of a Boolean object.

        Returns:
            Boolean | bool: True where this int value is greater than the argument.

        Raises:
            ValueError: If the shapes are incompatible.
        """

        return self.as_int().__gt__(arg, builtins=builtins)

##########################################################################################
# Useful class constants
##########################################################################################

Boolean.TRUE = Boolean(True).as_readonly()
Boolean.FALSE = Boolean(False).as_readonly()
Boolean.MASKED = Boolean(False, mask=True).as_readonly()

##########################################################################################
# Once the load is complete, we can fill in a reference to the Boolean class
# inside the Qube object.
##########################################################################################

Qube._BOOLEAN_CLASS = Boolean

##########################################################################################
