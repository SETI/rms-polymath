##########################################################################################
# polymath/unit.py
##########################################################################################

from collections import defaultdict
import functools
import math
import numbers
import re

import numpy as np

__all__ = ['Unit']

_TOKEN   = re.compile(r'\*\*|[-+]?\d+|[A-Za-z]+|\S')
_INTEGER = re.compile(r'[-+]?\d+')


class Unit:
    """Class to represent units and provide conversion methods.

    Attributes:
        exponents (tuple[int, int, int]): The exponents on dimensions of length, time, and
            angle, respectively.
        triple (tuple[int, int, int]): Three integers representing the exact factor that
            to multiply a value in this unit by to a value in standard units involving
            (km, seconds, and radians). This factor is represented by three numbers,
            (**numer**, **denom**, and **expo**), where the exact factor equals
            (`numer/denom * pi**expo`).
        name (str, dict or None): An optional name for this unit. Alternatively, a name
            can be defined by a dictionary keyed by unit names, returning exponents. For
            example, the name "km/s" can be given by `{"km":1, "s":-1}`.

    Examples:
        * `degree` unit is represented by exponents (0, 0, 1) and triple (1, 180, 1),
          because one multiplies degrees by pi/180 to obtain radians.
        * `m/s` unit is represented by exponents (1, -1, 0) and triple (1, 1000, 0),
          because one multiplies meters per second by 1/1000 to obtain km per second.

    Notes:
        * Most common unit values are defined as class constants, e.g., **Unit.DEGREE**
          and **Unit.STER**.
        * In most situations, a given Unit value of None is equivalent to
          **Unit.UNITLESS**, the Unit associated with a dimensionless value.
    """

    def __init__(self, exponents, triple, name=None):
        """Initialize a Unit object.

        Parameters:
            exponents (tuple[int, int, int]): A tuple of integers defining the exponents
                on distance, time and angle that are used for this unit.
            triple (tuple[int, int, int]): A tuple containing:

                * [0] The numerator of a factor that converts from a value in this unit to
                  a value using standard units of km, seconds, and radians.
                * [1] The denominator of this same factor.
                * [2] The exponent on pi that should multiply the numerator of this
                  factor.

            name (str or dict, optional): The name of the unit. It is represented by a
                string or by a dictionary of unit exponents keyed by the unit names.

        Notes:
            For example, a unit of degrees would have a triple (1,180,1). This defines a
            factor pi/180, which converts from degrees to radians.
        """

        self.exponents = tuple(exponents)

        # Convert to coefficients to ints with lowest common denominator if possible
        # Scale by 256 to compensate for possible floats that can be represented exactly
        numer = int(triple[0] * 256)
        denom = int(triple[1] * 256)

        gcd_value = math.gcd(numer, denom)
        numer //= gcd_value
        denom //= gcd_value

        if numer * triple[1] != denom * triple[0]:
            numer = triple[0]
            denom = triple[1]

        pi_expo = triple[2]

        self.triple = (numer, denom, pi_expo)

        # Factor to convert from these units to standard units
        self.factor = (numer / denom) * np.pi**pi_expo

        # Factor to convert from standard units to these units
        self.factor_inv = (denom / numer) / np.pi**pi_expo

        # Fill in the name
        self.name = name

    @property
    def from_unit_factor(self):
        return self.factor

    @property
    def into_unit_factor(self):
        return self.factor_inv

    @staticmethod
    def as_unit(arg):
        """Convert the given argument to a Unit object.

        Parameters:
            arg: The argument to convert. Can be an object of class Unit, one of the
                standard unit names, or None.

        Returns:
            Unit or None: The converted Unit object, or None if arg is None.

        Raises:
            ValueError: If the argument is not a recognized unit.
        """

        if arg is None:
            return None
        elif isinstance(arg, str):
            return Unit._NAME_TO_UNIT[arg]
        elif isinstance(arg, Unit):
            return arg
        else:
            raise ValueError('not a recognized unit: ' + str(arg))

    @staticmethod
    def can_match(first, second):
        """Check if the unit can match.

        Parameters:
            first (Unit or None): The first unit object.
            second (Unit or None): The second unit object.

        Returns:
            bool: True if the units can match, meaning that either they have the same
            exponents or one or both are None.
        """

        if first is None or second is None:
            return True

        return first.exponents == second.exponents

    @staticmethod
    def require_compatible(first, second, info=''):
        """Raise a ValueError if the arguments are not compatible units.

        Parameters:
            first (Unit or None): The first unit object.
            second (Unit or None): The second unit object.
            info (str, optional): Info to embed into the error message.

        Raises:
            ValueError: If the units are not compatible.
        """

        if not Unit.can_match(first, second):
            info_ = info + ' ' if info else ''
            raise ValueError(f'{info_}units are not compatible: {first}, {second}')

    @staticmethod
    def do_match(first, second):
        """Check if the units match.

        Parameters:
            first (Unit or None): The first unit object.
            second (Unit or None): The second unit object.

        Returns:
            bool: True if the units match, meaning that they have the same exponents.
            Values of None are treated as equivalent to unitless.
        """

        if first is None:
            first = Unit.UNITLESS
        if second is None:
            second = Unit.UNITLESS

        return first.exponents == second.exponents

    @staticmethod
    def require_match(first, second, info=''):
        """Raise a ValueError if the units are not the same.

        Parameters:
            first (Unit or None): The first unit object.
            second (Unit or None): The second unit object.
            info (str, optional): Info to embed into the error message.

        Raises:
            ValueError: If the units are not compatible.
        """

        if not Unit.do_match(first, second):
            info_ = info + ' ' if info else ''
            raise ValueError(f'{info_}units do not match: {first}, {second}')

    @staticmethod
    def is_angle(arg):
        """Check if the argument could be used as an angle.

        Parameters:
            arg (Unit or None): The unit object to check.

        Returns:
            bool: True if the argument could be used as an angle.
        """

        if arg is None:
            return True
        return (arg.exponents in ((0, 0, 0), (0, 0, 1)))

    @staticmethod
    def require_angle(arg, info=''):
        """Raise a ValueError if the argument could not be used as an angle.

        Parameters:
            arg (Unit or None): The unit object to check.
            info (str, optional): Info to embed into the error message.

        Raises:
            ValueError: If the units are incompatible with an angle.
        """

        if not Unit.is_angle(arg):
            info_ = info + ' ' if info else ''
            raise ValueError(f'{info_}unit is not compatible with an angle: {arg}')

    @staticmethod
    def is_unitless(arg):
        """True if the argument is unitless.

        Parameters:
            arg (Unit or None): The unit object to check.

        Returns:
            bool: True if the argument is unitless.
        """

        if arg is None:
            return True
        return (arg.exponents == (0, 0, 0))

    @staticmethod
    def require_unitless(arg, info=''):
        """Raise a ValueError if the argument is not unitless.

        Parameters:
            arg (Unit or None): The unit object to check.
            info (str, optional): Info to embed into the error message.

        Raises:
            ValueError: If a unit is not permitted.
        """

        if Unit.is_unitless(arg):
            return

        info_ = info + ' ' if info else ''
        raise ValueError(f'{info_}unit is not permitted: {arg}')

    def from_this(self, value):
        """Convert a scalar or numpy array in this unit to a standard unit.

        Parameters:
            value (scalar or numpy.ndarray): The value to convert from this unit to
                standard units of km, seconds and radians.

        Returns:
            scalar or numpy.ndarray: The value converted to a standard unit.
        """

        return self.factor * value

    def into_this(self, value):
        """Convert a scalar or numpy array from a standard unit to this unit.

        Parameters:
            value (scalar or numpy.ndarray): The value to convert from a standard unit
                to this unit.

        Returns:
            scalar or numpy.ndarray: The converted value in this unit.
        """

        return self.factor_inv * value

    @staticmethod
    def from_unit(unit, value):
        """Convert a scalar or numpy array in the given unit to a standard unit.

        Parameters:
            unit (Unit or None): The unit to convert from.
            value (scalar or numpy.ndarray): The value to convert.

        Returns:
            scalar or numpy.ndarray: The `value` converted from the given `unit` to
                standard units involving km, seconds and radians. If `unit` is None,
                `value` is returned untouched.
        """

        if unit is None:
            return value

        return unit.factor * value

    @staticmethod
    def into_unit(unit, value):
        """Convert a scalar or numpy array from a standard unit to given unit.

        Parameters:
            unit (Unit or None): The unit to convert to.
            value (scalar or numpy.ndarray): The value to convert.

        Returns:
            scalar or numpy.ndarray: The `value` in standard units involving km, seconds
                and radians converted to the given `unit`. If `unit` is None, `value` is
                returned untouched.
        """

        if unit is None:
            return value

        return unit.factor_inv * value

    def convert(self, value, unit, info=''):
        """Convert the unit of a scalar or NumPy array.

        The value is assumed to be in this unit, and it is returned in the new unit
        specified. Conversions are exact whenever possible.

        Parameters:
            value (scalar or numpy.ndarray): The value to convert.
            unit (Unit or None): The target unit. If None, converts to unitless.
            info (str, optional): Info to embed into the error message.

        Returns:
            scalar or numpy.ndarray: The converted value in the target unit.

        Raises:
            ValueError: If the units are incompatible for conversion.
        """

        if unit is None:
            unit = Unit.UNITLESS

        if self.exponents != unit.exponents:
            _info = (' in ' + info) if info else ''
            raise ValueError(f'cannot convert unit {self} to {unit}{_info}')

        # If the factor is unity, return the value without modification
        if (self.triple[2] == unit.triple[2] and
                self.triple[0] * unit.triple[1] == self.triple[1] * unit.triple[0]):
            return value

        factor = (self.triple[0] * unit.triple[1] / (self.triple[1] * unit.triple[0]) *
                  np.pi**(self.triple[2] - unit.triple[2]))
        return factor * value

    ######################################################################################
    # Arithmetic operators
    ######################################################################################

    def __mul__(self, arg):
        """Multiply this Unit object by another Unit object or scalar.

        Parameters:
            arg (Unit, None, or numbers.Real): The object to multiply by.

        Returns:
            Unit: The product of the unit multiplication. If the type of `arg` is not
            supported, NotImplemented is returned instead, so that Python falls back on
            the reflected operation of `arg`.

        Raises:
            TypeError: If neither operand supports the multiplication.
        """

        if isinstance(arg, Unit):
            return Unit((self.exponents[0] + arg.exponents[0],
                         self.exponents[1] + arg.exponents[1],
                         self.exponents[2] + arg.exponents[2]),
                        (self.triple[0] * arg.triple[0],
                         self.triple[1] * arg.triple[1],
                         self.triple[2] + arg.triple[2]),
                        Unit._mul_names(self.name, arg.name))

        if arg is None:
            return self

        if isinstance(arg, numbers.Real):
            return self * (Unit((0, 0, 0), (arg, 1,  0)))

        return NotImplemented

    def __rmul__(self, arg):
        return self.__mul__(arg)

    def __div__(self, arg):
        return self.__truediv__(arg)

    def __rdiv__(self, arg):
        return self.__rtruediv__(arg)

    def __truediv__(self, arg):
        """Divide this Unit object by another Unit object or scalar.

        Parameters:
            arg (Unit, None, or numbers.Real): The object to divide by.

        Returns:
            Unit: The quotient of the unit division. If the type of `arg` is not
            supported, NotImplemented is returned instead, so that Python falls back on
            the reflected operation of `arg`.

        Raises:
            TypeError: If neither operand supports the division.
        """

        if isinstance(arg, Unit):
            return Unit((self.exponents[0] - arg.exponents[0],
                         self.exponents[1] - arg.exponents[1],
                         self.exponents[2] - arg.exponents[2]),
                        (self.triple[0] * arg.triple[1],
                         self.triple[1] * arg.triple[0],
                         self.triple[2] - arg.triple[2]),
                        Unit._div_names(self.name, arg.name))

        if arg is None:
            return self

        if isinstance(arg, numbers.Real):
            return self * (Unit((0, 0, 0), (1, arg, 0)))

        return NotImplemented

    def __rtruediv__(self, arg):
        """Divide a scalar by this Unit object.

        Parameters:
            arg (None or numbers.Real): The scalar to divide.

        Returns:
            Unit: The reciprocal of this Unit object multiplied by arg. If the type of
            `arg` is not supported, NotImplemented is returned instead, so that Python
            falls back on the reflected operation of `arg`.

        Raises:
            TypeError: If neither operand supports the division.
        """

        if arg is None:
            arg = 1.

        if isinstance(arg, numbers.Real):
            return (self / arg)**(-1)

        return NotImplemented

    def __pow__(self, power):
        """Raise this Unit object to the specified power.

        Parameters:
            power (int or float): The exponent. Must be an integer or half-integer.

        Returns:
            Unit: This Unit object raised to the specified power.

        Raises:
            ValueError: If the power is not an integer or half-integer.
        """

        ipower = int(power)
        if power != ipower:
            if 2*power == int(2*power):
                return self.sqrt()**(int(2 * power))
            else:
                raise ValueError('units can only be raised to integer or half-integer '
                                 f'powers: {power}')
        else:
            power = ipower

        if power > 0:
            return Unit((power * self.exponents[0],
                         power * self.exponents[1],
                         power * self.exponents[2]),
                        (self.triple[0]**power,
                         self.triple[1]**power,
                         power * self.triple[2]),
                        Unit._name_power(self.name, power))
        else:
            return Unit((power * self.exponents[0],
                         power * self.exponents[1],
                         power * self.exponents[2]),
                        (self.triple[1]**(-power),
                         self.triple[0]**(-power),
                         power * self.triple[2]),
                        Unit._name_power(self.name, power))

    def sqrt(self):
        """The square root of this Unit object.

        Returns:
            Unit: The square root of this Unit object.

        Raises:
            ValueError: If the exponents are not even numbers.
        """

        if (self.exponents[0] % 2 != 0 or self.exponents[1] % 2 != 0
                or self.exponents[2] % 2 != 0):
            raise ValueError("illegal unit for sqrt(): " + self.get_name())

        exponents = (self.exponents[0]//2, self.exponents[1]//2, self.exponents[2]//2)

        numer = np.sqrt(self.triple[0])
        denom = np.sqrt(self.triple[1])
        if numer % 1 == 0:
            numer = int(numer)
        if denom % 1 == 0:
            denom = int(denom)

        pi_expo = self.triple[2] // 2
        if self.triple[2] != 2*pi_expo:
            numer *= np.pi**(self.triple[2] / 2.)
            pi_expo = 0

        name = Unit._name_power(self.name, 0.5)
        return Unit(exponents, (numer, denom, pi_expo), name)

    #####################################################
    # Static versions of arithmetic operations
    #####################################################

    @staticmethod
    def mul_units(arg1, arg2):
        """Multiply two Unit objects.

        Parameters:
            arg1 (Unit or None): The first Unit object.
            arg2 (Unit or None): The second Unit object.

        Returns:
            Unit or None: The product of the two Unit objects, or None if both arguments
            are None.
        """

        if arg1 is None:
            return arg2
        if arg2 is None:
            return arg1

        return arg1 * arg2

    @staticmethod
    def div_units(arg1, arg2):
        """Divide two Unit objects.

        Parameters:
            arg1 (Unit or None): The numerator Unit object.
            arg2 (Unit or None): The denominator Unit object.

        Returns:
            Unit or None: The quotient of the two Unit objects, or None if both arguments
            are None.
        """

        if arg1 is None:
            if arg2 is None:
                return None
            else:
                return arg2**(-1)

        return arg1 / arg2

    @staticmethod
    def sqrt_unit(unit):
        """The square root of a Unit object.

        Parameters:
            unit (Unit or None): The Unit object to take the square root of.

        Returns:
            Unit or None: The square root of the Unit object, or None if unit is None.

        Raises:
            ValueError: If the exponents are not even numbers.
        """

        if unit is None:
            return None

        return unit.sqrt()

    @staticmethod
    def unit_power(unit, power):
        """Raise a Unit object to the specified power.

        Parameters:
            unit (Unit or None): The Unit object to raise to a power.
            power (int or float): The exponent. Must be an integer or half-integer.

        Returns:
            Unit or None: The Unit object raised to the specified power, or None if unit
            is None.

        Raises:
            ValueError: If the power is not an integer or half-integer.
        """

        if unit is None:
            return None

        return unit**power

    ######################################################################################
    # Comparison operators
    ######################################################################################

    def __eq__(self, arg):
        """Check if this Unit object equals another.

        Parameters:
            arg (Unit or None): The Unit object to compare with.

        Returns:
            bool: True if the Unit objects are equal, False otherwise.
        """

        if not isinstance(arg, Unit):
            return False

        return (self.exponents == arg.exponents and self.factor == arg.factor)

    def __ne__(self, arg):
        """Check if this Unit object does not equal another.

        Parameters:
            arg (Unit or None): The Unit object to compare with.

        Returns:
            bool: True if the Unit objects are not equal, False otherwise.
        """

        if not isinstance(arg, Unit):
            return True

        return (self.exponents != arg.exponents or self.factor != arg.factor)

    ######################################################################################
    # Copy operations
    ######################################################################################

    def __copy__(self):
        return Unit(self.exponents, self.triple, self.name)

    def copy(self):
        """A copy of this Unit object.

        Returns:
            Unit: A copy of this Unit object.
        """

        return self.__copy__()

    ######################################################################################
    # String operations
    ######################################################################################

    def __str__(self):
        """A string representation of this Unit object.

        Returns:
            str: A string representation of the unit.
        """

        return self.get_name()

    def __repr__(self):
        """A detailed string representation of this Unit object.

        Returns:
            str: A detailed string representation of the unit.
        """

        return f'Unit({self})'

    @staticmethod
    def _mul_names(name1, name2):
        """Multiply two unit names.

        Parameters:
            name1 (str, dict, or None): The first unit name.
            name2 (str, dict, or None): The second unit name.

        Returns:
            str or dict or None: The product of the two unit names, or None if both
            arguments are None.
        """

        if name1 is None or name2 is None:
            return None

        name1 = Unit.name_to_dict(name1)
        name2 = Unit.name_to_dict(name2)

        new_name = name1.copy()
        for key, expo in name2.items():
            if key in new_name:
                expo += new_name[key]

            # pop() rather than del, because create_name() yields a zero exponent for
            # every unused dimension, and those keys need not appear in name1 at all
            if expo == 0:
                new_name.pop(key, None)
            else:
                new_name[key] = expo

        return new_name

    @staticmethod
    def _div_names(name1, name2):
        """Divide two unit names.

        Parameters:
            name1 (str, dict, or None): The numerator unit name.
            name2 (str, dict, or None): The denominator unit name.

        Returns:
            str or dict or None: The quotient of the two unit names, or None if both
            arguments are None.
        """

        if name1 is None or name2 is None:
            return None

        name1 = Unit.name_to_dict(name1)
        name2 = Unit.name_to_dict(name2)

        new_name = name1.copy()
        for key, expo in name2.items():
            if key in new_name:
                expo -= new_name[key]

            # pop() rather than del, because create_name() yields a zero exponent for
            # every unused dimension, and those keys need not appear in name1 at all
            if expo == 0:
                new_name.pop(key, None)
            else:
                new_name[key] = -expo

        return new_name

    @staticmethod
    def _name_power(name, power):
        """Raise a unit name to the specified power.

        Parameters:
            name (str, dict, or None): The unit name to raise to a power.
            power (int or float): The exponent.

        Returns:
            dict or None: The unit name raised to the specified power. The result is None
            if `name` is None, and also if the power would give any name a non-integer
            exponent, because no name written in these units can express the result. A
            unit left unnamed this way derives a name from its dimensions instead.

        Raises:
            ValueError: If the power is a string that does not denote an integer.
        """

        if name is None:
            return None

        name = Unit.name_to_dict(name)

        if isinstance(power, str):
            old_power = power
            power = Unit.name_to_dict(power)

            if not isinstance(power, int):
                raise ValueError(f'non-integer power on unit: "{old_power}"')

        new_name = {}

        for key, expo in name.items():
            new_power = expo * power
            int_power = int(new_power)
            if new_power != int_power:
                return None

            new_name[key] = int_power

        return new_name

    @staticmethod
    def name_to_dict(expr):
        """Convert a unit expression string to a dictionary.

        Parameters:
            expr (str or dict): The unit expression string to convert. It can contain "**"
                for exponentiation, "*" for multiply, and "/" for divide. It can contain
                nested substrings inside parentheses.

        Returns:
            dict: A dictionary keyed by each name that appears, whose value is the net
            number of times that name was multiplied, counting each division as -1. A
            name that cancels out entirely is absent from the result.

        Raises:
            ValueError: If the expression is neither a string nor a dictionary, or if it
                contains an unrecognized character, unbalanced parentheses, a missing
                operand, or a "**" not followed by an integer.
        """

        def parse_group():
            """Parse tokens up to the end or the next ")" and return their exponents."""

            nonlocal pos

            result = defaultdict(int)
            sign = 1
            while pos < len(tokens) and tokens[pos] != ')':
                token = tokens[pos]
                pos += 1

                if token == '(':
                    item = parse_group()
                    if pos >= len(tokens):
                        raise ValueError(f'missing ")" in unit "{expr}"')
                    pos += 1                        # step over the ")"
                elif token.isalpha():
                    item = {token: 1}
                else:
                    raise ValueError(f'unexpected "{token}" in unit "{expr}"')

                power = 1
                if pos < len(tokens) and tokens[pos] == '**':
                    pos += 1
                    if pos >= len(tokens) or not _INTEGER.fullmatch(tokens[pos]):
                        raise ValueError(f'"**" without an integer in unit "{expr}"')
                    power = int(tokens[pos])
                    pos += 1

                for name, count in item.items():
                    result[name] += sign * power * count

                if pos < len(tokens) and tokens[pos] in ('*', '/'):
                    sign = 1 if tokens[pos] == '*' else -1
                    pos += 1
                    if pos >= len(tokens) or tokens[pos] == ')':
                        raise ValueError(f'missing operand in unit "{expr}"')

            return result

        # A dictionary is already in the returned form, so it passes through untouched
        if isinstance(expr, dict):
            return expr

        if not isinstance(expr, str):
            raise ValueError(f'unit is not a string: "{expr}"')

        tokens = _TOKEN.findall(expr)
        pos = 0

        result = parse_group()
        if pos < len(tokens):
            raise ValueError(f'unbalanced ")" in unit "{expr}"')

        return {k: v for k, v in result.items() if v}

    @staticmethod
    def name_to_str(namedict):
        """Convert a unit name dictionary to a string.

        Parameters:
            namedict (dict or None): The unit name dictionary to convert.

        Returns:
            str: A string representation of the unit name, or empty string if namedict is
            None.

        Notes:
            This method contains nested helper functions for ordering keys and
            concatenating units.
        """

        def order_keys(namelist):
            """Internal method to order the units sensibly."""

            sorted_ = []

            # Coefficient first
            if '' in namelist:
                sorted_.append('')

            # Distances first
            templist = []
            for key in namelist:
                if key in Unit._NAME_TO_UNIT:
                    expo = Unit._NAME_TO_UNIT[key].exponents
                    if expo[0]:
                        templist.append(key)
            templist.sort()
            sorted_ += templist

            # Angles second
            templist = []
            for key in namelist:
                if key in Unit._NAME_TO_UNIT:
                    expo = Unit._NAME_TO_UNIT[key].exponents
                    if expo[2] and key not in sorted_:
                        templist.append(key)
            templist.sort()
            sorted_ += templist

            # Time units next
            templist = []
            for key in namelist:
                if key in Unit._NAME_TO_UNIT:
                    expo = Unit._NAME_TO_UNIT[key].exponents
                    if expo[1] and key not in sorted_:
                        templist.append(key)
            templist.sort()
            sorted_ += templist

            # Unrecognized units last
            templist = []
            for key in namelist:
                if key not in sorted_:
                    templist.append(key)
            templist.sort()
            sorted_ += templist

            return sorted_

        def cat_units(namelist, negate=False):
            """A string of names and exponents."""

            unitlist = []
            for key in namelist:
                expo = namedict[key]
                if key == '':
                    if expo != 1:
                        unitlist.append(str(expo))
                    continue

                if negate:
                    expo = -expo
                if expo == 1:
                    unitlist.append(key)
                elif expo > 1:
                    unitlist.append(key + '**' + str(expo))
                else:
                    unitlist.append(key + '**(' + str(expo) + ')')

            return '*'.join(unitlist)

        # Return a string immediately
        if isinstance(namedict, str):
            return namedict

        # Make list of numerator and denominator units
        numers = []
        denoms = []
        for key, expo in namedict.items():
            if key == '' or expo > 0:
                numers.append(key)
            elif expo < 0:
                denoms.append(key)

        # Sort the units
        numers = order_keys(numers)
        denoms = order_keys(denoms)

        if numers:
            if denoms:
                return cat_units(numers) + '/' + cat_units(denoms, negate=True)
            else:
                return cat_units(numers)
        else:
            if denoms:
                return cat_units(denoms, negate=False)
            else:
                return ''

    def create_name(self):
        """Create a name for this Unit object based on its exponents.

        Returns:
            str or dict: A name for this Unit object.
        """

        # Return the internal name, if defined
        if self.name is not None:
            return self.name

        return Unit._name_for_tuples(self.exponents, self.triple)

    @staticmethod
    @functools.lru_cache(maxsize=256)
    def _name_for_tuples(exponents, triple):
        """The name to use for an unnamed unit with these exponents and triple.

        The search over combinations of standard units is costly relative to how often a
        given unit recurs, so results are cached. The returned dictionary is shared
        between callers and must not be modified.

        Parameters:
            exponents (tuple[int, int, int]): Three exponents on length, time, and angle.
            triple (tuple[int, int, int]): The (numerator, denominator, pi exponent)
                conversion factor.

        Returns:
            str or dict: The name, either a string or a dictionary of exponents keyed by
            unit name.
        """

        # Return the name from the dictionary, if found
        try:
            name = Unit._TUPLES_TO_UNIT[(exponents, triple)].name
            if name is not None:
                return name
        except KeyError:
            pass

        expo = exponents

        # Search for combinations that might work
        options = [[], [], []]
        for i in range(3):
            target_power = exponents[i]
            if target_power:
                for unit in Unit._UNITS_BY_EXPO[i]:
                    actual_power = unit.exponents[i]
                    p = target_power // actual_power
                    if p * actual_power == target_power:
                        if p > 0:
                            new_triple = (unit.triple[0]**p,
                                          unit.triple[1]**p,
                                          unit.triple[2] * p)
                        else:
                            new_triple = (unit.triple[1]**(-p),     # swapped!
                                          unit.triple[0]**(-p),
                                          unit.triple[2] * p)

                        options[i].append((unit, p, new_triple))
            else:
                options[i].append((Unit._UNITS_BY_EXPO[i][0], 0, (1, 1, 0)))

        # Check every possible combination for the one that yields the correct
        # coefficient
        successes = []
        for _d, d_option in enumerate(options[0]):
            d_unit, d_power, d_triple = d_option
            d_numer, d_denom, d_expo = d_triple

            for _t, t_option in enumerate(options[1]):
                t_unit, t_power, t_triple = t_option
                t_numer, t_denom, t_expo = t_triple

                for _a, a_option in enumerate(options[2]):
                    a_unit, a_power, a_triple = a_option
                    a_numer, a_denom, a_expo = a_triple

                    numer = d_numer * t_numer * a_numer
                    denom = d_denom * t_denom * a_denom
                    expo  = d_expo  + t_expo  + a_expo

                    gcd_value = math.gcd(numer, denom)
                    numer //= gcd_value
                    denom //= gcd_value

                    if (numer, denom, expo) == triple:
                        successes.append({d_unit.name: d_power,
                                          t_unit.name: t_power,
                                          a_unit.name: a_power})

        # Return the success with the fewest keys
        if successes:
            lengths = [len(k) for k in successes]
            best = min(lengths)
            for k, length in enumerate(lengths):  # pragma: no cover
                if length == best:  # pragma: no cover
                    return successes[k]

        # Failing that, use a standard unit and define the coefficient too
        (numer, denom, pi_expo) = triple
        if denom == 1 and pi_expo == 0:
            coefft = numer
        else:
            coefft = numer / denom * np.pi**pi_expo

        new_dict = {''   : coefft,
                    'km' : exponents[0],
                    's'  : exponents[1],
                    'rad': exponents[2]}

        return new_dict

    def get_name(self):
        """Get the name of this Unit object.

        Returns:
            str or dict or None: The name of this Unit object.
        """

        name = self.name or self.create_name()
        return Unit.name_to_str(name)

    def set_name(self, name):
        """Set the name of this Unit object.

        Parameters:
            name (str or dict): The new name for this Unit object.
        """

        self.name = name
        return self

##########################################################################################
# Define the most common units and their names
##########################################################################################

Unit.UNITLESS    = Unit((0, 0, 0), (1, 1, 0), '')

Unit.KM          = Unit((1, 0, 0), (1,          1, 0), 'km')
Unit.KILOMETER   = Unit((1, 0, 0), (1,          1, 0), 'kilometer')
Unit.KILOMETERS  = Unit((1, 0, 0), (1,          1, 0), 'kilometers')
Unit.M           = Unit((1, 0, 0), (1,       1000, 0), 'm')
Unit.METER       = Unit((1, 0, 0), (1,       1000, 0), 'meter')
Unit.METERS      = Unit((1, 0, 0), (1,       1000, 0), 'meters')
Unit.CM          = Unit((1, 0, 0), (1,     100000, 0), 'cm')
Unit.CENTIMETER  = Unit((1, 0, 0), (1,     100000, 0), 'centimeter')
Unit.CENTIMETERS = Unit((1, 0, 0), (1,     100000, 0), 'centimeters')
Unit.MM          = Unit((1, 0, 0), (1,    1000000, 0), 'mm')
Unit.MILLIMETER  = Unit((1, 0, 0), (1,    1000000, 0), 'millimeter')
Unit.MILLIMETERS = Unit((1, 0, 0), (1,    1000000, 0), 'millimeters')
Unit.MICRON      = Unit((1, 0, 0), (1, 1000000000, 0), 'micron')
Unit.MICRONS     = Unit((1, 0, 0), (1, 1000000000, 0), 'microns')

Unit.S           = Unit((0, 1, 0), (    1,    1, 0), 's')
Unit.SEC         = Unit((0, 1, 0), (    1,    1, 0), 'sec')
Unit.SECOND      = Unit((0, 1, 0), (    1,    1, 0), 'second ')
Unit.SECONDS     = Unit((0, 1, 0), (    1,    1, 0), 'seconds')
Unit.MIN         = Unit((0, 1, 0), (   60,    1, 0), 'min')
Unit.MINUTE      = Unit((0, 1, 0), (   60,    1, 0), 'minute')
Unit.MINUTES     = Unit((0, 1, 0), (   60,    1, 0), 'minutes')
Unit.H           = Unit((0, 1, 0), ( 3600,    1, 0), 'h')
Unit.HOUR        = Unit((0, 1, 0), ( 3600,    1, 0), 'hour')
Unit.HOURS       = Unit((0, 1, 0), ( 3600,    1, 0), 'hours')
Unit.D           = Unit((0, 1, 0), (86400,    1, 0), 'd')
Unit.DAY         = Unit((0, 1, 0), (86400,    1, 0), 'day')
Unit.DAYS        = Unit((0, 1, 0), (86400,    1, 0), 'days')
Unit.MS          = Unit((0, 1, 0), (    1, 1000, 0), 'ms')
Unit.MSEC        = Unit((0, 1, 0), (    1, 1000, 0), 'msec')

Unit.RAD         = Unit((0, 0, 1), (1,        1, 0), 'rad')
Unit.RADIAN      = Unit((0, 0, 1), (1,        1, 0), 'radian')
Unit.RADIANS     = Unit((0, 0, 1), (1,        1, 0), 'radians')
Unit.MRAD        = Unit((0, 0, 1), (1,     1000, 0), 'mrad')
Unit.MILLIRAD    = Unit((0, 0, 1), (1,     1000, 0), 'millirad')
Unit.DEG         = Unit((0, 0, 1), (1,      180, 1), 'deg')
Unit.DEGREE      = Unit((0, 0, 1), (1,      180, 1), 'degree')
Unit.DEGREES     = Unit((0, 0, 1), (1,      180, 1), 'degrees')
Unit.ARCHOUR     = Unit((0, 0, 1), (1,       12, 1), 'archour')
Unit.ARCHOURS    = Unit((0, 0, 1), (1,       12, 1), 'archours')
Unit.ARCMIN      = Unit((0, 0, 1), (1,   180*60, 1), 'arcmin')
Unit.ARCMINUTE   = Unit((0, 0, 1), (1,   180*60, 1), 'arcminute')
Unit.ARCMINUTES  = Unit((0, 0, 1), (1,   180*60, 1), 'arcminutes')
Unit.ARCSEC      = Unit((0, 0, 1), (1, 180*3600, 1), 'arcsec')
Unit.ARCSECOND   = Unit((0, 0, 1), (1, 180*3600, 1), 'arcsecond')
Unit.ARCSECONDS  = Unit((0, 0, 1), (1, 180*3600, 1), 'arcseconds')
Unit.REV         = Unit((0, 0, 1), (2,        1, 1), 'rev')
Unit.REVS        = Unit((0, 0, 1), (2,        1, 1), 'revs')
Unit.ROTATION    = Unit((0, 0, 1), (2,        1, 1), 'rotation')
Unit.ROTATIONS   = Unit((0, 0, 1), (2,        1, 1), 'rotations')
Unit.CYCLE       = Unit((0, 0, 1), (2,        1, 1), 'cycle')
Unit.CYCLES      = Unit((0, 0, 1), (2,        1, 1), 'cycles')

Unit.STER        = Unit((0, 0, 2), (1,        1, 0), 'ster')

# Create dictionaries to convert between name and units
Unit._NAME_TO_UNIT = {}
Unit._TUPLES_TO_UNIT = {}

# Assemble a list of all the recognized units
Unit._DISTANCE_LIST = [Unit.KM, Unit.M, Unit.CM, Unit.MM, Unit.MICRON]
Unit._TIME_LIST  = [Unit.S, Unit.D, Unit.H, Unit.MIN, Unit.MSEC]
Unit._ANGLE_LIST = [Unit.RAD, Unit.MRAD, Unit.DEG, Unit.ARCSEC, Unit.ARCMIN, Unit.ARCHOUR,
                    Unit.CYCLES, Unit.STER]

Unit._UNITS_BY_EXPO = [Unit._DISTANCE_LIST,     # index = 0
                       Unit._TIME_LIST,         # index = 1
                       Unit._ANGLE_LIST]        # index = 2

Unit._STANDARD_LIST = ([Unit.UNITLESS] +
                       Unit._DISTANCE_LIST +
                       Unit._TIME_LIST +
                       Unit._ANGLE_LIST)

# Fill in the dictionaries
for _unit in Unit._STANDARD_LIST:
    Unit._NAME_TO_UNIT[_unit.name] = _unit
    Unit._TUPLES_TO_UNIT[(_unit.exponents, _unit.triple)] = _unit

del _unit

##########################################################################################
