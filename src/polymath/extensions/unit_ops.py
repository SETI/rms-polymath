##########################################################################################
# polymath/extensions/unit_ops.py: Unit operations
##########################################################################################

from polymath.qube import Qube
from polymath.unit import Unit

__all__ = ['confirm_unit', 'into_unit', 'is_unitless', 'set_unit', 'without_unit']


def set_unit(self, unit, *, override=False):
    """Set the unit of this object.

    Parameters:
        unit (Unit or None): The new unit.
        override (bool, optional): If True, the unit can be modified on a read-only
            object.

    Raises:
        ValueError: If this object is read-only and `override` is False.
    """

    if not self._UNITS_OK:
        if Unit.is_unitless(unit):
            return
        raise TypeError(f'units are disallowed in class {type(self).__name__}')

    if not override:
        self.require_writeable()

    unit = Unit.as_unit(unit)

    Unit.require_compatible(unit, self._unit)
    self._unit = unit
    self._cache.clear()


def without_unit(self, *, recursive=True):
    """A shallow copy of this object without units.

    A read-only object remains read-only. If recursive is True, derivatives are also
    stripped of their units.

    Parameters:
        recursive (bool, optional): True to include derivatives with their units
            stripped; False to omit all derivatives.

    Returns:
        Qube: A shallow copy of this object with the unit stripped.
    """

    if self._unit is None and not self._derivs:
        return self

    obj = self.clone(recursive=recursive)
    obj._unit = None

    # Strip units from derivatives if recursive is True
    if recursive and obj._derivs:
        for key, deriv in obj._derivs.items():
            if deriv._unit is not None:
                obj._derivs[key] = deriv.without_unit(recursive=True)

    return obj


def into_unit(self, *, recursive=False):
    """The values property of this object, converted to its unit.

    This method converts values from standard units (kilometers, seconds, radians)
    to this object's specified unit. For example, if the object has unit=Unit.M
    (meters) and the internal values are in kilometers (standard units), this
    method converts from km to m by multiplying by 1000.

    Parameters:
        recursive (bool, optional): If True, also return the derivatives converted to
            their units.

    Returns:
        (numpy.ndarray, float, int, bool, or tuple): The values attribute of this
        object, converted from standard units to this object's unit. If `recursive`
        is True, it returns a tuple (`values`, `derivs`), where `derivs` is a
        dictionary of the derivative values converted to their units.

    Examples:
        >>> a = Scalar([1.0, 2.0, 3.0], unit=Unit.M)  # values in km (standard)
        >>> a.into_unit()  # Returns [1000.0, 2000.0, 3000.0] (converted to meters)
    """

    if self._unit is None or self._unit.into_unit_factor == 1.:
        values = self._values
    else:
        values = Unit.into_unit(self._unit, self._values)

    if not recursive:
        return values

    derivs = {}
    for key, deriv in self._derivs.items():
        derivs[key] = Unit.into_unit(deriv._unit, deriv._values)

    return (values, derivs)


def confirm_unit(self, unit):
    """Raises a ValueError if the unit is not compatible with this object.

    Parameters:
        unit (Unit or None): The new unit.

    Returns:
        Qube: This object.

    Raises:
        ValueError: If this object has a unit that are incompatible with the new unit.
    """

    if not Unit.can_match(self._unit, unit):
        raise ValueError(f'units are not compatible with {type(self).__name__} '
                         f'object: {unit}, {self._unit}')

    return self


def is_unitless(self):
    """True if this object is unitless."""

    return Unit.is_unitless(self._unit)


def _require_unitless(self, op=''):
    """Raise a ValueError if this object is not unitless.

    Parameters:
        info (str, optional): An info string to embed into the error message.

    Raises:
        ValueError: If units are present.
    """

    if self.is_unitless():
        return

    Unit.require_unitless(self._unit, info=self._opstr(op))


def _require_angle(self, op=''):
    """Raise a ValueError if this object is not either unitless or has a dimension of
    angle.

    Parameters:
        op (str, optional): Operation name to embed into the error message.

    Raises:
        ValueError: If units are not compatible with an angle.
    """

    if Unit.is_angle(self._unit):
        return

    Unit.require_angle(self._unit, info=self._opstr(op))


def _require_compatible_units(self, arg, op=''):
    """Raise a ValueError if these objects do not have compatible units.

    Parameters:
        op (str, optional): Operation name to embed into the error message.

    Raises:
        ValueError: If units are not compatible.
    """

    if not isinstance(arg, Qube):
        return True

    if Unit.can_match(self._unit, arg._unit):
        return True

    Unit.require_compatible(self._unit, arg._unit, info=self._opstr(op))

##########################################################################################
