=====
Units
=====

A PolyMath object can carry a :class:`~polymath.Unit`. Units make the meaning of a
quantity explicit, catch the addition of a distance to a time, and control how values are
presented, but they do not change how values are stored: the numbers inside every object
are always in the standard units of kilometers, seconds, and radians, or products and
powers of them.

Values Are Always in Standard Units
===================================

Because storage is standardized, the ``unit`` argument of a constructor describes how to
present the values, not how to interpret them. An object constructed from the number 1
with a unit of meters holds one kilometer and displays it as 1000 meters:

.. code-block:: python

    >>> import numpy as np
    >>> from polymath import Scalar, Unit
    >>> d = Scalar([1., 2.], unit=Unit.M)
    >>> d
    Scalar(1000. 2000.; m)
    >>> d.values
    array([1., 2.])
    >>> d.into_unit()
    array([1000., 2000.])

To construct an object from values expressed in some unit, convert them to standard units
first. :meth:`~polymath.Unit.from_this` converts a number from a unit into standard units,
and :meth:`~polymath.Unit.into_this` converts the other way.

.. code-block:: python

    >>> angle = Scalar(Unit.DEG.from_this(90.), unit=Unit.DEG)
    >>> angle
    Scalar(90.0; deg)
    >>> angle.sin()
    Scalar(1.0)
    >>> Unit.DEG.into_this(np.pi)
    180.0

Available Units
===============

The :class:`~polymath.Unit` class defines constants for the common units.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Dimension
     - Constants
   * - Distance
     - :attr:`~polymath.Unit.KM`, :attr:`~polymath.Unit.M`, :attr:`~polymath.Unit.CM`,
       :attr:`~polymath.Unit.MM`, :attr:`~polymath.Unit.MICRON`, and their spelled-out
       forms such as :attr:`~polymath.Unit.KILOMETERS`.
   * - Time
     - :attr:`~polymath.Unit.S`, :attr:`~polymath.Unit.MS`, :attr:`~polymath.Unit.MIN`,
       :attr:`~polymath.Unit.H`, :attr:`~polymath.Unit.D`, and their spelled-out forms
       such as :attr:`~polymath.Unit.SECONDS`.
   * - Angle
     - :attr:`~polymath.Unit.RAD`, :attr:`~polymath.Unit.MRAD`, :attr:`~polymath.Unit.DEG`,
       :attr:`~polymath.Unit.ARCHOUR`, :attr:`~polymath.Unit.ARCMIN`,
       :attr:`~polymath.Unit.ARCSEC`, :attr:`~polymath.Unit.REV`,
       :attr:`~polymath.Unit.CYCLE`, and their spelled-out forms such as
       :attr:`~polymath.Unit.DEGREES`.
   * - Solid angle
     - :attr:`~polymath.Unit.STER`.
   * - None
     - :attr:`~polymath.Unit.UNITLESS`, which is equivalent to a unit of None.

Units multiply, divide, and raise to powers to form compound units, and
:meth:`~polymath.Unit.as_unit` looks up a unit by its standard name.

.. code-block:: python

    >>> Unit.KM / Unit.S
    Unit(km/s)
    >>> (Unit.KM / Unit.S) ** 2
    Unit(km**2/s**2)
    >>> Unit.as_unit('deg')
    Unit(deg)

A unit is defined by its :attr:`~polymath.Unit.exponents` on distance, time, and angle and
by a :attr:`~polymath.Unit.triple` of integers giving the exact conversion factor into
standard units as a numerator, a denominator, and a power of pi. A degree has exponents
``(0, 0, 1)`` and triple ``(1, 180, 1)``, meaning that the factor is pi/180. The
constructor takes the same two tuples and an optional name, so a unit that is not
predefined can be built.

.. code-block:: python

    >>> Unit.DEG.exponents, Unit.DEG.triple
    ((0, 0, 1), (1, 180, 1))

Units in Arithmetic
===================

Arithmetic propagates units. Multiplication and division combine them, addition and
subtraction require compatible ones, and the result of adding a distance in meters to one
in kilometers takes the unit of the left operand.

.. code-block:: python

    >>> Scalar(2., unit=Unit.KM) * Scalar(3., unit=Unit.S)
    Scalar(6.0; km*s)
    >>> Scalar(1., unit=Unit.KM) + Scalar(1., unit=Unit.M)
    Scalar(2.0; km)
    >>> Scalar(1., unit=Unit.KM) + Scalar(1., unit=Unit.S)
    Traceback (most recent call last):
    ...
    ValueError: Scalar "+" units are not compatible: km, s

The trigonometric functions require an angle or a unitless value:

.. code-block:: python

    >>> Scalar(1., unit=Unit.KM).sin()
    Traceback (most recent call last):
    ...
    ValueError: Scalar.sin() unit is not compatible with an angle: km

Changing the Unit
=================

:attr:`~polymath.Qube.unit_`, or its synonym :attr:`~polymath.Qube.units`, returns the
unit, or None for a unitless object. :meth:`~polymath.Qube.set_unit` changes it in place,
which requires a writable object and a unit compatible with the existing one;
:meth:`~polymath.Qube.without_unit` returns a copy with no unit;
:meth:`~polymath.Qube.confirm_unit` raises an error unless the object has the given unit;
and :meth:`~polymath.Qube.is_unitless` reports whether there is one.

.. code-block:: python

    >>> d = Scalar([1., 2.], unit=Unit.KM)
    >>> d.unit_
    Unit(km)
    >>> d.set_unit(Unit.M)
    >>> d
    Scalar(1000. 2000.; m)
    >>> d.without_unit()
    Scalar(1. 2.)

:class:`~polymath.Boolean`, :class:`~polymath.Matrix3`, and :class:`~polymath.Quaternion`
never carry a unit, because a truth value or a rotation is dimensionless.
