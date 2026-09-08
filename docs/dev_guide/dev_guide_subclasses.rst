==============
The Subclasses
==============

Overview
========

Each subclass lives in its own module under ``src/polymath/``, and each follows the same
pattern: the class constants, a static conversion method named after the class, the
methods that give the class its meaning, and at the bottom of the module the read-only
class constants and the registration of the class on :class:`~polymath.Qube`. This
chapter describes what each class fixes and what it overrides, so that a change lands at
the right level of the hierarchy. The base class itself is covered in
:doc:`dev_guide_architecture`.

The Module Pattern
==================

**Class constants.** Every subclass module sets all seven of the constraint constants
described in :doc:`dev_guide_architecture`, one per line with a trailing comment, even
where a value merely repeats the parent's. The table gives the values in use. A dash means
the constant is inherited.

.. list-table::
   :header-rows: 1
   :widths: 16 9 10 9 9 9 9 9 20

   * - Class
     - Rank
     - Numerator
     - Floats
     - Ints
     - Bools
     - Units
     - Derivs
     - Default value
   * - :class:`~polymath.Scalar`
     - 0
     - ``()``
     - yes
     - yes
     - no
     - yes
     - yes
     - 1
   * - :class:`~polymath.Boolean`
     - 0
     - ``()``
     - no
     - no
     - yes
     - no
     - no
     - False
   * - :class:`~polymath.Vector`
     - 1
     - free
     - yes
     - yes
     - no
     - yes
     - yes
     - ones
   * - :class:`~polymath.Pair`
     - 1
     - ``(2,)``
     - yes
     - yes
     - no
     - yes
     - yes
     - ``[1, 1]``
   * - :class:`~polymath.Vector3`
     - 1
     - ``(3,)``
     - yes
     - no
     - no
     - yes
     - yes
     - ``[1, 1, 1]``
   * - :class:`~polymath.Quaternion`
     - 1
     - ``(4,)``
     - yes
     - no
     - no
     - no
     - yes
     - ``[1, 0, 0, 0]``
   * - :class:`~polymath.Polynomial`
     - 1
     - free
     - yes
     - no
     - --
     - --
     - --
     - ones
   * - :class:`~polymath.Matrix`
     - 2
     - free
     - yes
     - no
     - no
     - yes
     - yes
     - ones
   * - :class:`~polymath.Matrix3`
     - 2
     - ``(3, 3)``
     - yes
     - no
     - no
     - no
     - yes
     - identity

**The converter.** Each class has a static method, such as
:meth:`~polymath.Pair.as_pair`, that converts an arbitrary argument to the class. It
returns an argument of the right class unchanged (or without its derivatives when
``recursive`` is False), reinterprets the item axes of any other :class:`~polymath.Qube`
by flattening a 1xN or Nx1 numerator or by splitting surplus numerator axes into the
denominator, and passes anything else to the constructor. The converters with a fixed
item shape treat a single number as that number repeated. These methods are the
preferred way for an operation to accept either a PolyMath object or a plain value, and
they are cheap when no conversion is needed.

**The constants.** Each module ends by building the read-only constants of the class,
such as :attr:`~polymath.Scalar.ZERO`, :attr:`~polymath.Vector3.XAXIS`, and
:attr:`~polymath.Matrix3.IDENTITY`. They are built after the class body because
constructing them calls methods that the extension binding supplies, and they are made
read-only because they are shared by every caller. Every class has a constant holding a
single masked value.

**The registration.** The final statement of each module assigns the class to an
attribute of :class:`~polymath.Qube`, such as ``Qube._PAIR_CLASS``, so that the extension
modules can reach it without importing it.

Scalar and Boolean
==================

:class:`~polymath.Scalar` holds a single number per item and is the only class that
defines the ordering comparisons, which return a :class:`~polymath.Boolean`. It overrides
:meth:`~polymath.Scalar.__pow__` with lookup tables for the common integer and
half-integer exponents, so that a square or a square root does not go through the general
power function, and it overrides :meth:`~polymath.Scalar.reciprocal`,
:meth:`~polymath.Scalar.identity`, and :meth:`~polymath.Scalar.abs`. Its own methods are
the transcendental functions, the reductions, the index conversions, and the quadratic
solver. The functions that can fail for some inputs, such as
:meth:`~polymath.Scalar.sqrt` and :meth:`~polymath.Scalar.arcsin`, take a ``check``
keyword that decides whether to mask the offending elements or to let NumPy raise.

:class:`~polymath.Boolean` is a :class:`~polymath.Scalar` restricted to truth values, with
units and derivatives disallowed. It overrides every arithmetic operator to convert itself
to an integer :class:`~polymath.Scalar` first, so that the sum of two truth values is a
count rather than a truth value, and it overrides :meth:`~polymath.Boolean.sum` to count
the True elements. :meth:`~polymath.Boolean.as_index` returns a NumPy boolean array for
indexing. The extension modules reach the class as ``Qube._BOOLEAN_CLASS``, which is how
the comparison operators build their results.

Vector and Its Subclasses
=========================

:class:`~polymath.Vector` fixes the numerator rank at one and leaves the length free. Its
constructor accepts a single Python number, and its methods are the vector algebra:
:meth:`~polymath.Vector.dot`, :meth:`~polymath.Vector.cross`,
:meth:`~polymath.Vector.norm`, :meth:`~polymath.Vector.unit`, and the rest. Most of these
are thin wrappers around the general functions in the vector operations extension module,
which operate on any object with suitable item axes; the wrappers fix the axes and the
result class. The conversions to and from :class:`~polymath.Scalar` components and to
:class:`~polymath.Matrix` rows, columns, and diagonals are here as well.

:class:`~polymath.Pair` fixes the length at two. It adds the two-dimensional operations
and keeps integers allowed, because a :class:`~polymath.Pair` of integers is used as an
index into two axes. :class:`~polymath.Vector3` fixes the length at three, allows floats
only, and adds the conversions to and from spherical and cylindrical coordinates,
rotation about an axis, and the angular offsets. Both add a
:meth:`~polymath.Vector3.from_scalars` that assembles the vector from named components.

:class:`~polymath.Quaternion` fixes the length at four and disallows units. It overrides
:meth:`~polymath.Quaternion.__mul__` and :meth:`~polymath.Quaternion.__truediv__` as the
quaternion product and its inverse, along with :meth:`~polymath.Quaternion.reciprocal`
and :meth:`~polymath.Quaternion.identity`, and it adds the conversions to and from a
:class:`~polymath.Matrix3`, an angle and axis, a scalar and vector part, and Euler
angles. The Euler conventions are shared with :class:`~polymath.Matrix3`.

:class:`~polymath.Polynomial` leaves the length free but reinterprets the components as
coefficients in decreasing order of power. It is the one subclass that overrides
:meth:`~polymath.Polynomial.__init__`: a lone :class:`~polymath.Vector` argument is
converted by copying its attributes directly, and every derivative is converted to a
:class:`~polymath.Polynomial`, so that the derivatives of a polynomial are always
polynomials. It overrides the arithmetic operators as polynomial arithmetic and the
equality operators to compare polynomials of different orders, and it adds evaluation,
differentiation, root finding, and the :attr:`~polymath.Polynomial.order` property. It
is not registered on :class:`~polymath.Qube`, because no extension function needs it.

Matrix and Matrix3
==================

:class:`~polymath.Matrix` fixes the numerator rank at two, allows floats only, and adds
the matrix algebra: :meth:`~polymath.Matrix.transpose`, :meth:`~polymath.Matrix.inverse`,
:meth:`~polymath.Matrix.solve`, :meth:`~polymath.Matrix.unitary`, and
:meth:`~polymath.Matrix.is_diagonal`, along with the row and column extractions. It
overrides :meth:`~polymath.Matrix.__abs__`, :meth:`~polymath.Matrix.__floordiv__`, and
:meth:`~polymath.Matrix.__mod__` to raise, because those operations have no meaning for a
matrix, and :meth:`~polymath.Matrix.identity` and :meth:`~polymath.Matrix.reciprocal`
to mean the identity matrix and the inverse.

:class:`~polymath.Matrix3` fixes the shape at 3x3, disallows units, and represents
rotations. It names :class:`~polymath.Matrix` as its derivative class, because the
derivative of a rotation matrix is a general matrix. It overrides the additive operators
and negation to raise, since the sum of two rotations is not a rotation, and overrides
multiplication so that a product with a :class:`~polymath.Vector3` rotates the vector
and a product with another :class:`~polymath.Matrix3` composes the rotations. It adds the
constructors for rotations about an axis, from a pole, from two vectors, and from Euler
angles, and their inverses. It also overrides :meth:`~polymath.Matrix3.__getstate__` and
:meth:`~polymath.Matrix3.__setstate__` to pickle a rotation matrix as a
:class:`~polymath.Quaternion` when it can, which stores four numbers instead of nine.

Unit
====

:class:`~polymath.Unit` is not a :class:`~polymath.Qube`. It holds three integer
exponents, a triple of integers defining an exact conversion factor, and an optional
name, and it implements the arithmetic that combines units and the conversions of values
into and out of a unit. Names are parsed and generated by
:meth:`~polymath.Unit.name_to_dict` and :meth:`~polymath.Unit.name_to_str`, and a
registry keyed by name serves :meth:`~polymath.Unit.as_unit`, which accepts only the
standard names; a compound unit such as kilometers per second is built with the
operators. The class constants for the common units are built at the bottom of the
module. Equality compares the exponents and the factor, not the name, so two units that
convert the same way are equal.

API Reference
=============

The public methods of every class are in :doc:`/module`. The modules, including their
private helpers and class constants, are in :doc:`dev_guide_internal_api`.
