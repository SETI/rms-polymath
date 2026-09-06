=============================
Arithmetic and Math Functions
=============================

PolyMath objects support the ordinary Python operators, and each class adds the functions
that make sense for what it represents. Every operation broadcasts over the shape, masks
any element it cannot compute, checks that units are compatible, and propagates
derivatives. This chapter describes the operations themselves; the chapters that follow
describe what happens to masks, derivatives, and units along the way.

Operators
=========

The arithmetic operators ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, and ``**`` are defined
along with their in-place forms, and the operands can be PolyMath objects, Python numbers,
NumPy arrays, or nested sequences in any combination.

.. code-block:: python

    >>> import numpy as np
    >>> from polymath import Boolean, Matrix, Matrix3, Pair, Scalar, Vector, Vector3
    >>> x = Scalar([1., 4., 9.])
    >>> x + 1
    Scalar( 2.  5. 10.)
    >>> x * 2
    Scalar( 2.  8. 18.)
    >>> x ** 0.5
    Scalar(1. 2. 3.)
    >>> x // 2
    Scalar(0. 2. 4.)
    >>> x + np.array([10., 20., 30.])
    Scalar(11. 24. 39.)
    >>> np.array([10., 20., 30.]) + x
    Scalar(11. 24. 39.)

What ``*`` means depends on the classes involved. A :class:`~polymath.Scalar` scales
anything. A :class:`~polymath.Matrix` times a :class:`~polymath.Vector` is a matrix-vector
product, a :class:`~polymath.Matrix` times a :class:`~polymath.Matrix` is a matrix product,
and a :class:`~polymath.Matrix3` times a :class:`~polymath.Vector3` rotates the vector. Two
vectors cannot be multiplied with ``*``, because the product would be ambiguous; use
:meth:`~polymath.Vector.dot`, :meth:`~polymath.Vector.cross`, or
:meth:`~polymath.Vector.element_mul` instead. Vectors add and subtract as usual.

.. code-block:: python

    >>> Vector3.XAXIS * 2
    Vector3(2. 0. 0.)
    >>> Vector3.XAXIS + Vector3.YAXIS
    Vector3(1. 1. 0.)
    >>> Matrix([[2., 0.], [0., 4.]]) * Vector([1., 1.])
    Vector(2. 4.)
    >>> Vector3([1., 2., 3.]).element_mul(Vector3([2., 2., 2.]))
    Vector3(2. 4. 6.)

The comparison operators ``==`` and ``!=`` work for every class and return a
:class:`~polymath.Boolean`. The ordering operators ``<``, ``<=``, ``>``, and ``>=`` are
defined for :class:`~polymath.Scalar` and :class:`~polymath.Boolean` only. When both
operands are single values, a comparison returns a Python ``bool`` instead.

.. code-block:: python

    >>> x > 3
    Boolean(False  True  True)
    >>> x == 4
    Boolean(False  True False)
    >>> Vector3.XAXIS == Vector3.XAXIS
    True

:class:`~polymath.Boolean` objects combine with ``&``, ``|``, ``^``, and ``~``, and
:meth:`~polymath.Qube.any` and :meth:`~polymath.Qube.all` reduce them.

.. code-block:: python

    >>> Boolean([True, False]) & Boolean([True, True])
    Boolean( True False)
    >>> ~Boolean([True, False])
    Boolean(False  True)
    >>> (x > 0).all()
    Boolean(True)

:func:`abs` and :func:`len` work as expected, with :func:`len` counting along the first
axis of the shape.

.. code-block:: python

    >>> abs(Scalar([-1., 2.]))
    Scalar(1. 2.)
    >>> len(x)
    3

Scalar Functions
================

:class:`~polymath.Scalar` provides the common math functions as methods. Each takes a
``recursive`` keyword, described in :doc:`user_guide_derivatives`, and those that can fail
for some inputs mask the elements where they do.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Methods
     - Purpose
   * - :meth:`~polymath.Scalar.sin`, :meth:`~polymath.Scalar.cos`,
       :meth:`~polymath.Scalar.tan`, :meth:`~polymath.Scalar.arcsin`,
       :meth:`~polymath.Scalar.arccos`, :meth:`~polymath.Scalar.arctan`,
       :meth:`~polymath.Scalar.arctan2`
     - Trigonometry, in radians. The inverse functions mask inputs outside their domain.
   * - :meth:`~polymath.Scalar.sqrt`, :meth:`~polymath.Scalar.log`,
       :meth:`~polymath.Scalar.exp`
     - Square root, natural logarithm, and exponential. The first two mask the inputs for
       which they are undefined.
   * - :meth:`~polymath.Scalar.abs`, :meth:`~polymath.Scalar.sign`,
       :meth:`~polymath.Scalar.int`, :meth:`~polymath.Scalar.frac`
     - Absolute value, sign, integer part rounded toward negative infinity, and
       fractional part.
   * - :meth:`~polymath.Scalar.reciprocal`
     - One over the value, masking zeros.
   * - :meth:`~polymath.Scalar.max`, :meth:`~polymath.Scalar.min`,
       :meth:`~polymath.Scalar.argmax`, :meth:`~polymath.Scalar.argmin`,
       :meth:`~polymath.Scalar.median`, :meth:`~polymath.Scalar.sort`
     - Reductions over one axis or over the whole shape, ignoring masked elements.
   * - :meth:`~polymath.Scalar.maximum`, :meth:`~polymath.Scalar.minimum`
     - Element-by-element maximum and minimum of several objects.
   * - :meth:`~polymath.Scalar.solve_quadratic`, :meth:`~polymath.Scalar.eval_quadratic`
     - Roots and values of a quadratic given its three coefficients.

.. code-block:: python

    >>> x.sqrt()
    Scalar(1. 2. 3.)
    >>> Scalar([1.5, -2.5]).int()
    Scalar( 1 -3)
    >>> Scalar([1.5, -2.5]).frac()
    Scalar(0.5 0.5)
    >>> Scalar([3., 1., 2.]).max()
    Scalar(3.0)
    >>> Scalar([3., 1., 2.]).argmin()
    Scalar(1)
    >>> Scalar.maximum(Scalar([1., 5.]), Scalar([3., 2.]))
    Scalar(3. 5.)
    >>> Scalar.solve_quadratic(1., -3., 2.)
    (Scalar(1.0), Scalar(2.0))

:meth:`~polymath.Qube.sum` and :meth:`~polymath.Qube.mean` are available on every class
and accept an ``axis``:

.. code-block:: python

    >>> Scalar([[1., 2.], [3., 4.]]).sum(axis=0)
    Scalar(4. 6.)
    >>> Scalar([1., 2., 3.]).mean()
    Scalar(2.0)

Vector Functions
================

:class:`~polymath.Vector` and its subclasses provide the vector algebra.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Methods
     - Purpose
   * - :meth:`~polymath.Vector.dot`, :meth:`~polymath.Vector.cross`,
       :meth:`~polymath.Vector.ucross`, :meth:`~polymath.Vector.outer`
     - Dot product, cross product, unit cross product, and outer product, the last of
       which is a :class:`~polymath.Matrix`.
   * - :meth:`~polymath.Vector.norm`, :meth:`~polymath.Vector.norm_sq`,
       :meth:`~polymath.Vector.unit`, :meth:`~polymath.Vector.with_norm`
     - Length, squared length, the unit vector in the same direction, and the vector
       scaled to a given length.
   * - :meth:`~polymath.Vector.sep`, :meth:`~polymath.Vector.perp`,
       :meth:`~polymath.Vector.proj`
     - The angle between two vectors, and the components of one vector perpendicular to
       and projected onto another.
   * - :meth:`~polymath.Vector.element_mul`, :meth:`~polymath.Vector.element_div`
     - Element-by-element product and quotient.

.. code-block:: python

    >>> Vector3([3., 4., 0.]).norm()
    Scalar(5.0)
    >>> Vector3([3., 4., 0.]).unit()
    Vector3(0.6 0.8 0. )
    >>> Vector3.XAXIS.dot(Vector3([1., 1., 0.]))
    Scalar(1.0)
    >>> Vector3.XAXIS.cross(Vector3.YAXIS)
    Vector3(0. 0. 1.)
    >>> Vector3.XAXIS.sep(Vector3.YAXIS)
    Scalar(1.5707963267948968)
    >>> Vector3([1., 1., 0.]).perp(Vector3.XAXIS)
    Vector3(0. 1. 0.)

:class:`~polymath.Vector3` adds conversions between Cartesian coordinates and angles:
:meth:`~polymath.Vector3.from_ra_dec_length` and
:meth:`~polymath.Vector3.to_ra_dec_length` for right ascension and declination,
:meth:`~polymath.Vector3.from_cylindrical` and :meth:`~polymath.Vector3.to_cylindrical`
for cylindrical coordinates, and :meth:`~polymath.Vector3.longitude` and
:meth:`~polymath.Vector3.latitude`. :meth:`~polymath.Vector3.spin` rotates a vector about
an axis by an angle, and :meth:`~polymath.Vector3.offset_angles` gives the angular offsets
of one vector from another.

.. code-block:: python

    >>> Vector3([1., 1., 0.]).longitude()
    Scalar(0.7853981633974483)
    >>> Vector3([1., 1., 0.]).to_ra_dec_length()
    (Scalar(0.7853981633974483), Scalar(0.0), Scalar(1.4142135623730951))

:class:`~polymath.Pair` adds :meth:`~polymath.Pair.swapxy`, :meth:`~polymath.Pair.rot90`,
:meth:`~polymath.Pair.angle`, and :meth:`~polymath.Pair.clip2d`:

.. code-block:: python

    >>> Pair([1., 2.]).swapxy()
    Pair(2. 1.)
    >>> Pair([1., 0.]).rot90()
    Pair( 0. -1.)
    >>> Pair([1., 1.]).angle()
    Scalar(0.7853981633974483)

Matrices and Rotations
======================

:class:`~polymath.Matrix` provides :meth:`~polymath.Matrix.transpose` (also available as
the :attr:`~polymath.Matrix.T` property), :meth:`~polymath.Matrix.inverse`,
:meth:`~polymath.Matrix.solve` for linear systems, :meth:`~polymath.Matrix.unitary` for
the nearest orthonormal matrix, :meth:`~polymath.Matrix.is_diagonal`, and
:meth:`~polymath.Matrix.identity`.

.. code-block:: python

    >>> m = Matrix([[2., 0.], [0., 4.]])
    >>> m.inverse()
    Matrix([0.5  0.  ]
     [0.   0.25])
    >>> m.solve(Vector([2., 4.]))
    Vector(1. 1.)

:class:`~polymath.Matrix3` represents rotations. :meth:`~polymath.Matrix3.x_rotation`,
:meth:`~polymath.Matrix3.y_rotation`, :meth:`~polymath.Matrix3.z_rotation`, and
:meth:`~polymath.Matrix3.axis_rotation` build a rotation about one axis by an angle in
radians; :meth:`~polymath.Matrix3.pole_rotation` builds one from the right ascension and
declination of a pole; :meth:`~polymath.Matrix3.from_euler` builds one from three Euler
angles, which :meth:`~polymath.Matrix3.to_euler` recovers; and
:meth:`~polymath.Matrix3.twovec` builds the rotation that aligns two given vectors with
two axes. Apply a rotation with ``*`` or :meth:`~polymath.Matrix3.rotate`, and apply its
inverse with :meth:`~polymath.Matrix3.unrotate`.

.. code-block:: python

    >>> r = Matrix3.z_rotation(np.pi / 2)
    >>> r * Vector3.XAXIS
    Vector3(6.123234e-17 1.000000e+00 0.000000e+00)
    >>> r.unrotate(Vector3.YAXIS)
    Vector3(1.000000e+00 6.123234e-17 0.000000e+00)
    >>> Matrix3.from_euler(0.1, 0.2, 0.3).to_euler()
    (Scalar(0.10000000000000002), Scalar(0.2), Scalar(0.29999999999999993))

:class:`~polymath.Quaternion` represents the same rotations in four components.
:meth:`~polymath.Quaternion.from_rotation` builds one from an angle and an axis, and
:meth:`~polymath.Quaternion.to_rotation` reverses it;
:meth:`~polymath.Quaternion.to_matrix3` and :meth:`~polymath.Quaternion.from_matrix3`
convert to and from a :class:`~polymath.Matrix3`; :meth:`~polymath.Quaternion.conj` is the
conjugate; and ``*`` is the quaternion product.

.. code-block:: python

    >>> from polymath import Quaternion
    >>> q = Quaternion.from_rotation(np.pi / 2, Vector3.ZAXIS)
    >>> q
    Quaternion(0.70710678 0.         0.         0.70710678)
    >>> q.to_matrix3()
    Matrix3([ 0. -1.  0.]
     [ 1.  0.  0.]
     [ 0.  0.  1.])
    >>> q * q.conj()
    Quaternion(1. 0. 0. 0.)

Polynomials
===========

A :class:`~polymath.Polynomial` holds coefficients in order of decreasing power.
:meth:`~polymath.Polynomial.eval` evaluates it, :meth:`~polymath.Polynomial.deriv`
differentiates it, and :meth:`~polymath.Polynomial.roots` finds its roots; the arithmetic
operators combine polynomials as polynomials.

.. code-block:: python

    >>> from polymath import Polynomial
    >>> p = Polynomial([1., -3., 2.])
    >>> p.eval(Scalar([0., 1., 2.]))
    Scalar(2. 0. 0.)
    >>> p.roots()
    Scalar(1. 2.)
    >>> p.deriv()
    Polynomial( 2. -3.)

Results as Python Numbers
=========================

A result with an empty shape is still a PolyMath object, so that its mask, unit, and
derivatives are preserved. :meth:`~polymath.Qube.as_builtin` converts such an object to a
Python ``float``, ``int``, or ``bool`` when that loses nothing, many methods take a
``builtins`` keyword to make the same decision for one call, and the global setting
:meth:`~polymath.Qube.prefer_builtins` makes the reductions and comparisons return builtins
by default.

.. code-block:: python

    >>> Scalar([1., 2.]).sum()
    Scalar(3.0)
    >>> Scalar([1., 2.]).sum().as_builtin()
    3.0
    >>> Scalar([1., 2.]).sum(builtins=True)
    3.0
