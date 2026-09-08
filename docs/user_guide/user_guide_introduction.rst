=============================
Introduction and Installation
=============================

Purpose
=======

PolyMath is a wrapper around NumPy for geometry calculations. It defines classes for the
quantities that such calculations use, such as scalars, 3-vectors, rotation matrices, and
quaternions, and it lets every one of them stand for an arbitrary array of such quantities
at once. A single :class:`~polymath.Vector3` can hold one vector or a million of them, and
the code that operates on it is written the same way in either case.

The package was written for the OOPS library of the PDS Ring-Moon Systems Node, where it
describes the geometry of planetary images: the line of sight through every pixel, the time
each photon arrived, and the rotation between the camera and the sky. Any calculation that
combines arrays of vectors and matrices, tracks which elements are undefined, or needs
derivatives carried through a chain of operations can use it in the same way.

PolyMath adds four things to a NumPy array:

* **Shape separate from item.** Each object distinguishes the axes that index its items
  from the axes that make up each item. A 2x2 array of 3x3 matrices has a
  :attr:`~polymath.Qube.shape` of ``(2, 2)`` and an :attr:`~polymath.Qube.item` of
  ``(3, 3)``. Broadcasting applies to the shape only, so a :class:`~polymath.Scalar`
  multiplies a :class:`~polymath.Vector3` without any reshaping or ``np.newaxis``.
* **Masks.** Every object carries a boolean mask marking the elements whose value is
  undefined. Operations that would raise an error, such as dividing by zero or taking the
  square root of a negative number, mask the result instead of failing.
* **Units.** An object can carry a :class:`~polymath.Unit`, which records what the values
  measure and controls how they are presented.
* **Derivatives.** An object can carry named derivatives, which every arithmetic operation
  and math function propagates automatically.

Overview
========

A typical use of the package has four stages.

.. mermaid::

   flowchart LR
       A["Construct<br>numbers, sequences,<br>NumPy arrays"] --> B["Compute<br>operators and methods,<br>broadcast over the shape"]
       B --> C["Inspect<br>values, mask, derivs,<br>indexing, iteration"]
       C --> D["Store<br>pickle"]

1. **Construct** objects from Python numbers, nested sequences, or NumPy arrays,
   optionally attaching a mask, a unit, or derivatives. :doc:`user_guide_objects`
   describes this stage.
2. **Compute** with the ordinary arithmetic operators and the methods of each class.
   Masks, units, and derivatives travel with the results. :doc:`user_guide_math`,
   :doc:`user_guide_masks`, :doc:`user_guide_derivatives`, and :doc:`user_guide_units`
   describe this stage.
3. **Inspect** the results through the :attr:`~polymath.Qube.values`,
   :attr:`~polymath.Qube.mask`, and :attr:`~polymath.Qube.derivs` properties, by indexing,
   or by iterating. :doc:`user_guide_indexing` describes this stage.
4. **Store** objects with the standard :mod:`pickle` module, which PolyMath extends with
   compression. :doc:`user_guide_pickling` describes this stage.

The Classes
===========

Every class derives from :class:`~polymath.Qube`, and the methods of
:class:`~polymath.Qube` are available on every object. What distinguishes the subclasses
is the shape of one item and the operations that make sense for it.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Class
     - Item shape
     - Represents
   * - :class:`~polymath.Scalar`
     - ``()``
     - A number, either integer or floating-point.
   * - :class:`~polymath.Boolean`
     - ``()``
     - A True or False value. A subclass of :class:`~polymath.Scalar`.
   * - :class:`~polymath.Vector`
     - ``(n,)``
     - A vector of any length.
   * - :class:`~polymath.Pair`
     - ``(2,)``
     - A coordinate pair or 2-vector. A subclass of :class:`~polymath.Vector`.
   * - :class:`~polymath.Vector3`
     - ``(3,)``
     - A 3-vector. A subclass of :class:`~polymath.Vector`.
   * - :class:`~polymath.Quaternion`
     - ``(4,)``
     - A quaternion, usable as a rotation. A subclass of :class:`~polymath.Vector`.
   * - :class:`~polymath.Polynomial`
     - ``(n,)``
     - The coefficients of a polynomial in one variable, highest power first. A subclass
       of :class:`~polymath.Vector`.
   * - :class:`~polymath.Matrix`
     - ``(m, n)``
     - A matrix of any size.
   * - :class:`~polymath.Matrix3`
     - ``(3, 3)``
     - A 3x3 rotation matrix. A subclass of :class:`~polymath.Matrix`.
   * - :class:`~polymath.Qube`
     - any
     - The base class of all of the above. It is rarely constructed directly.

The :class:`~polymath.Unit` class is not a :class:`~polymath.Qube`; it describes the unit
that any of the above can carry. See :doc:`user_guide_units`.

Installation
============

PolyMath requires Python 3.11 or later and runs on Linux, macOS, and Windows. Install it
from PyPI:

.. code-block:: sh

   pip install rms-polymath

This also installs its two dependencies: NumPy 2.0 or later, and ``rms-fpzip``, the
floating-point compressor used when objects are pickled. The package reads no environment
variables, needs no configuration files, and requires no external data.

Confirm the installation by printing the version:

.. code-block:: sh

   python -c "import polymath; print(polymath.__version__)"

Importing
=========

Every class is available from the top-level package:

.. code-block:: python

   from polymath import (Boolean, Matrix, Matrix3, Pair, Polynomial, Quaternion, Qube,
                         Scalar, Unit, Vector, Vector3)

The type aliases described in :doc:`user_guide_typing` live in :mod:`polymath.typedefs`.

A First Calculation
===================

The following multiplies three speeds by one direction to obtain three velocities, then
takes their lengths. The :class:`~polymath.Scalar` has a shape of ``(3,)`` and the
:class:`~polymath.Vector3` has an empty shape, so the two broadcast together, and the
product is a :class:`~polymath.Vector3` with a shape of ``(3,)`` and an item of ``(3,)``.
No index bookkeeping is needed at any step.

.. code-block:: python

    >>> from polymath import Scalar, Vector3
    >>> speed = Scalar([1., 2., 3.])
    >>> direction = Vector3([0.6, 0.8, 0.])
    >>> velocity = speed * direction
    >>> velocity
    Vector3([0.6 0.8 0. ]
     [1.2 1.6 0. ]
     [1.8 2.4 0. ])
    >>> velocity.shape, velocity.item
    ((3,), (3,))
    >>> velocity.norm()
    Scalar(1. 2. 3.)

Global Settings
===============

PolyMath has no configuration files or environment variables. Two settings apply
process-wide, and each is set by calling a method on :class:`~polymath.Qube`.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Setting
     - Effect
   * - :meth:`~polymath.Qube.prefer_builtins`
     - When True, reductions and comparisons that produce a single unmasked value return a
       Python ``float``, ``int``, or ``bool`` rather than a PolyMath object. The default
       is False. See :doc:`user_guide_math`.
   * - :meth:`~polymath.Qube.set_default_pickle_digits`
     - The floating-point precision used when pickling any object that has no setting of
       its own. The default preserves full double precision. See
       :doc:`user_guide_pickling`.

A per-object setting made with :meth:`~polymath.Qube.set_pickle_digits` takes precedence
over the global default, which in turn takes precedence over the built-in default.

Neither setting is synchronized. PolyMath objects are safe to read from several threads
at once, but modifying one while another thread reads it is not, and neither is changing
a global setting once other threads are running. Set them before starting any threads.
