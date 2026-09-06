================
Type Annotations
================

The PolyMath modules carry no inline type annotations. Instead, the package ships a
``py.typed`` marker and two stub files, one for the package and one for
:mod:`polymath.typedefs`, so a type checker such as mypy sees the signature of every
public class, method, and property when it checks code that imports PolyMath. Nothing
needs to be configured; installing the package is enough. Import from the package
itself, as every example in this guide does. The submodules that define the classes are
an implementation detail: an import such as one from a module named after a class is
not supported and carries no type information.

The Type Aliases
================

The :mod:`polymath.typedefs` module supplements the stubs with aliases for use in your
own annotations. Each alias names everything the corresponding constructor accepts, which
is broader than the class itself.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Accepts
   * - :data:`~polymath.typedefs.ScalarLike`, :data:`~polymath.typedefs.BooleanLike`,
       :data:`~polymath.typedefs.QubeLike`
     - A PolyMath object, a numeric array, a nested sequence of numbers, or a single
       number.
   * - :data:`~polymath.typedefs.VectorLike`
     - A PolyMath object, a numeric array with one or more axes, or a nested sequence.
   * - :data:`~polymath.typedefs.PairLike`, :data:`~polymath.typedefs.Vector3Like`,
       :data:`~polymath.typedefs.QuaternionLike`
     - A PolyMath object, a numeric array whose last axis has length two, three, or four
       respectively, or a nested sequence.
   * - :data:`~polymath.typedefs.MatrixLike`
     - A PolyMath object, a numeric array with two or more axes, or a nested sequence.
   * - :data:`~polymath.typedefs.Matrix3Like`
     - A PolyMath object, a numeric array whose last two axes each have length three, or
       a nested sequence.
   * - :data:`~polymath.typedefs.ValsType`
     - Anything the :attr:`~polymath.Qube.values` property can return: a number or a
       NumPy array.
   * - :data:`~polymath.typedefs.MaskType`
     - Anything the :attr:`~polymath.Qube.mask` property can return: a boolean or a
       boolean NumPy array.

Using an Alias
==============

An alias is an ordinary runtime object, so it can be imported and used in an annotation
anywhere, without a ``TYPE_CHECKING`` guard:

.. code-block:: python

    >>> from polymath import Scalar, Vector3
    >>> from polymath.typedefs import Vector3Like
    >>> def speed(velocity: Vector3Like) -> Scalar:
    ...     return Vector3.as_vector3(velocity).norm()
    >>> speed([3., 4., 0.])
    Scalar(5.0)

Three caveats apply. First, every alias includes :class:`~polymath.Qube`, because each
constructor re-wraps any PolyMath object, so annotating a parameter as
:data:`~polymath.typedefs.Vector3Like` documents intent and rules out unrelated types such
as strings and dictionaries but does not restrict the argument to a
:class:`~polymath.Vector3`. Convert inside the function, as the example does. Second, the
arithmetic operators are declared once, on :class:`~polymath.Qube`, and their declared
result is a :class:`~polymath.Qube`, even though at runtime a product of a
:class:`~polymath.Vector3` and a :class:`~polymath.Scalar` is a
:class:`~polymath.Vector3`. A function that returns the result of an operation therefore
either declares :class:`~polymath.Qube` as its return type or passes the result through
the converter of the class it expects:

.. code-block:: python

    >>> from polymath.typedefs import ScalarLike
    >>> def scale(vector: Vector3Like, factor: ScalarLike) -> Vector3:
    ...     return Vector3.as_vector3(Vector3.as_vector3(vector) * Scalar.as_scalar(factor))
    >>> scale([1., 2., 3.], 2)
    Vector3(2. 4. 6.)

Third, a return type is ``Any`` wherever the docstring does not state one, so some
results need a cast or an ``isinstance`` check before a type checker will allow a
class-specific method to be called on them.

Checking Your Code
==================

Run mypy on your own modules as usual. The stubs are found through the installed package,
so no path configuration is needed:

.. code-block:: sh

   mypy your_module.py
