============
Architecture
============

Class Hierarchy
===============

.. mermaid::

   classDiagram
       class Qube {
           +values
           +mask
           +derivs
           +unit_
           +shape
           +item
           +numer
           +denom
           +readonly
           +__init__(arg, mask, derivs, unit, nrank, drank, example, default)
           +clone()
           +zeros()
           +ones()
           +filled()
           +_new_from_parts()
           +_set_values()
           +_set_mask()
       }
       class Unit {
           +exponents
           +triple
           +name
           +as_unit()
           +from_this()
           +into_this()
       }
       class Scalar {
           +as_scalar()
           +sin()
           +sqrt()
           +max()
       }
       class Boolean {
           +as_boolean()
           +as_index()
       }
       class Vector {
           +as_vector()
           +dot()
           +cross()
           +norm()
           +to_scalars()
       }
       class Pair {
           +as_pair()
           +swapxy()
           +angle()
       }
       class Vector3 {
           +as_vector3()
           +from_ra_dec_length()
           +spin()
       }
       class Quaternion {
           +as_quaternion()
           +to_matrix3()
           +conj()
       }
       class Polynomial {
           +as_polynomial()
           +eval()
           +roots()
       }
       class Matrix {
           +as_matrix()
           +inverse()
           +solve()
       }
       class Matrix3 {
           +as_matrix3()
           +x_rotation()
           +rotate()
       }
       Qube <|-- Scalar
       Scalar <|-- Boolean
       Qube <|-- Vector
       Vector <|-- Pair
       Vector <|-- Vector3
       Vector <|-- Quaternion
       Vector <|-- Polynomial
       Qube <|-- Matrix
       Matrix <|-- Matrix3
       Qube o-- "0..1" Unit : _unit
       Qube o-- "0..*" Qube : _derivs

None of the classes is abstract. :class:`~polymath.Qube` is fully functional on its own
and can be instantiated with any numerator rank, which is how a few operations build
intermediate results; it is left out of the public constructors only because a concrete
class documents intent better. Every subclass is a specialization that fixes some of the
class constants described below and adds methods.

The three lineages differ in the rank of their numerator. :class:`~polymath.Scalar` and
its subclass :class:`~polymath.Boolean` have rank 0. :class:`~polymath.Vector` and its
four subclasses have rank 1, with :class:`~polymath.Pair`, :class:`~polymath.Vector3`,
and :class:`~polymath.Quaternion` fixing the length at 2, 3, and 4, and
:class:`~polymath.Polynomial` leaving it free but reinterpreting the components as
coefficients. :class:`~polymath.Matrix` and its subclass :class:`~polymath.Matrix3` have
rank 2, with :class:`~polymath.Matrix3` fixing the shape at 3x3 and adding the rotation
methods. :class:`~polymath.Unit` stands apart: it is an ordinary class that a
:class:`~polymath.Qube` refers to, and it has no shape, mask, or derivatives.

Anatomy of a Qube
=================

Every object is a small bundle of attributes, all set by the constructor and read
directly by the extension functions. The table lists them in the order the constructor
assigns them, which is also the order in the class-level tuple that the copying methods
use to transfer them. Keep the two in step when adding an attribute.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Attribute
     - Contents and invariant
   * - ``_values``
     - A NumPy array, or a Python ``float``, ``int``, or ``bool`` when the shape and item
       are both empty. Never a zero-dimensional array and never a NumPy scalar type;
       the construction path reduces both. Always in standard units.
   * - ``_mask``
     - A Python ``bool``, or a boolean array whose shape equals ``_shape``. Never an
       array with item axes.
   * - ``_is_array``, ``_is_scalar``
     - Whether ``_values`` is an array. Exactly one is True.
   * - ``_shape``, ``_ndims``, ``_size``
     - The leading axes, their count, and their product.
   * - ``_rank``, ``_nrank``, ``_drank``
     - The number of item axes, and its split into numerator and denominator axes.
   * - ``_item``, ``_numer``, ``_denom``
     - The item shape and its split. ``_item == _numer + _denom``, and
       ``np.shape(_values) == _shape + _item``.
   * - ``_isize``, ``_nsize``, ``_dsize``
     - The products of the three item shapes.
   * - ``_unit``
     - A :class:`~polymath.Unit`, or None for a unitless object. A unit of
       :attr:`~polymath.Unit.UNITLESS` is stored as None.
   * - ``_readonly``
     - True if the values array is flagged non-writable. The mask array is flagged the
       same way whenever the object is read-only.
   * - ``_truth_if_any``, ``_truth_if_all``
     - Flags consulted when the object is used in a boolean context.
   * - ``_default``
     - The value stored in masked elements when an object is unpickled or unshrunk. It
       has the item shape and the object's data type.
   * - ``_derivs``
     - A dictionary of derivatives, each a :class:`~polymath.Qube` broadcastable to
       ``_shape``. Each is also exposed as an attribute named ``d_d`` plus its key.
   * - ``_cache``
     - Values derived from the object and cleared whenever it changes. See
       `Caching and Shrinking`_.
   * - ``_added_attrs``
     - A frozenset of the names added by :meth:`~polymath.Qube.add_attr`. The
       class-level default is shared and never modified in place.
   * - ``_pickle_digits``, ``_pickle_reference``
     - Present only once :meth:`~polymath.Qube.set_pickle_digits` has been called.

Two further invariants apply to the class as a whole. An object is not hashable, because
it compares by value and is mutable; the binding module sets the hash to None explicitly,
because defining the equality operator after the class body would otherwise leave the
default identity hash in place. And nothing is synchronized: reading an object from
several threads is safe, but modifying one while another thread reads it is not, and
neither is changing a global setting once other threads are running.

Class Constants
===============

A subclass declares what it accepts through class constants, which the constructor and
the conversion functions consult. Each subclass module sets all of them explicitly, one
per line with a trailing comment, even where the value matches the base class.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Constant
     - Meaning
   * - ``_NRANK``
     - The number of numerator axes, or None to leave it unconstrained. The base class
       uses None.
   * - ``_NUMER``
     - The numerator shape, or None to leave it unconstrained. :class:`~polymath.Vector`
       and :class:`~polymath.Matrix` fix the rank but not the shape.
   * - ``_FLOATS_OK``, ``_INTS_OK``, ``_BOOLS_OK``
     - Which data types the class may hold. Integer input to a class with only
       ``_FLOATS_OK`` is converted; boolean input to a class without ``_BOOLS_OK`` is an
       error.
   * - ``_UNITS_OK``
     - Whether the class may carry a unit. False for :class:`~polymath.Boolean`,
       :class:`~polymath.Matrix3`, and :class:`~polymath.Quaternion`.
   * - ``_DERIVS_OK``
     - Whether the class may carry derivatives or a denominator. False for
       :class:`~polymath.Boolean`.
   * - ``_DEFAULT_VALUE``
     - The value for masked elements of an object with no denominator. Absent from the
       base class, which then uses ones of the item shape.
   * - ``_DERIV_CLASS``
     - The class of a derivative of this class, or None when a derivative has the same
       class as the object. :class:`~polymath.Matrix3` names :class:`~polymath.Matrix`,
       because the derivative of a rotation matrix is not a rotation matrix.

Three more constants on the base class are switches for testing. ``_DISABLE_CACHE``
bypasses the cache, ``_DISABLE_SHRINKING`` turns :meth:`~polymath.Qube.shrink` and
:meth:`~polymath.Qube.unshrink` into no-ops, and ``_IGNORE_UNSHRUNK_AS_CACHED`` makes
:meth:`~polymath.Qube.unshrink` ignore its cached result. A calculation must give the same
answer with any combination of these set. ``__array_priority__`` is set so that NumPy
defers to PolyMath's operators when an array appears on the left of an expression.

Construction Paths
==================

There are three ways an object comes into being, and choosing the right one is most of
what writing an operation correctly involves.

**The constructor** :meth:`~polymath.Qube.__init__` is the public path. It accepts any
array-like argument, infers the split between shape and item from the class constants
and the ``nrank`` and ``drank`` arguments, coerces the data type, validates the mask,
checks the unit and derivatives against the class constants, and installs the
derivatives. Its ``example`` argument copies the mask, unit, ranks, and default from
another object wherever they were not given explicitly, and its ``op`` argument names the
operation for error messages. This path does the most work and the most checking, so
operations use it only when the input is not yet known to be valid.

**The fast path** ``_new_from_parts`` is the internal counterpart. It takes a values
array whose data type is already acceptable, a mask already broadcastable to the leading
shape, and the ranks, and it assigns every attribute directly without checks. It never
copies derivatives; the caller inserts them afterward. Its ``example`` argument lets it
reuse the size products and the default from an operand whose item shape and data type
carried through, which is purely an optimization. Every arithmetic operator uses this
path, so an error in a caller's bookkeeping surfaces as a corrupt object rather than an
exception. Use it only when the caller has computed the result itself and can vouch for
every part.

**Cloning** produces a shallow copy that shares the values array. :meth:`~polymath.Qube.clone`
carries the attributes added by :meth:`~polymath.Qube.add_attr`, and it is the right basis
for a result that describes the same quantity, such as a reshaped or remasked view. The
private variant ``_clone_new_values`` omits those attributes, and it is the right basis
for a result that is about to be given different values, such as a negation. Both
optionally clone the derivatives, and both start with an empty cache unless asked to
retain it.

After construction, two low-level methods modify an object in place. ``_set_values``
replaces the values array, optionally only where an antimask is True, and re-derives the
read-only state from the array's flags. ``_set_mask`` replaces the mask and preserves the
read-only state. Both clear the cache. The class methods :meth:`~polymath.Qube.zeros`,
:meth:`~polymath.Qube.ones`, and :meth:`~polymath.Qube.filled` build constant objects on
top of the constructor, and ``_default_for`` computes the default value for a class, item
shape, and data type.

Extension Binding
=================

The module ``src/polymath/qube.py`` defines the class, and the package
``src/polymath/extensions/`` defines almost everything the class can do. Each extension
module is a collection of plain functions whose first parameter is ``self``, and the file
``src/polymath/extensions/__init__.py`` assigns each one onto :class:`~polymath.Qube` as
an attribute, so that it becomes a method. The assignments are grouped by module and
listed in the order the functions appear in the module, which makes the binding file a
table of contents for the class.

Three rules follow from this arrangement.

1. **Import order.** ``src/polymath/__init__.py`` imports the extensions package before
   any subclass module. Each subclass builds its read-only constants as it loads, and
   constructing those objects calls bound methods such as
   :meth:`~polymath.Qube.as_readonly`, which do not exist until the binding has run.
2. **No subclass imports in extensions.** An extension module cannot import a subclass at
   module level without creating an import cycle, because every subclass imports
   :class:`~polymath.Qube`. Instead, each subclass module registers itself on the base
   class at the bottom of the file, as ``Qube._SCALAR_CLASS``, ``Qube._BOOLEAN_CLASS``,
   ``Qube._VECTOR_CLASS``, ``Qube._PAIR_CLASS``, ``Qube._VECTOR3_CLASS``,
   ``Qube._QUATERNION_CLASS``, ``Qube._MATRIX_CLASS``, or ``Qube._MATRIX3_CLASS``, and
   the extension functions reach the subclasses through those attributes at call time.
3. **Class methods are wrapped at the binding site.** A ``@staticmethod`` or
   ``@property`` can be written at module level and bound directly, but a module-level
   ``@classmethod`` is not a function and stubtest rejects it. Write the plain function
   with the class as its first parameter and wrap it in ``classmethod`` in the binding
   file, as the data-type helpers in the dtypes module are.

Operator Dispatch
=================

The arithmetic operators live in the math operations module. Each binary operator first
converts its right operand into something compatible with the left one, which is where a
Python number, a NumPy array, or a nested sequence becomes a :class:`~polymath.Qube`, and
then dispatches on the combination of classes: a number or a :class:`~polymath.Scalar`
operand scales the other operand item by item, while two operands with item axes must
have matching numerators. Each operator computes the values with NumPy, combines the two
masks, checks or combines the units, builds the result with the fast construction path,
and then propagates the derivatives with a helper of its own, such as the one that
applies the product rule for multiplication. The derivative helpers are where the
denominator axes matter: a derivative may have a denominator that its parent lacks, and
the helper must broadcast the parent's values against it correctly.

Subclasses override an operator only to change its meaning. :class:`~polymath.Matrix3`
and :class:`~polymath.Quaternion` redefine multiplication as rotation composition and
the quaternion product, :class:`~polymath.Polynomial` redefines the arithmetic operators
as polynomial arithmetic, and :class:`~polymath.Boolean` redefines the arithmetic
operators to return a :class:`~polymath.Scalar`, since the sum of two truth values is a
count.

The error helpers in the errors module phrase every message the same way, naming the
operation, the classes involved, and the offending shapes. Use them rather than raising
directly, so that error messages stay consistent.

Caching and Shrinking
=====================

Each object carries a dictionary of cached results, cleared by every method that changes
the object. The keys in use are the corners of the unmasked region and the slice that
selects it, both used by shrinking; the copy without derivatives returned by
:attr:`~polymath.Qube.wod`; and the shrunken and unshrunken forms of the object, which
let a sequence of operations on a shrunken object reuse one shrink and one unshrink.
A method that returns a copy sharing the values array may retain the cache, but must
drop the entries that describe a different object.

Shrinking exists because the objects that describe an image are often mostly masked.
:meth:`~polymath.Qube.shrink` finds the smallest hypercube containing the unmasked
elements, from the cached corners, slices it out, and then flattens it to the elements
selected by the antimask, returning a read-only one-dimensional object.
:meth:`~polymath.Qube.unshrink` reverses the process. Every operation must give the same
result on shrunken and unshrunken operands, which the ``_DISABLE_SHRINKING`` switch
exists to verify.

The Unit Class
==============

:class:`~polymath.Unit` records three integer exponents on distance, time, and angle, and
a triple of integers giving the exact factor that converts a value in the unit into the
standard units of kilometers, seconds, and radians, as a numerator, a denominator, and a
power of pi. Because the factor is exact, converting a value out of a unit and back
loses nothing. The values inside a :class:`~polymath.Qube` are always in standard units,
so a unit affects only construction, display, and compatibility checks; arithmetic on
units, which builds compound units, is implemented on the class itself. The class
constants for the common units are built at the bottom of the module, and a registry
keyed by name serves :meth:`~polymath.Unit.as_unit`.
