=================================
Objects, Shapes, and Broadcasting
=================================

Every PolyMath class derives from :class:`~polymath.Qube`, and every object wraps a NumPy
array together with a mask, an optional unit, and an optional set of derivatives. This
chapter covers how objects are built and how their axes are organized, which is the
foundation for everything that follows.

Constructing Objects
====================

Each class is constructed from anything NumPy can turn into an array: a number, a nested
list or tuple, a NumPy array, or another PolyMath object.

.. code-block:: python

    >>> import numpy as np
    >>> from polymath import Boolean, Matrix3, Pair, Scalar, Vector3
    >>> Scalar(3)
    Scalar(3)
    >>> Scalar([1, 2, 3])
    Scalar(1 2 3)
    >>> Vector3([1., 2., 3.])
    Vector3(1. 2. 3.)
    >>> Pair([3., 4.])
    Pair(3. 4.)
    >>> Boolean([True, False])
    Boolean( True False)
    >>> Matrix3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Matrix3([1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.])

The trailing axes of the input become the item; whatever precedes them becomes the shape.
A NumPy array of shape ``(4, 3)`` therefore constructs four 3-vectors:

.. code-block:: python

    >>> v = Vector3(np.zeros((4, 3)))
    >>> v.shape
    (4,)
    >>> v.item
    (3,)

Each class checks its input. A :class:`~polymath.Vector3` requires a last axis of length
three, a :class:`~polymath.Matrix3` requires two trailing axes of length three, and a
:class:`~polymath.Boolean` accepts only truth values. Integer input to a class that holds
floating-point values only, such as :class:`~polymath.Vector3` or
:class:`~polymath.Matrix3`, is converted to floating point, as the
:class:`~polymath.Matrix3` example above shows. A :class:`~polymath.Scalar` keeps integers
as integers.

The constructor takes keyword arguments for the other parts of an object. ``mask`` marks
undefined elements, ``unit`` attaches a :class:`~polymath.Unit`, and ``derivs`` attaches a
dictionary of derivatives; each has its own chapter.

.. code-block:: python

    >>> from polymath import Unit
    >>> Scalar([1., 2., 3.], mask=[False, True, False])
    Scalar(1.0 -- 3.0; mask)
    >>> Scalar([1., 2.], unit=Unit.KM)
    Scalar(1. 2.; km)

Every class also has a static conversion method named after it, such as
:meth:`~polymath.Scalar.as_scalar`, :meth:`~polymath.Vector3.as_vector3`, and
:meth:`~polymath.Pair.as_pair`. These return the argument unchanged when it is already of
the right class, which makes them the cheapest way to accept either a PolyMath object or a
plain value:

.. code-block:: python

    >>> Scalar.as_scalar(5)
    Scalar(5)
    >>> Vector3.as_vector3([1, 2, 3])
    Vector3(1. 2. 3.)

Other constructors build objects from components or fill them with a constant.
:meth:`~polymath.Vector3.from_scalars` assembles a vector from its components, each of
which can itself be an array, and :meth:`~polymath.Vector.to_scalars` reverses it. The
class methods :meth:`~polymath.Qube.zeros`, :meth:`~polymath.Qube.ones`, and
:meth:`~polymath.Qube.filled` create objects of a given shape.

.. code-block:: python

    >>> Vector3.from_scalars(1., 2., 3.)
    Vector3(1. 2. 3.)
    >>> Vector3.from_scalars(np.arange(3.), 0., 1.)
    Vector3([0. 0. 1.]
     [1. 0. 1.]
     [2. 0. 1.])
    >>> Vector3([1., 2., 3.]).to_scalars()
    (Scalar(1.0), Scalar(2.0), Scalar(3.0))
    >>> Vector3.zeros((2,))
    Vector3([0. 0. 0.]
     [0. 0. 0.])
    >>> Vector3.filled((2,), (1., 2., 3.))
    Vector3([1. 2. 3.]
     [1. 2. 3.])

Each class defines read-only constants for its most common values, such as
:attr:`~polymath.Scalar.ZERO`, :attr:`~polymath.Scalar.PI`,
:attr:`~polymath.Vector3.XAXIS`, :attr:`~polymath.Matrix3.IDENTITY`, and
:attr:`~polymath.Boolean.TRUE`. Each class also has a constant holding a single masked
value, such as :attr:`~polymath.Scalar.MASKED`.

Shape Versus Item
=================

NumPy has one notion of shape. PolyMath splits it in two. The
:attr:`~polymath.Qube.shape` of an object is the shape of the array of items it holds, and
its :attr:`~polymath.Qube.item` is the shape of one item. The underlying NumPy array,
available through :attr:`~polymath.Qube.values`, has the two concatenated.

.. code-block:: python

    >>> m = Matrix3(np.zeros((2, 2, 3, 3)))
    >>> m.shape
    (2, 2)
    >>> m.item
    (3, 3)
    >>> m.values.shape
    (2, 2, 3, 3)

Several properties describe these axes.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Meaning
   * - :attr:`~polymath.Qube.shape`
     - The leading axes, which index the items.
   * - :attr:`~polymath.Qube.ndims`
     - The number of leading axes. :attr:`~polymath.Qube.ndim` is a synonym.
   * - :attr:`~polymath.Qube.size`
     - The number of items, which is the product of the shape.
   * - :attr:`~polymath.Qube.item`
     - The trailing axes, which make up one item.
   * - :attr:`~polymath.Qube.rank`
     - The number of item axes.
   * - :attr:`~polymath.Qube.isize`
     - The number of elements in one item.
   * - :attr:`~polymath.Qube.numer`, :attr:`~polymath.Qube.denom`
     - The item axes split into a numerator and a denominator; see
       :doc:`user_guide_derivatives`. For an object without a denominator,
       :attr:`~polymath.Qube.numer` equals :attr:`~polymath.Qube.item` and
       :attr:`~polymath.Qube.denom` is empty.

An object with an empty shape holds a single item. For a :class:`~polymath.Scalar` or
:class:`~polymath.Boolean` with an empty shape, :attr:`~polymath.Qube.values` is a Python
number rather than a NumPy array:

.. code-block:: python

    >>> Scalar(3.).values
    3.0
    >>> Scalar([1, 2]).values
    array([1, 2])

Reading Values Back
===================

:attr:`~polymath.Qube.values`, or its synonym :attr:`~polymath.Qube.vals`, returns the
data as a NumPy array in the standard units of kilometers, seconds, and radians regardless
of the unit the object carries. :attr:`~polymath.Qube.mvals` returns the same data as a
:class:`numpy.ma.MaskedArray` whose mask is the object's mask, and
:meth:`~polymath.Qube.into_unit` returns the values converted into the object's unit.

.. code-block:: python

    >>> x = Scalar([1., 2., 3.], mask=[False, True, False])
    >>> x.mvals
    masked_array(data=[1.0, --, 3.0],
                 mask=[False,  True, False],
           fill_value=1e+20)

The string form of an object, produced by :meth:`~polymath.Qube.__str__`, shows the class
name and the values, followed by a semicolon and any suffixes that apply: the denominator
shape, the word ``mask`` if any element is masked, the unit, and the names of the
derivatives. Masked elements print as ``--``.

Reshaping
=========

The methods :meth:`~polymath.Qube.reshape`, :meth:`~polymath.Qube.flatten`,
:meth:`~polymath.Qube.swap_axes`, :meth:`~polymath.Qube.roll_axis`, and
:meth:`~polymath.Qube.move_axis` rearrange the leading axes and leave the items alone.
Each returns a shallow copy that shares memory with the original, and each applies the
same change to the derivatives.

.. code-block:: python

    >>> a = Scalar(np.arange(6.))
    >>> a.reshape((2, 3))
    Scalar([0. 1. 2.]
     [3. 4. 5.])
    >>> a.reshape((2, 3)).flatten()
    Scalar(0. 1. 2. 3. 4. 5.)
    >>> b = Scalar(np.arange(24.).reshape(2, 3, 4))
    >>> b.swap_axes(0, 2).shape
    (4, 3, 2)
    >>> b.roll_axis(2).shape
    (4, 2, 3)
    >>> b.move_axis(0, -1).shape
    (3, 4, 2)

:meth:`~polymath.Qube.stack` joins several objects along a leading axis:

.. code-block:: python

    >>> from polymath import Qube
    >>> Qube.stack(Vector3.XAXIS, Vector3.YAXIS)
    Vector3([1. 0. 0.]
     [0. 1. 0.])

Broadcasting
============

When two objects are combined, their shapes are broadcast together following the NumPy
rules, described at https://numpy.org/doc/stable/user/basics.broadcasting.html. The item
axes never participate, so no reshaping is needed to combine objects whose items differ.
A :class:`~polymath.Scalar` of shape ``(2,)`` times a single :class:`~polymath.Vector3`
gives two vectors:

.. code-block:: python

    >>> Scalar([1., 2.]) * Vector3.XAXIS
    Vector3([1. 0. 0.]
     [2. 0. 0.])

Two rotation matrices of shape ``(2,)`` applied to vectors of shape ``(5, 1)`` give
vectors of shape ``(5, 2)``:

.. code-block:: python

    >>> rotation = Matrix3.x_rotation(np.array([0., np.pi / 2]))
    >>> v = Vector3(np.zeros((5, 1, 3)) + [0., 1., 0.])
    >>> (rotation * v).shape
    (5, 2)

:meth:`~polymath.Qube.broadcasted_shape` reports the shape an operation will produce
without performing it, and :meth:`~polymath.Qube.broadcast_to` and
:meth:`~polymath.Qube.broadcast` expand objects explicitly. Broadcast objects share memory
with their source, so they are read-only.

.. code-block:: python

    >>> Qube.broadcasted_shape(rotation, v)
    (5, 2)
    >>> Scalar(1.).broadcast_to((3,))
    Scalar(1. 1. 1.)
    >>> Scalar(1.).broadcast_to((3,)).readonly
    True

Converting Between Classes
==========================

:meth:`~polymath.Qube.cast` reinterprets an object as another class with a compatible item
shape, and :meth:`~polymath.Qube.as_this_type` converts an argument to the class of the
object it is called on. Classes with fixed items also convert on construction, so passing
a :class:`~polymath.Vector` with three components to the :class:`~polymath.Vector3`
constructor works.

.. code-block:: python

    >>> from polymath import Matrix, Vector
    >>> Vector([1., 2., 3.]).cast(Vector3)
    Vector3(1. 2. 3.)
    >>> Vector3.XAXIS.as_this_type(Vector([4., 5., 6.]))
    Vector3(4. 5. 6.)

Vectors and matrices convert into each other. :meth:`~polymath.Vector.as_column`,
:meth:`~polymath.Vector.as_row`, and :meth:`~polymath.Vector.as_diagonal` turn a vector
into a matrix; :meth:`~polymath.Matrix.row_vector`, :meth:`~polymath.Matrix.row_vectors`,
:meth:`~polymath.Matrix.column_vector`, and :meth:`~polymath.Matrix.column_vectors`
extract vectors from a matrix; and :meth:`~polymath.Vector.to_scalar` and
:meth:`~polymath.Matrix.to_scalar` extract one component.

.. code-block:: python

    >>> Vector([1., 2.]).as_column()
    Matrix([1.]
     [2.])
    >>> Matrix([[1., 2.], [3., 4.]]).column_vectors()
    (Vector(1. 3.), Vector(2. 4.))
    >>> Vector3([1., 2., 3.]).to_scalar(1)
    Scalar(2.0)

The numeric type of an object changes with :meth:`~polymath.Qube.as_float`,
:meth:`~polymath.Qube.as_int`, and :meth:`~polymath.Qube.as_bool`, and
:meth:`~polymath.Qube.is_float`, :meth:`~polymath.Qube.is_int`, and
:meth:`~polymath.Qube.is_bool` report it. Converting a :class:`~polymath.Scalar` to truth
values yields a :class:`~polymath.Boolean`, where zero is False and anything else is True.

.. code-block:: python

    >>> Scalar([1, 2]).as_float()
    Scalar(1. 2.)
    >>> Scalar([1.7, 2.2]).as_int()
    Scalar(1 2)
    >>> Scalar([0, 2]).as_bool()
    Boolean(False  True)

Read-Only Objects
=================

Objects are writable when constructed. :meth:`~polymath.Qube.as_readonly` makes an object
and its derivatives read-only, after which any attempt to assign into it raises
:class:`ValueError`. The :attr:`~polymath.Qube.readonly` property reports the state. There
is no way back; :meth:`~polymath.Qube.copy` returns a writable copy instead, and
:meth:`~polymath.Qube.clone` returns a shallow copy that keeps the read-only state.

.. code-block:: python

    >>> r = Scalar([1., 2.]).as_readonly()
    >>> r.readonly
    True
    >>> r[0] = 5.
    Traceback (most recent call last):
    ...
    ValueError: Scalar object is read-only
    >>> r.copy().readonly
    False

An operation that shares memory with a read-only object, such as reshaping or
broadcasting, returns a read-only object; an operation that computes fresh values, such
as a negation, returns a writable one. The class constants are read-only, so a
calculation can never corrupt :attr:`~polymath.Vector3.ZERO` by accident. Read-only
status is useful in general whenever objects share memory, which is common, because it
stops one object from being modified by way of another.

Custom Attributes
=================

:meth:`~polymath.Qube.add_attr` attaches an attribute of your own to an object, so that
application-specific information can travel with it. The attribute is carried along by
every copy and clone but not by an operation that computes different values, and a name
beginning with ``d_d`` is reserved for derivatives.

.. code-block:: python

    >>> obj = Scalar([1., 2.])
    >>> obj.add_attr('label', 'north pole')
    Scalar(1. 2.)
    >>> obj.label
    'north pole'
    >>> obj.copy().label
    'north pole'
    >>> hasattr(obj + 1, 'label')
    False
