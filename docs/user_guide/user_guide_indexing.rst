======================
Indexing and Iteration
======================

Indexing a PolyMath object works much as indexing a NumPy array does, with the index
applying to the shape and leaving the items intact. Beyond what NumPy accepts, an index
may itself be a PolyMath object, and a masked index selects nothing at the masked
locations.

Basic Indexing
==============

Integers, slices, ellipses, and NumPy arrays index the leading axes.

.. code-block:: python

    >>> import numpy as np
    >>> from polymath import Boolean, Pair, Scalar, Vector3
    >>> a = Scalar(np.arange(12.).reshape(3, 4))
    >>> a[0]
    Scalar(0. 1. 2. 3.)
    >>> a[1, 2]
    Scalar(6.0)
    >>> a[:, 1]
    Scalar(1. 5. 9.)
    >>> a[..., -1]
    Scalar( 3.  7. 11.)

An index of ``True`` selects the whole object, and an index of ``False`` selects nothing,
leaving a first axis of length zero.

.. code-block:: python

    >>> a[True].shape
    (3, 4)
    >>> a[False].shape
    (0, 4)

Indexing with PolyMath Objects
==============================

A :class:`~polymath.Boolean` selects the elements where it is True, like a boolean NumPy
array. A :class:`~polymath.Scalar` of integers selects by position, like an integer NumPy
array. A :class:`~polymath.Pair` of integers indexes two consecutive axes at once, and a
:class:`~polymath.Vector` with more components indexes as many axes. In every case, the
elements selected by a masked location of the index come back masked.

.. code-block:: python

    >>> a[Boolean([True, False, True])]
    Scalar([ 0.  1.  2.  3.]
     [ 8.  9. 10. 11.])
    >>> a[Boolean([True, False, True], mask=[False, False, True])]
    Scalar([0.0 1.0 2.0 3.0]
     [-- -- -- --]; mask)
    >>> a[Scalar([0, 2])]
    Scalar([ 0.  1.  2.  3.]
     [ 8.  9. 10. 11.])
    >>> a[Pair([[0, 1], [2, 3]])]
    Scalar( 1. 11.)

As in NumPy, the shape of an array-valued index appears in the shape of the result. When
several array-valued indices are used together, their broadcasted shape appears at the
position of the first one, which differs slightly from the NumPy rule for indices
separated by a slice. With ``A`` of shape ``(6, 7, 8, 9)``, ``B`` of shape ``(3, 1)``, and
``C`` of shape ``(4,)``:

.. code-block:: python

    >>> A = Scalar(np.zeros((6, 7, 8, 9)))
    >>> B = Scalar(np.zeros((3, 1), dtype=int))
    >>> C = Scalar(np.zeros((4,), dtype=int))
    >>> A[B].shape
    (3, 1, 7, 8, 9)
    >>> A[:, B].shape
    (6, 3, 1, 8, 9)
    >>> A[B, C].shape
    (3, 4, 8, 9)
    >>> A[:, B, :, C].shape
    (6, 3, 4, 8)

Assigning by Index
==================

Assignment through an index modifies the object in place, and it requires a writable
object. Where a :class:`~polymath.Boolean` or :class:`~polymath.Scalar` index is masked,
the corresponding elements are left unchanged.

.. code-block:: python

    >>> b = Scalar(np.zeros(4))
    >>> b[1] = 5.
    >>> b[Boolean([False, False, True, True])] = 7.
    >>> b
    Scalar(0. 5. 7. 7.)
    >>> b[Scalar([0, 3], mask=[False, True])] = 9.
    >>> b
    Scalar(9. 5. 7. 7.)

Indexing NumPy Arrays
=====================

Several methods convert a PolyMath object into something that can index a NumPy array.
:meth:`~polymath.Scalar.as_index` and :meth:`~polymath.Vector.as_index` return integer
indices, :meth:`~polymath.Scalar.as_index_and_mask` and
:meth:`~polymath.Vector.as_index_and_mask` return the indices together with a mask, and
:meth:`~polymath.Boolean.as_index` returns a boolean array. The
:attr:`~polymath.Qube.antimask` property and the :meth:`~polymath.Qube.as_mask_where_zero`
family described in :doc:`user_guide_masks` return boolean arrays derived from the mask.

.. code-block:: python

    >>> Scalar([1, 0]).as_index()
    array([1, 0])
    >>> Boolean([True, False]).as_index()
    array([ True, False])

Iteration
=========

Iterating over an object walks its first axis, yielding one object per index, and
:func:`len` gives the length of that axis. :meth:`~polymath.Qube.ndenumerate` iterates
over every item of a multidimensional object, yielding each index along with the item.

.. code-block:: python

    >>> v = Vector3(np.arange(6.).reshape(2, 3))
    >>> for item in v:
    ...     print(item)
    Vector3(0. 1. 2.)
    Vector3(3. 4. 5.)
    >>> for index, item in Scalar([[1., 2.], [3., 4.]]).ndenumerate():
    ...     print(index, item)
    (0, 0) Scalar(1.0)
    (0, 1) Scalar(2.0)
    (1, 0) Scalar(3.0)
    (1, 1) Scalar(4.0)
