=====
Masks
=====

Every PolyMath object carries a boolean mask that marks the elements whose value is
undefined. The mask is what lets a calculation proceed over an entire array even when
some elements have no meaningful answer, such as lines of sight that miss a planet, and it
is what prevents the warnings and exceptions that NumPy would raise for a division by
zero or the square root of a negative number.

What a Mask Means
=================

The :attr:`~polymath.Qube.mask` property is a single ``False`` when nothing is masked, a
single ``True`` when everything is masked, and otherwise a boolean NumPy array with the
object's :attr:`~polymath.Qube.shape`. It never has item axes; a
:class:`~polymath.Vector3` is masked as a whole vector, not component by component.
:attr:`~polymath.Qube.antimask` is its logical inverse, which is convenient as an index
that selects the valid elements.

.. code-block:: python

    >>> import numpy as np
    >>> from polymath import Boolean, Scalar, Vector3
    >>> x = Scalar([1., 2., 3.], mask=[False, True, False])
    >>> x
    Scalar(1.0 -- 3.0; mask)
    >>> x.mask
    array([False,  True, False])
    >>> x.antimask
    array([ True, False,  True])
    >>> Scalar([1., 2., 3.]).mask
    False

A masked element still has a value in the underlying array, which
:attr:`~polymath.Qube.values` returns unchanged. Treat that value as meaningless.
:attr:`~polymath.Qube.mvals` returns a :class:`numpy.ma.MaskedArray` that hides it, and
:meth:`~polymath.Qube.without_mask` returns a copy with the mask removed, which exposes
it. Each object also has a :attr:`~polymath.Qube.default` value, chosen so as not to break
arithmetic, which is what masked elements hold after an object is unpickled or restored by
:meth:`~polymath.Qube.unshrink`.

Under normal circumstances a masked value means "this value does not exist". This
resembles NumPy's not-a-number, but the rules differ:

* Two masked values of the same class compare equal, and a masked value never equals an
  unmasked one.
* Any unary or binary operation involving a masked element produces a masked element.
* Reductions ignore masked elements. :meth:`~polymath.Scalar.max` returns the maximum of
  the unmasked values, :meth:`~polymath.Qube.mean` averages them, and
  :meth:`~polymath.Qube.all` is True if every unmasked value is True.

.. code-block:: python

    >>> y = Scalar([4., -1., 9.]).mask_where_lt(0)
    >>> y + 1
    Scalar(5.0 -- 10.0; mask)
    >>> y.max()
    Scalar(9.0)
    >>> Scalar([1., 2.], mask=[True, False]) == Scalar([3., 2.], mask=[True, False])
    Boolean( True  True)

Where Masks Come From
=====================

A mask is set at construction with the ``mask`` argument, and it appears automatically
when an operation has no defined result:

.. code-block:: python

    >>> Scalar([4., -1., 9.]).sqrt()
    Scalar(2.0 -- 3.0; mask)
    >>> Scalar([1., 2.]) / Scalar([0., 1.])
    Scalar(-- 2.0; mask)

A family of methods masks elements by value. Each returns a copy, and each accepts a
``replace`` argument giving a value to store in the newly masked elements.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Method
     - Masks the elements that are
   * - :meth:`~polymath.Qube.mask_where`
     - True in a given mask or :class:`~polymath.Boolean`.
   * - :meth:`~polymath.Qube.mask_where_eq`, :meth:`~polymath.Qube.mask_where_ne`
     - Equal to, or not equal to, a value.
   * - :meth:`~polymath.Qube.mask_where_lt`, :meth:`~polymath.Qube.mask_where_le`,
       :meth:`~polymath.Qube.mask_where_gt`, :meth:`~polymath.Qube.mask_where_ge`
     - Below or above a limit.
   * - :meth:`~polymath.Qube.mask_where_between`,
       :meth:`~polymath.Qube.mask_where_outside`
     - Inside or outside a range, with a ``mask_endpoints`` option.
   * - :meth:`~polymath.Qube.clip`
     - Outside a range, after clipping them to it; pass ``remask=False`` to clip without
       masking.

.. code-block:: python

    >>> x = Scalar([1., 2., 3., 4., 5.])
    >>> x.mask_where(x > 4)
    Scalar(1.0 2.0 3.0 4.0 --; mask)
    >>> x.mask_where_between(2., 4.)
    Scalar(1.0 2.0 -- 4.0 5.0; mask)
    >>> x.mask_where_outside(2., 4.)
    Scalar(-- 2.0 3.0 4.0 --; mask)
    >>> Scalar([-2., 0.5, 3.]).clip(0., 1.)
    Scalar(-- 0.5 --; mask)
    >>> Scalar([-2., 0.5, 3.]).clip(0., 1., remask=False)
    Scalar(0.  0.5 1. )

A mask cannot be assigned directly. :meth:`~polymath.Qube.remask` returns a shallow copy
with a replacement mask, :meth:`~polymath.Qube.remask_or` returns one with the given mask
added to the existing one, and :meth:`~polymath.Qube.as_all_masked` masks everything.
Every class has a constant holding a single masked value, such as
:attr:`~polymath.Scalar.MASKED` and :attr:`~polymath.Vector3.MASKED`.

.. code-block:: python

    >>> x.remask([True, False, False, False, False])
    Scalar(-- 2.0 3.0 4.0 5.0; mask)
    >>> Vector3.MASKED
    Vector3(-- -- --; mask)

Inspecting a Mask
=================

:meth:`~polymath.Qube.count_masked`, :meth:`~polymath.Qube.count_unmasked`, and
:meth:`~polymath.Qube.is_all_masked` summarize the mask.
:meth:`~polymath.Qube.expand_mask` returns a copy whose mask is a full array even if
nothing is masked, and :meth:`~polymath.Qube.collapse_mask` returns one whose mask is a
single boolean when the array allows it. :meth:`~polymath.Qube.as_mask_where_nonzero`,
:meth:`~polymath.Qube.as_mask_where_zero`,
:meth:`~polymath.Qube.as_mask_where_nonzero_or_masked`, and
:meth:`~polymath.Qube.as_mask_where_zero_or_masked` derive a NumPy boolean array from the
values and the mask together, for use in indexing NumPy arrays.

.. code-block:: python

    >>> y = Scalar([4., -1., 9.]).mask_where_lt(0)
    >>> print(y.count_masked(), y.count_unmasked(), y.is_all_masked())
    1 2 False

Three-Valued Logic
==================

Sometimes a masked value is better read as "unknown" than as "nonexistent". The methods
whose names begin with ``tvl_`` follow the rules of three-valued logic, in which a result
is True or False whenever the unknown elements could not change it and is masked only
when they could.

* :meth:`~polymath.Qube.tvl_and` is False if either operand is False, even when the other
  is masked.
* :meth:`~polymath.Qube.tvl_or` is True if either operand is True, even when the other is
  masked.
* :meth:`~polymath.Qube.tvl_all` is True only if every value is True, False if any value
  is False, and masked if the only values are True and unknown.
* :meth:`~polymath.Qube.tvl_any` is True if any value is True, False if every value is
  False, and masked if the only values are False and unknown.
* :meth:`~polymath.Qube.tvl_eq`, :meth:`~polymath.Qube.tvl_ne`,
  :meth:`~polymath.Qube.tvl_lt`, :meth:`~polymath.Qube.tvl_le`,
  :meth:`~polymath.Qube.tvl_gt`, and :meth:`~polymath.Qube.tvl_ge` are the comparisons,
  masked wherever either operand is masked.

Compare the ordinary ``==``, which treats a masked element as equal to another masked
element and unequal to anything else:

.. code-block:: python

    >>> Boolean([True, False], mask=[True, False]).tvl_and(False)
    Boolean(False False)
    >>> Boolean([True, True], mask=[True, False]).tvl_all()
    Boolean(--; mask)
    >>> Scalar([1., 2.], mask=[True, False]).tvl_eq(1.)
    Boolean(-- False; mask)
    >>> Scalar([1., 2.], mask=[True, False]) == 1.
    Boolean(False False)

Shrinking
=========

When most of an object is masked, the masked elements still cost time in every operation.
:meth:`~polymath.Qube.shrink` returns a one-dimensional, read-only copy holding only the
elements selected by an antimask, and :meth:`~polymath.Qube.unshrink` restores the
original shape afterward, masking everything the antimask excluded. A calculation
performed on shrunken objects gives the same result as one performed on the originals,
provided that every object involved is shrunk by the same antimask.

.. code-block:: python

    >>> big = Scalar(np.arange(10.)).mask_where(np.arange(10) % 2 == 0)
    >>> big
    Scalar(-- 1.0 -- 3.0 -- 5.0 -- 7.0 -- 9.0; mask)
    >>> small = big.shrink(big.antimask)
    >>> small
    Scalar(1. 3. 5. 7. 9.)
    >>> (small * 2).unshrink(big.antimask, big.shape)
    Scalar(-- 2.0 -- 6.0 -- 10.0 -- 14.0 -- 18.0; mask)
