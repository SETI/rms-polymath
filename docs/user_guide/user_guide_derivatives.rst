===========
Derivatives
===========

A PolyMath object can carry derivatives, each of which is another PolyMath object of the
same shape. Every operator and math function propagates them, applying the chain rule as
it goes, so an algorithm written once in terms of positions yields velocities for free
when the positions carry a time derivative.

Attaching a Derivative
======================

:meth:`~polymath.Qube.insert_deriv` attaches a derivative under a name. The
:attr:`~polymath.Qube.derivs` property is the dictionary of all of them, and each is also
available as an attribute named ``d_d`` followed by the name, so the derivative with
respect to ``t`` is ``d_dt``. Each derivative appears as a suffix in the printed form.

.. code-block:: python

    >>> import numpy as np
    >>> from polymath import Pair, Scalar, Vector3
    >>> t = Scalar([0., 1., 2.])
    >>> x = t ** 2
    >>> x.insert_deriv('t', 2 * t)
    Scalar(0. 1. 4.; d_dt)
    >>> x.d_dt
    Scalar(0. 2. 4.)
    >>> list(x.derivs)
    ['t']

Propagation
===========

Operations apply the chain rule. The derivative of the sine of ``x`` with respect to
``t`` is the cosine of ``x`` times the derivative of ``x``, and the product rule applies
to a product:

.. code-block:: python

    >>> x.sin().d_dt
    Scalar( 0.          1.08060461 -2.61457448)
    >>> (x * x).d_dt
    Scalar( 0.  4. 32.)

The same holds for vectors. If a position carries a velocity, every quantity derived from
it carries its own rate of change:

.. code-block:: python

    >>> pos = Vector3([3., 4., 0.])
    >>> pos.insert_deriv('t', Vector3([1., 0., 0.]))
    Vector3(3. 4. 0.; d_dt)
    >>> pos.norm().d_dt
    Scalar(0.6)
    >>> pos.unit().d_dt
    Vector3( 0.128 -0.096  0.   )

Derivatives cost time. Most methods take a ``recursive`` keyword, True by default; pass
``recursive=False`` to compute the value alone. :attr:`~polymath.Qube.wod`, short for
"without derivatives", returns a shallow copy with no derivatives, which is the cheapest
way to drop them for the rest of a calculation.

.. code-block:: python

    >>> x.sin(recursive=False)
    Scalar( 0.          0.84147098 -0.7568025 )
    >>> x.wod
    Scalar(0. 1. 4.)

Partial Derivatives and Denominators
====================================

The derivative of a vector with respect to a scalar is a vector with the same item shape.
The derivative of a vector with respect to another vector has more components: the partial
derivatives of a :class:`~polymath.Vector3` with respect to a :class:`~polymath.Pair` form
a 3x2 array of numbers. PolyMath represents this by splitting the item axes into a
numerator, which is the item shape of the quantity being differentiated, and a
denominator, which is the item shape of the variable. The :attr:`~polymath.Qube.numer`,
:attr:`~polymath.Qube.denom`, :attr:`~polymath.Qube.nrank`, and
:attr:`~polymath.Qube.drank` properties describe the split. Construct such an object with
the ``drank`` argument, which states how many trailing axes belong to the denominator.

.. code-block:: python

    >>> dpos_duv = Vector3([[1., 0.], [0., 1.], [0., 0.]], drank=1)
    >>> dpos_duv
    Vector3([[1. 0.]
     [0. 1.]
     [0. 0.]]; denom=(2,))
    >>> dpos_duv.numer, dpos_duv.denom, dpos_duv.item
    ((3,), (2,), (3, 2))

A class constrains only the numerator, so this object is still a
:class:`~polymath.Vector3`, and the derivative of a :class:`~polymath.Vector3` with respect
to anything can be attached to one. Propagation keeps the denominator:

.. code-block:: python

    >>> pos = Vector3([1., 2., 3.])
    >>> pos.insert_deriv('uv', dpos_duv)
    Vector3(1. 2. 3.; d_duv)
    >>> pos.norm().d_duv
    Scalar([0.26726124 0.53452248]; denom=(2,))

:meth:`~polymath.Qube.chain` multiplies derivatives together, contracting the denominator
of one against the numerator of the next; the ``@`` operator does the same. Given the
derivative of a position with respect to a coordinate pair and the derivative of that pair
with respect to time, the chain gives the derivative of the position with respect to time:

.. code-block:: python

    >>> duv_dt = Pair([1., 2.])
    >>> dpos_duv @ duv_dt
    Vector3(1. 2. 0.)

Managing Derivatives
====================

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Method
     - Effect
   * - :meth:`~polymath.Qube.insert_deriv`, :meth:`~polymath.Qube.insert_derivs`
     - Attach one derivative, or a dictionary of them, in place.
   * - :meth:`~polymath.Qube.delete_deriv`, :meth:`~polymath.Qube.delete_derivs`
     - Remove one derivative, or all of them, in place.
   * - :meth:`~polymath.Qube.rename_deriv`
     - Rename a derivative in place.
   * - :meth:`~polymath.Qube.with_deriv`, :meth:`~polymath.Qube.without_deriv`,
       :meth:`~polymath.Qube.without_derivs`, :attr:`~polymath.Qube.wod`
     - Return a shallow copy with a derivative added or removed.

The item axes of a derivative can be rearranged with
:meth:`~polymath.Qube.extract_numer`, :meth:`~polymath.Qube.extract_denom`,
:meth:`~polymath.Qube.extract_denoms`, :meth:`~polymath.Qube.slice_numer`,
:meth:`~polymath.Qube.transpose_numer`, :meth:`~polymath.Qube.reshape_numer`,
:meth:`~polymath.Qube.flatten_numer`, :meth:`~polymath.Qube.transpose_denom`,
:meth:`~polymath.Qube.reshape_denom`, :meth:`~polymath.Qube.flatten_denom`,
:meth:`~polymath.Qube.join_items`, :meth:`~polymath.Qube.split_items`, and
:meth:`~polymath.Qube.swap_items`.
