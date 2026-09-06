====================
Pickling and Storage
====================

PolyMath objects pickle with the standard :mod:`pickle` module, and the package takes the
opportunity to compress them. Objects such as image backplanes are numerous and large, so
the savings matter.

.. code-block:: python

    >>> import pickle
    >>> import numpy as np
    >>> from polymath import Qube, Scalar
    >>> x = Scalar(np.linspace(0., 1., 1000).reshape(10, 100))
    >>> y = pickle.loads(pickle.dumps(x))
    >>> (y == x).all()
    Boolean(True)

Only the unmasked elements are stored. After unpickling, the masked elements hold the
object's :attr:`~polymath.Qube.default` value:

.. code-block:: python

    >>> z = pickle.loads(pickle.dumps(Scalar([1., 2., 3.], mask=[False, True, False])))
    >>> z
    Scalar(1.0 -- 3.0; mask)
    >>> z.values
    array([1., 1., 3.])

How Values Are Compressed
=========================

Integer arrays are compressed losslessly with BZ2 after being reduced to the fewest bytes
that cover their range. Arrays of truth values are packed into bits and then compressed
with BZ2.
Floating-point arrays are handled in one of four ways:

1. Very small arrays are compressed with BZ2.
2. Constant arrays are stored as a single value plus a shape.
3. Values are divided by a constant, rounded to integers, and compressed as integers.
4. Values are compressed, with or without loss, by fpzip, which is especially effective
   for arrays such as backplanes that vary smoothly from pixel to pixel. See
   https://pypi.org/project/rms-fpzip.

Choosing the Precision
======================

:meth:`~polymath.Qube.set_pickle_digits` sets the floating-point precision for one object,
and :meth:`~polymath.Qube.set_default_pickle_digits` sets the default for every object
that has no setting of its own. Both take the same two arguments, and
:meth:`~polymath.Qube.pickle_digits` and :meth:`~polymath.Qube.pickle_reference` report
the values in effect for an object. The built-in default preserves full double precision
with lossless fpzip compression.

**digits** (``str``, ``int``, or ``float``): The number of digits to preserve.

* ``"double"``: preserve full precision using lossless fpzip compression.
* ``"single"``: convert the array to single precision and then store it using lossless
  fpzip compression.
* A number from 7 to 16, defining the number of significant digits to preserve.

**reference** (``str`` or ``float``): How to interpret a numeric value of **digits**.

* ``"fpzip"``: Use lossy fpzip compression, preserving the given number of digits.
* A number: Preserve every value to the same absolute precision, obtained by scaling the
  number of **digits** by this value. For example, if **digits** is 8 and **reference** is
  100, every value is rounded to the nearest 1.e-6 before storage. This uses the third
  method above, in which values are converted to integers for storage.
* ``"smallest"``: The absolute precision is ``10**(-digits)`` times the nonzero value
  closest to zero. This guarantees that every value preserves at least the requested
  number of digits, and it is reasonable when all values fall within a similar dynamic
  range.
* ``"largest"``: The absolute precision is ``10**(-digits)`` times the value furthest from
  zero. This suits arrays with a limited range of values, such as the components of a
  unit vector or angles known to fall between zero and two pi, where the extra precision
  of values that happen to fall close to zero is not needed.
* ``"mean"``: The absolute precision is ``10**(-digits)`` times the mean of the absolute
  values.
* ``"median"``: The absolute precision is ``10**(-digits)`` times the median of the
  absolute values. This is a good choice when a minority of values differ greatly from
  the rest, such as noise spikes or undefined geometry, so that the precision is based on
  the typical values.
* ``"logmean"``: The absolute precision is ``10**(-digits)`` times the logarithmic mean
  of the absolute values.

Either argument can also be a tuple of two values, in which case the second applies to
the derivatives of the object.

.. code-block:: python

    >>> x.set_pickle_digits(8, 'fpzip')
    >>> x.pickle_digits()
    (8.0, 8.0)
    >>> x.pickle_reference()
    ('fpzip', 'fpzip')
    >>> Qube.set_default_pickle_digits('double', 'fpzip')
