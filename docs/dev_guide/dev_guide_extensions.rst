=====================
The Extension Modules
=====================

Overview
========

Every module under ``src/polymath/extensions/`` holds plain functions that the binding
module attaches to :class:`~polymath.Qube`, as described in
:doc:`dev_guide_architecture`. Together they are the implementation of the class. This
chapter describes each module's responsibility, the methods it supplies, and the
invariants it maintains, so that a change lands in the right place. The public methods
are documented under :class:`~polymath.Qube` in :doc:`/module`; the modules themselves,
with their private functions, are documented in :doc:`dev_guide_internal_api`.

Two of the modules are exceptions to the pattern. The iterator module defines two
classes, which are documented as :mod:`polymath.extensions.iterator`, and the pickler
module defines the compression functions as module-level functions, documented as
:mod:`polymath.extensions.pickler`, in addition to the methods it binds.

Module by Module
================

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Module
     - Responsibility
   * - ``attr_ops``
     - :meth:`~polymath.Qube.add_attr`, the only public function. It records the added
       name in the object's frozenset of added attributes, replacing the set rather than
       modifying it, because the class-level default is shared by every object.
   * - ``broadcaster``
     - :meth:`~polymath.Qube.broadcast_to`, :meth:`~polymath.Qube.broadcast`,
       :meth:`~polymath.Qube.broadcasted_shape`, and
       :meth:`~polymath.Qube.broadcast_into_shape`. Broadcasting applies to the leading
       axes only, and a broadcast object shares memory with its source, so it is returned
       read-only.
   * - ``casting``
     - Tests for a lone value (:meth:`~polymath.Qube.is_one_true`,
       :meth:`~polymath.Qube.is_one_false`, :meth:`~polymath.Qube.as_one_bool`) and
       conversions between classes (:meth:`~polymath.Qube.cast`,
       :meth:`~polymath.Qube.as_this_type`, :meth:`~polymath.Qube.as_all_constant`,
       :meth:`~polymath.Qube.as_size_zero`). Conversion consults the class constants of
       the target and the derivative class of the source.
   * - ``deriv_ops``
     - Insertion, deletion, and renaming of derivatives, and the copies with and without
       them. Insertion broadcasts the derivative to the object's shape, converts it to the
       derivative class, makes it read-only if the object is, and sets the ``d_d``
       attribute. Every other module that touches derivatives goes through these
       functions.
   * - ``dtypes``
     - Interpretation of an arbitrary constructor argument as a data type, a value, and
       a mask; the checks :meth:`~polymath.Qube.is_float`, :meth:`~polymath.Qube.is_int`,
       and :meth:`~polymath.Qube.is_bool`; and the conversions
       :meth:`~polymath.Qube.as_float`, :meth:`~polymath.Qube.as_int`, and
       :meth:`~polymath.Qube.as_bool`. Its three helpers that take the class as their
       first argument are wrapped in ``classmethod`` at the binding site.
   * - ``errors``
     - The functions that raise the shared exceptions, so that every message names the
       operation, the classes, and the shapes the same way, along with the preconditions
       many operations begin with, such as requiring an object to have no denominator or
       to be a :class:`~polymath.Scalar`.
   * - ``indexer``
     - :meth:`~polymath.Qube.__getitem__` and :meth:`~polymath.Qube.__setitem__`. The
       index is first normalized: a :class:`~polymath.Boolean` or an integer
       :class:`~polymath.Qube` becomes a NumPy index plus a mask, a
       :class:`~polymath.Pair` or :class:`~polymath.Vector` becomes an index into
       consecutive axes, and the rule that places the broadcasted shape of several
       array indices at the position of the first one is applied here. Assignment
       requires a writable object and leaves elements selected by a masked index
       unchanged.
   * - ``item_ops``
     - Restructuring of the item axes: extraction, slicing, reshaping, flattening, and
       transposition of the numerator and denominator separately, joining and splitting
       of the two, :meth:`~polymath.Qube.chain`, and the ``@`` operator. These are how
       a derivative's denominator is manipulated and how an object is reinterpreted as
       another class.
   * - ``iterator``
     - :meth:`~polymath.Qube.__iter__`, which walks the first axis, and
       :meth:`~polymath.Qube.ndenumerate`, which walks every item with its index.
   * - ``masking``
     - Conversion of an argument into a mask of suitable shape, the combining functions
       :meth:`~polymath.Qube.or_` and :meth:`~polymath.Qube.and_`, the counts, and the
       copies with a replaced, removed, expanded, or collapsed mask. A mask is a Python
       bool or a boolean array of the object's shape, never anything else, and this
       module is where that is enforced.
   * - ``mask_ops``
     - The ``mask_where`` family, :meth:`~polymath.Qube.clip`, and the range tests
       :meth:`~polymath.Qube.is_inside`, :meth:`~polymath.Qube.is_outside`,
       :meth:`~polymath.Qube.is_above`, and :meth:`~polymath.Qube.is_below`. Each masks
       elements by value and optionally replaces them.
   * - ``math_ops``
     - The unary, binary, in-place, and reflected arithmetic operators, the comparison
       and logical operators, :meth:`~polymath.Qube.__bool__`,
       :meth:`~polymath.Qube.__float__`, :meth:`~polymath.Qube.__int__`, and the
       reductions :meth:`~polymath.Qube.sum`, :meth:`~polymath.Qube.mean`,
       :meth:`~polymath.Qube.any`, and :meth:`~polymath.Qube.all`. It is the largest
       module and the one described under operator dispatch in
       :doc:`dev_guide_architecture`. The binding module sets the hash to None
       immediately after binding the equality operator from here.
   * - ``pickler``
     - :meth:`~polymath.Qube.__getstate__` and :meth:`~polymath.Qube.__setstate__`, the
       encoders and decoders for float, integer, and boolean arrays, and the precision
       settings. Only the unmasked elements are stored; masked elements are restored
       from the default value.
   * - ``readonly_ops``
     - :meth:`~polymath.Qube.as_readonly`, :meth:`~polymath.Qube.require_writeable`,
       :meth:`~polymath.Qube.match_readonly`, :meth:`~polymath.Qube.copy`, and
       :meth:`~polymath.Qube.__copy__`. Read-only status is implemented by clearing the
       writable flag of the values array and the mask array, so a determined caller can
       defeat it; the API only makes modification difficult.
   * - ``shaper``
     - :meth:`~polymath.Qube.reshape`, :meth:`~polymath.Qube.flatten`,
       :meth:`~polymath.Qube.swap_axes`, :meth:`~polymath.Qube.roll_axis`,
       :meth:`~polymath.Qube.move_axis`, and :meth:`~polymath.Qube.stack`. Each applies
       the same change to the derivatives, and each returns a view where NumPy can
       provide one.
   * - ``shrinker``
     - :meth:`~polymath.Qube.shrink` and :meth:`~polymath.Qube.unshrink`, including the
       caching of each other's results.
   * - ``tvl``
     - The three-valued logic operations, in which a masked value means "maybe".
   * - ``unit_ops``
     - :meth:`~polymath.Qube.set_unit`, :meth:`~polymath.Qube.without_unit`,
       :meth:`~polymath.Qube.into_unit`, :meth:`~polymath.Qube.confirm_unit`,
       :meth:`~polymath.Qube.is_unitless`, and the private checks that an operation's
       operands have compatible units, that an argument is an angle, or that an object
       is unitless.
   * - ``vector_ops``
     - :meth:`~polymath.Qube.dot`, :meth:`~polymath.Qube.norm`,
       :meth:`~polymath.Qube.norm_sq`, :meth:`~polymath.Qube.cross`,
       :meth:`~polymath.Qube.outer`, :meth:`~polymath.Qube.as_diagonal`, and
       :meth:`~polymath.Qube.rms`, each operating on a chosen pair of item axes. They
       are defined here rather than on :class:`~polymath.Vector` because they apply to
       any object whose item axes have suitable lengths, and the
       :class:`~polymath.Vector` methods of the same names are thin wrappers that fix
       the axes and the result class.

Invariants Every Extension Must Respect
=======================================

* **Values are in standard units.** Never scale values by a unit inside an operation.
  A unit is checked for compatibility or combined, and the values are left alone.
* **Masks combine with logical or.** The mask of a result is the union of the masks of
  the operands, plus whatever the operation itself could not compute. Use the combining
  functions in the masking module rather than NumPy directly, because a mask may be a
  single bool.
* **Derivatives are propagated unless ``recursive`` is False.** A method that produces a
  new value and takes a ``recursive`` keyword must propagate every derivative when it is
  True and produce an object with no derivatives when it is False. Insert derivatives
  through the derivative operations module so that broadcasting, class conversion, and
  read-only status are handled once.
* **Shared memory is read-only.** An operation whose result shares memory with its
  input, such as a reshape, a broadcast, or a shrink, returns a read-only object. An
  operation that computes fresh values may return a writable one.
* **Never modify an operand.** Operations return new objects; only the in-place operators
  and the methods documented as in-place modify their receiver, and those call
  :meth:`~polymath.Qube.require_writeable` first.
* **Clear or prune the cache.** Anything that changes an object clears its cache, and a
  copy that retains the cache drops the entries that describe the original.
* **Shrinking is transparent.** The result of an operation must not depend on whether
  its operands were shrunk, and the switches described in :doc:`dev_guide_architecture`
  must leave every test passing.
* **Reach subclasses through the registry.** Use ``Qube._SCALAR_CLASS`` and its
  siblings rather than importing a subclass module.

API Reference
=============

The public methods are listed under :class:`~polymath.Qube` in :doc:`/module`. The
extension modules themselves, including their private functions, are in
:doc:`dev_guide_internal_api`.
