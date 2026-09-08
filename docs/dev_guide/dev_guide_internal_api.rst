======================
Internal API Reference
======================

This page is a second copy of the API reference, generated from the same docstrings as
:doc:`/module` but including the private methods, the private module-level helpers, the
special methods, and the class constants. It also documents each extension module as a
module, so that a function can be found where it is defined as well as under the name it
is bound to on :class:`~polymath.Qube`.

Nothing on this page is part of the public API. A name beginning with an underscore may
change without notice. The class entries here duplicate the public ones and are not
indexed, so a cross-reference to a class or method always resolves to :doc:`/module`.

The Base Class
==============

.. automodule:: polymath.qube
    :no-members:

.. autoclass:: polymath.Qube
    :no-index:
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__, __abstractmethods__

The Extension Modules
=====================

The methods that the modules below define are bound onto :class:`~polymath.Qube` and are
listed under the base class above. Here they appear as the module-level functions they
are, together with each module's private helpers.

.. automodule:: polymath.extensions
    :members:

.. automodule:: polymath.extensions.attr_ops
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

.. automodule:: polymath.extensions.broadcaster
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

.. automodule:: polymath.extensions.casting
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

.. automodule:: polymath.extensions.deriv_ops
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

.. automodule:: polymath.extensions.dtypes
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

.. automodule:: polymath.extensions.errors
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

.. automodule:: polymath.extensions.indexer
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:

.. automodule:: polymath.extensions.item_ops
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:

The iterator module is documented in full as :mod:`polymath.extensions.iterator`.

.. automodule:: polymath.extensions.mask_ops
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

.. automodule:: polymath.extensions.masking
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

.. automodule:: polymath.extensions.math_ops
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:

The public functions of the pickler module are documented as
:mod:`polymath.extensions.pickler`; its private encoders and decoders are listed under
the base class above.

.. automodule:: polymath.extensions.readonly_ops
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:

.. automodule:: polymath.extensions.shaper
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

.. automodule:: polymath.extensions.shrinker
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

.. automodule:: polymath.extensions.tvl
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

.. automodule:: polymath.extensions.unit_ops
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

.. automodule:: polymath.extensions.vector_ops
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:

The Subclasses
==============

Each class is listed with the members it defines itself; inherited members appear under
the parent class.

.. autoclass:: polymath.Scalar
    :no-index:
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__, __abstractmethods__

.. autoclass:: polymath.Boolean
    :no-index:
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__, __abstractmethods__

.. autoclass:: polymath.Vector
    :no-index:
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__, __abstractmethods__

.. autoclass:: polymath.Pair
    :no-index:
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__, __abstractmethods__

.. autoclass:: polymath.Vector3
    :no-index:
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__, __abstractmethods__

.. autoclass:: polymath.Quaternion
    :no-index:
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__, __abstractmethods__

.. autoclass:: polymath.Polynomial
    :no-index:
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__, __abstractmethods__

.. autoclass:: polymath.Matrix
    :no-index:
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__, __abstractmethods__

.. autoclass:: polymath.Matrix3
    :no-index:
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__, __abstractmethods__

The Unit Class
==============

.. autoclass:: polymath.Unit
    :no-index:
    :member-order: bysource
    :members:
    :undoc-members:
    :private-members:
    :special-members:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__, __abstractmethods__
