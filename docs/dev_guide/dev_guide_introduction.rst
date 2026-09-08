============
Introduction
============

Who This Guide Is For
=====================

This guide is for anyone who changes PolyMath: fixing a bug, adding a method, defining a
subclass, or cutting a release. It assumes a competent Python developer who is fluent with
NumPy and pytest but new to this code base. The :doc:`/user_guide/user_guide` explains how
to use the package and never asks its reader to open the source. This guide is about the
source, and it concentrates on the contracts that hold the package together rather than
restating what the code says.

Package Overview
================

PolyMath, distributed as ``rms-polymath`` and imported as :mod:`polymath`, wraps NumPy
arrays in objects that keep the shape of an array separate from the shape of its items,
and that carry a mask, an optional :class:`~polymath.Unit`, and optional derivatives
through every operation. One base class, :class:`~polymath.Qube`, implements all of that
machinery. The nine public subclasses, :class:`~polymath.Scalar`,
:class:`~polymath.Boolean`, :class:`~polymath.Vector`, :class:`~polymath.Pair`,
:class:`~polymath.Vector3`, :class:`~polymath.Quaternion`, :class:`~polymath.Polynomial`,
:class:`~polymath.Matrix`, and :class:`~polymath.Matrix3`, add constraints on the item
shape and the operations that make sense for it.

Two design decisions shape everything a developer touches, and both are explained in
:doc:`dev_guide_architecture`:

* The file for :class:`~polymath.Qube` holds only what defines an object. Nearly every
  method is written as a plain function in a module under ``src/polymath/extensions/``
  and bound onto the class when the package is imported.
* The source carries no type annotations. Public type information lives in two stub
  files, ``__init__.pyi`` and ``typedefs.pyi``, matching the only two supported import
  paths, and a check called stubtest keeps them honest.

Runtime requirements are Python 3.11 or later, NumPy 2.0 or later, and ``rms-fpzip``,
which compresses floating-point arrays when objects are pickled. The package runs on
Linux, macOS, and Windows, and the test matrix covers all three.

Development requirements are declared as extras in ``pyproject.toml``. The ``dev`` extra
brings the linters, the test tools, the packaging check, and the type checker whose
stubtest subcommand validates the stubs; it also pulls in the ``docs`` extra, which
brings Sphinx 9 or later and its extensions.

Where to Look
=============

* :doc:`/module` is the public API reference, generated from the docstrings.
* :doc:`dev_guide_internal_api` is a second copy of the API reference that includes
  the private methods and the extension modules, generated from the same docstrings.
* :doc:`/contributing` is the contribution guide, covering issues, enhancement requests,
  and the legal terms of a code contribution.
* ``CLAUDE.md`` and the files under ``.claude/rules/`` in the repository root are the
  detailed working rules, written for AI-assisted development but binding on everyone.
  :doc:`dev_guide_conventions` summarizes them.
