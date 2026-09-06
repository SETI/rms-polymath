===========================
Type Stubs and Type Aliases
===========================

Overview
========

The rule for the source tree is that no module under ``src/`` carries type annotations,
with one exception described below. Parameter and return types belong in the docstrings,
where Napoleon renders them into the API reference. Public type information for
downstream type checkers is published separately, through stub files, and a check keeps
the two in agreement.

The Stub Files
==============

The only supported imports are ``from polymath import ...`` and
``from polymath.typedefs import ...``. A user never imports from a submodule such as
``polymath.scalar``, so the public type information lives in exactly two stubs:
``src/polymath/__init__.pyi``, which declares every public class in full, and
``src/polymath/typedefs.pyi``, which mirrors the aliases. The package ships a
``py.typed`` marker so that installed copies are recognized as typed. No other module
has a stub, and none may be added, because a per-module stub would make an import from
that module look supported.

A stub replaces its module entirely for a type checker: whatever the stub omits becomes
invisible to downstream code. The two stubs must therefore cover the whole public
surface, and adding, renaming, or re-signing any public member means updating
``__init__.pyi`` in the same change. Most of the methods of :class:`~polymath.Qube` are
bound from the extension modules at import time, and every one of them must appear under
the class in ``__init__.pyi`` as though it were defined in the class body. The stub
follows these conventions:

* The classes appear in dependency order: :class:`~polymath.Unit`,
  :class:`~polymath.Qube`, and then each subclass after its parent. Within a class the
  constants come first, then the methods in alphabetical order with the dunder methods
  first.
* Signature shapes are exact: every parameter, its keyword-only status, and whether it
  has a default, written as ``...``.
* Types come from the docstrings where those state one unambiguously, and are ``Any``
  where they do not. An ``Any`` is a deliberate statement that the docstring does not
  commit to a type, not an omission to fill in by guessing.
* Constructor arguments use the aliases from :mod:`polymath.typedefs`, which
  ``__init__.pyi`` imports from ``typedefs.pyi``. The two stubs import each other, which
  a type checker accepts.
* A method whose signature deliberately differs from its parent's carries a
  ``# type: ignore[override]`` comment.

The stubtest subcommand of mypy compares the stubs against the runtime API and fails on
any discrepancy. It runs in the check script and in CI:

.. code-block:: sh

   python -m mypy.stubtest polymath --mypy-config-file pyproject.toml --allowlist .stubtest-allowlist

Two pieces of configuration make this work, and both name the stub-less modules
explicitly, so a new module must be added to each. The ``exclude`` setting and the
per-module overrides under ``[tool.mypy]`` in ``pyproject.toml`` keep mypy from building
or following an import into the unannotated modules, which it would otherwise compare
against themselves. The allowlist in ``.stubtest-allowlist`` accepts the one finding
that remains for each of them, that no stub exists; it accepts nothing else, so a public
name missing from the two stubs still fails the check. Two further consequences for
authors: a module-level ``@classmethod`` in an extension module is not a function and
stubtest rejects it, so the binding module wraps such functions instead; and a method
stubtest cannot see at runtime, such as one bound conditionally, cannot appear in the
stub.

The Property Exception
======================

A property may carry an inline return annotation in the source. A property has no
parameters, and Sphinx renders the annotation as the property's type beside its name, so
that :attr:`~polymath.Qube.shape` reads as a property of type ``tuple[int, ...]``. A
property documented only through a ``Returns:`` block renders its type on a separate
trailing line instead, so the two styles do not mix: annotate the property and keep its
docstring to a one-line summary. Where the annotation names something from
:mod:`polymath.typedefs`, quote it and import it under ``if TYPE_CHECKING:``, because
that module imports :class:`~polymath.Qube` and a runtime import would be circular.

The Type Aliases
================

:mod:`polymath.typedefs` is the one module under ``src/`` that is written with
annotations, because its purpose is to define them. Each public alias names what the
corresponding constructor accepts: a :class:`~polymath.Qube`, a NumPy array with the
required trailing axes, a nested sequence, and for the rank-0 classes a single number.
The private aliases that build them describe NumPy arrays by their shape type. The
aliases are ordinary runtime objects, used both by the stubs and by downstream code.
``typedefs.pyi`` repeats the same definitions for the type checker, taking
:class:`~polymath.Qube` from the package rather than from its module, so that the two
stay in step by construction: an alias added to one must be added to the other, and
stubtest reports one that is not.

Sphinx documents each alias from the docstring that follows it. Sphinx 9 is the first
release whose Python domain resolves a class reference, which Napoleon generates for
every docstring type, to the data target that autodoc creates for a type alias, so the
``docs`` extra requires that version.

Checking the Tests
==================

The tests are fully annotated, and mypy in strict mode runs against ``tests/`` when the
``--mypy`` check is enabled. That run reads the stubs, so it is also the most realistic
test that the stubs describe an API a downstream project can use. Run it by hand with:

.. code-block:: sh

   MYPYPATH=src mypy tests
