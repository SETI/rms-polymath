====================
Extending the System
====================

This chapter gives a step-by-step recipe for each of the three kinds of addition: a
method available on every class, a method on one subclass, and a subclass of its own.
Each recipe ends with the same checklist, because every addition touches the same set of
files: the implementation, the stub, the tests, and the documentation. The contracts the
recipes rely on are described in :doc:`dev_guide_architecture`,
:doc:`dev_guide_extensions`, and :doc:`dev_guide_subclasses`; this chapter does not
repeat them.

Adding a Method to Every Class
==============================

A method that applies to any :class:`~polymath.Qube` is written as a plain function in
an extension module and bound onto the class. The example adds a method that repeats an
object along a leading axis.

1. **Choose the module.** Put the function in the extension module that owns the
   concern; a method that rearranges the leading axes belongs in
   ``src/polymath/extensions/shaper.py``. Create a module only for a concern none of the
   existing ones covers, and give it the same header, docstring, and ``__all__`` as its
   neighbors.

2. **Write the function.** Its first parameter is ``self``. Read the private attributes
   directly, build the result with the fast construction path, and propagate the
   derivatives yourself.

   .. code-block:: python

      def tile(self, reps, *, recursive=True):
          """Repeat this object along a new leading axis.

          Parameters:
              reps (int): The number of repetitions.
              recursive (bool, optional): True to tile the derivatives as well; False to
                  return an object without derivatives.

          Returns:
              Qube: An object of shape ``(reps,) + self.shape``, sharing no memory with
              this one.

          Raises:
              ValueError: If `reps` is less than one.
          """

          if reps < 1:
              raise ValueError(f'invalid repetition count for '
                               f'{type(self).__name__}.tile(): {reps}')

          values = np.repeat(np.asarray(self._values)[np.newaxis], reps, axis=0)
          mask = self._mask
          if isinstance(mask, np.ndarray):
              mask = np.repeat(mask[np.newaxis], reps, axis=0)

          obj = type(self)._new_from_parts(values, mask, nrank=self._nrank,
                                           drank=self._drank, unit=self._unit,
                                           example=self)

          if recursive:
              for key, deriv in self._derivs.items():
                  obj.insert_deriv(key, deriv.tile(reps, recursive=False))

          return obj

   Add the name to the module's ``__all__``. Note what the function does not do: it
   does not import a subclass, it does not scale the values by the unit, and it does not
   modify ``self``.

3. **Bind it.** Add one line to ``src/polymath/extensions/__init__.py``, in the block for
   the module, keeping the aligned assignment style:

   .. code-block:: python

      Qube.tile               = shaper.tile

4. **Declare it in the stub.** Add the signature under :class:`~polymath.Qube` in
   ``src/polymath/__init__.pyi``, in alphabetical order, taking the types from the
   docstring:

   .. code-block:: python

      def tile(self, reps: int, *, recursive: bool = ...) -> Qube: ...

5. **Test it.** Add tests to the file for the class and topic, here
   ``tests/test_qube_reshaping.py``, or start a new file named the same way. Each test
   is annotated, independent, and asserts exact values:

   .. code-block:: python

      def test_tile_repeats_values_and_mask() -> None:
          """Tiling repeats the values and the mask along a new leading axis."""

          np.random.seed(1234)
          x = Scalar(np.random.randn(3), mask=[False, True, False])
          tiled = x.tile(2)
          assert tiled.shape == (2, 3)
          assert np.all(tiled.values == np.stack([x.values, x.values]))
          assert np.all(tiled.mask == np.stack([x.mask, x.mask]))


      def test_tile_propagates_derivatives() -> None:
          """Derivatives are tiled alongside the values unless recursive is False."""

          x = Vector3([1., 2., 3.])
          x.insert_deriv('t', Vector3([0., 0., 1.]))
          assert x.tile(4).d_dt.shape == (4,)
          assert not hasattr(x.tile(4, recursive=False), 'd_dt')

6. **Document it.** The method appears in :doc:`/module` automatically through the
   docstring. Mention it in the relevant chapter of the user guide, and in the module's
   entry in :doc:`dev_guide_extensions` if it changes what the module is responsible
   for.

Adding a Method to One Subclass
===============================

A method that makes sense for one class only, such as a new coordinate conversion for
:class:`~polymath.Vector3`, is written in the subclass module as an ordinary method.

1. Write the method in the class body of ``src/polymath/<class>.py``, with a docstring
   and the ``recursive`` keyword if it produces a value with derivatives. A method that
   the subclass shares with a sibling belongs on their common parent instead.
2. Add the signature to the class in ``src/polymath/__init__.pyi``, in alphabetical
   order. Never create a stub beside the module: the only supported import is from the
   package, and a per-module stub would make an import from the module look supported.
3. If the method changes the meaning of an operator inherited from
   :class:`~polymath.Qube`, say so in the docstring with the sentence the existing
   overrides use, and give the override the same signature as the base method, adding
   ``# type: ignore[override]`` in the stub only when the signature must differ.
4. Add tests to ``tests/test_<class>_<topic>.py``.
5. Mention the method in the user guide chapter for the class.

Adding a Subclass
=================

A subclass fixes the constraints on an item and adds methods. The example defines a
4-vector; the steps are the same for a class with any item shape.

1. **Create the module** ``src/polymath/vector4.py`` following the pattern of the
   sibling modules. Set all seven constraint constants, define the converter, build the
   constants after the class body, and register the class.

   .. code-block:: python

      ##########################################################################################
      # polymath/vector4.py: Vector4 subclass of PolyMath Vector
      ##########################################################################################
      """The :class:`~polymath.Vector4` subclass, representing 4-vectors."""

      import numpy as np
      import numbers

      from polymath.qube   import Qube
      from polymath.vector import Vector

      __all__ = ['Vector4']


      class Vector4(Vector):
          """Represent 4-vectors in the PolyMath framework."""

          _NRANK = 1          # The number of numerator axes.
          _NUMER = (4,)       # Shape of the numerator.
          _FLOATS_OK = True   # True to allow floating-point numbers.
          _INTS_OK = True     # True to allow integers.
          _BOOLS_OK = False   # True to allow booleans.
          _UNITS_OK = True    # True to allow units; False to disallow them.
          _DERIVS_OK = True   # True to allow derivatives and denominators; False to disallow.
          _DEFAULT_VALUE = np.array([1, 1, 1, 1])

          @staticmethod
          def as_vector4(arg, *, recursive=True):
              """Convert the argument to Vector4 if possible.

              Parameters:
                  arg (Any): The object to convert.
                  recursive (bool, optional): If True, derivatives are also converted.

              Returns:
                  Vector4: The converted object.

              Notes:
                  A single number is repeated in all four components.
              """

              if isinstance(arg, Vector4):
                  return arg if recursive else arg.wod

              if isinstance(arg, Qube):
                  if arg._numer in ((1, 4), (4, 1)):
                      return arg.flatten_numer(Vector4, recursive=recursive)
                  if arg.rank > 1 and arg._numer[0] == 4:
                      arg = arg.split_items(1, Vector4)
                  arg = Vector4(arg._values, arg._mask, example=arg)
                  return arg if recursive else arg.wod

              if isinstance(arg, numbers.Real):
                  return Vector4((arg, arg, arg, arg))

              return Vector4(arg)


      # Read-only constants, built after the class exists so that the bound methods are
      # available, and shared safely because they cannot be modified
      Vector4.ZERO   = Vector4((0., 0., 0., 0.)).as_readonly()
      Vector4.ONES   = Vector4((1., 1., 1., 1.)).as_readonly()
      Vector4.MASKED = Vector4((1, 1, 1, 1), True).as_readonly()

      # Register the class so that the extension modules can reach it without importing it
      Qube._VECTOR4_CLASS = Vector4

      ##########################################################################################

   Decide the derivative class. If a derivative of the class satisfies the same
   constraints, leave the inherited None; if it does not, as for a rotation matrix,
   name the more general class. Decide the default value, which masked elements take on
   after unpickling, and choose one that does not break arithmetic.

2. **Export it.** In ``src/polymath/__init__.py``, add the import to the block of
   subclass imports, which comes after the import of the extensions package, and add
   the name to ``__all__``. Then add the module to the ``exclude`` list and the override
   list under ``[tool.mypy]`` in ``pyproject.toml`` and to ``.stubtest-allowlist``, so
   that stubtest treats it like its siblings; :doc:`dev_guide_typing` explains why.

3. **Declare it in the stub.** In ``src/polymath/__init__.pyi``, add the name to
   ``__all__`` and add the class after its parent, following the sibling classes: the
   constants typed as the class, and every public method in alphabetical order. Do not
   create ``vector4.pyi``.

   .. code-block:: python

      class Vector4(Vector):
          MASKED: Vector4
          ONES: Vector4
          ZERO: Vector4
          @staticmethod
          def as_vector4(arg: Any, *, recursive: bool = ...) -> Vector4: ...

4. **Add a type alias**, if downstream code will annotate parameters that accept the
   class. In ``src/polymath/typedefs.py``, define the array alias for the item shape,
   define the public alias with a docstring, and add its name to ``__all__``; then make
   the same additions to ``src/polymath/typedefs.pyi``.

5. **Test it.** Create ``tests/test_vector4_basic.py`` and cover construction from each
   kind of input, rejection of the wrong item shape with ``pytest.raises`` and a
   ``match=``, the converter's every branch, the constants, and any methods. Run the full
   suite, because the change to the package namespace affects every test.

6. **Document it.** The class appears in :doc:`/module` through the ``__all__`` of the
   package. Add a module entry to :doc:`dev_guide_internal_api`, a row to the class table
   in the user guide introduction, a paragraph to :doc:`dev_guide_subclasses`, and a
   bullet to the feature list in ``README.md``.

Checklist
=========

Before opening a pull request for any of the above:

* The docstring is complete enough to write a black-box test from, and uses
  ``Parameters:``.
* The stub matches the implementation, and stubtest passes.
* The tests are annotated, independent, and assert exact values, and coverage has not
  dropped.
* The user guide and this guide say what changed, and the documentation builds with no
  warnings.
* ``./scripts/run-all-checks.sh`` passes.
