# Code critique: rms-polymath — 2026-08-09

Scope: the whole of `src/polymath/` (25 modules, ~19,600 lines) on branch `mark-reorg` at
commit `5f46fcc`. Configuration (`pyproject.toml`, `.github/workflows/`,
`scripts/run-all-checks.sh`) reviewed for consistency. The test suite was not audited in
depth — use the `critique-test-suite` skill for that; likewise the documentation, which has
its own `critique-documentation` skill.

Every finding tagged **[confirmed]** was reproduced by running code against the project
virtualenv (NumPy 2.5.2, Python 3.12). Findings without that tag come from reading and are
stated as such.

## Summary

The library is in good shape structurally: the class hierarchy is coherent, the
`Qube`/extension split keeps individual files navigable, docstrings are thorough, and the
mask/derivative/unit machinery is carefully thought through. The problems are concentrated
in three places.

1. **A handful of genuine defects reachable from the public API**, most of them in code
   paths that the 90% coverage floor does not exercise: an `AttributeError` from a
   misspelled method name, an `UnboundLocalError` in `__setitem__`, `Matrix.inverse()`
   silently mutating its input, `Scalar.sort()` discarding the mask it just computed, and
   several aliasing bugs where two objects end up sharing one `_derivs` or `_cache` dict.
2. **Two performance problems that dominate everything else.** `Qube.__init__` runs on
   every arithmetic result and spends ~32% of its time in four `np.prod()` calls on
   tuples; and `_prep_index` builds a `set(range(axis_length))` on *every* array-index
   operation, making indexing O(size of the axis being indexed) rather than O(number of
   indices). The latter costs 33 ms to pull 100 elements out of a million-element Scalar.
3. **Consistency drift** — `recursive=` ignored in one method, `Vector` and `Scalar`
   disagreeing on keyword-only-ness, dead code that was never wired up, and several
   `TODO`/`XXX` markers recording unresolved correctness questions in shipped code.

Top three priorities: fix the confirmed defects in §1; make the two performance changes in
§5.1 and §5.2 (both are small, local, and together should be worth several-fold on hot
paths); then close the aliasing class of bug in §1.5, which is the one most likely to
produce a mystifying downstream failure.

---

## 1. Confirmed defects

> **Status: all of §1 was fixed on 2026-08-09**, each item with a regression test that
> fails against the previous code. Everything from §2 onward is unchanged and still open.

### 1.1 `Qube.__pow__` calls a method that does not exist — **critical** [confirmed]

`src/polymath/extensions/math_ops.py:1155`

```python
if arg._mask:
    return self.as_fully_masked(recursive=True)
```

There is no `as_fully_masked` anywhere in `src/`; the method is called `as_all_masked`
(`qube.py:2696`). Raising a non-`Scalar` Qube to a masked exponent therefore raises
`AttributeError` instead of returning a masked object:

```python
>>> Matrix3.IDENTITY ** Scalar.MASKED
AttributeError: 'Matrix3' object has no attribute 'as_fully_masked'
```

`Scalar` overrides `__pow__`, so only the non-scalar classes (Matrix, Matrix3, Quaternion,
Vector) are affected — which is exactly why no test caught it.

**Fix**: rename to `as_all_masked`. Add a test that raises a Matrix3 to `Scalar.MASKED`.

### 1.2 `Matrix.inverse()` mutates its input — **critical** [confirmed]

`src/polymath/matrix.py:357-363`

```python
mask = (det == 0.)
if np.any(mask):
    self._values[mask] = np.diag(np.ones(self._numer[0]))   # writes into self!
    new_mask = Qube.or_(self._mask, mask)
```

When any matrix in the array is singular, the *caller's* object is overwritten in place
with identity matrices. The caller gets a correct answer and a silently corrupted input:

```python
>>> m = Matrix([[[1.,0.],[0.,1.]], [[0.,0.],[0.,0.]]])
>>> m.inverse()
>>> m.values[1]        # was the zero matrix
array([[1., 0.], [0., 1.]])
```

On a read-only Matrix this instead raises a NumPy "assignment destination is read-only"
error, so the failure mode depends on the object's read-only status.

**Fix**: work on a copy — `values = self._values.copy()` before the assignment — and build
the result from that. Note the same pattern would need checking anywhere else a `_values`
array is written during a nominally non-mutating operation.

### 1.3 `Scalar.sort()` discards the mask and leaks sentinel values — **high** [confirmed]

`src/polymath/scalar.py:1364-1387`

The masked entries are replaced with `_maxval(dtype)` so they sort to the end, the mask is
sorted alongside, and then:

```python
result = Scalar(new_values, new_mask, unit=self._unit)
result[new_mask] = max_possible        # <-- this clears new_mask
```

`__setitem__` assigns the *value and mask* of the right-hand side, and `max_possible` is an
unmasked float, so the final line unmasks precisely the elements that were just masked:

```python
>>> a = Scalar([3., 9., 1.], mask=[False, False, True])
>>> r = a.sort()
>>> r.values        # array([ 3.,  9., inf])   <- inf sentinel is now visible
>>> r.mask          # array([False, False, False])   <- mask is gone
```

So `sort()` on a partially masked Scalar returns `inf` (or the dtype's max integer) as a
real, unmasked value. For an integer Scalar this silently produces `2**63 - 1`.

**Fix**: drop the `result[new_mask] = max_possible` line entirely; the values already hold
the sentinel and the mask is already correct. Better still, fill the masked slots with
`self._default` instead of the sentinel so a caller who ignores the mask sees a benign
value. Add a test asserting both `.values` and `.mask` of a sorted partially-masked Scalar.

### 1.4 `__setitem__` raises `UnboundLocalError` on non-consecutive array indices — **high** [confirmed]

`src/polymath/extensions/indexer.py:179-186`

```python
if moved_to_front:
    arg_values = np.moveaxis(arg._values, after, before)
    if np.shape(arg._mask):
        arg_mask = np.moveaxis(arg._mask, after, before)     # only bound in this branch
else:
    arg_values = arg._values
    arg_mask = arg._mask
```

When the index has non-consecutive array components *and* the right-hand side has a scalar
(non-array) mask, `arg_mask` is never bound. It is then read at line 193 or 213 whenever
`self`'s own mask is an array:

```python
>>> a = Scalar(np.zeros((4,5,6,7)), mask=np.zeros((4,5,6,7), dtype=bool))
>>> a[:, np.array([0,1]), :, np.array([0,1])] = Scalar(np.ones((4,2,6)))
UnboundLocalError: cannot access local variable 'arg_mask' where it is not associated with a value
```

The bug hides when `self`'s mask is a plain `False`, which is why it survived testing.

**Fix**: initialize `arg_mask = arg._mask` before the `if`, and overwrite it only inside
the `np.shape(...)` branch.

### 1.5 Aliased `_derivs` / `_cache` dictionaries — **high** [confirmed]

Three places copy an object's `__dict__` wholesale and end up sharing mutable dicts between
two supposedly independent objects.

**(a) `Qube.wod`** (`qube.py:1723-1736`). After `wod.__init__(...)` gives the clone a fresh
`_cache`, the loop copies every non-deriv attribute from `self.__dict__`, which puts
`self._cache` itself back onto the clone:

```python
>>> s = Scalar(np.arange(6.)); s.insert_deriv('t', Scalar(np.ones(6)))
>>> s.wod._cache is s._cache
True
```

Every cache write on the derivative-free view — including `shrink()`'s
`obj._cache['unshrunk'] = self` — lands in the parent's cache and vice versa. The two
objects have different derivatives but one shared memo table.

**(b) `Polynomial.__init__`** (`polynomial.py:45-47`) copies `args[0].__dict__` by
reference, so `Polynomial(vector)` shares both dicts with the Vector it wraps:

```python
>>> p = Polynomial(v)
>>> p._derivs is v._derivs, p._cache is v._cache
(True, True)
```

**(c) `Polynomial.as_vector()`** (`polynomial.py:105-115`) does the same, then calls
`obj.insert_derivs(derivs)` — which mutates the dict that is still `self._derivs`. The
result is that asking a Polynomial for its Vector view silently downgrades the
Polynomial's own derivatives:

```python
>>> p = Polynomial([1., 2.]); p.insert_deriv('t', Polynomial([1., 0.]))
>>> type(p.derivs['t']).__name__      # 'Polynomial'
>>> _ = p.as_vector()
>>> type(p.derivs['t']).__name__      # 'Vector'  <- p was modified
```

**Fix**: in all three, copy the dicts (`dict(...)`) rather than rebinding them.
`Qube.clone()` already does this correctly (`qube.py:983`) — the same `isinstance(value,
dict)` guard belongs in `wod` and in the two `Polynomial` methods. Consider factoring the
attribute-transfer loop into one helper so the rule is stated once.

### 1.6 `np.ma.stack(*arg)` is called with the wrong signature — **high** [confirmed]

`qube.py:433`, `qube.py:480`, `qube.py:627`

`np.ma.stack(arrays, axis=0)` takes a *sequence*. Unpacking the list passes the second
masked array as `axis`:

```python
>>> Scalar([np.ma.MaskedArray([1.,2.], [0,1]), np.ma.MaskedArray([3.,4.], [1,0])])
TypeError: only integer scalar arrays can be converted to a scalar index
```

Constructing any Qube from a list of `MaskedArray`s fails. (The adjacent
`Qube.stack(*arg)` calls are correct — that one really is variadic, which is probably how
the mistake was introduced.)

**Fix**: `np.ma.stack(arg)` in all three places, and add a test constructing each of
Scalar/Vector/Boolean from a list of MaskedArrays.

### 1.7 `Boolean ** int` fails for shapeless operands — **medium** [confirmed]

`src/polymath/boolean.py:451`

```python
vals = (self._values | (arg._values == 0)).view(np.int8)
```

For a shapeless Boolean, `self.as_int()._values` is a Python `int` and `arg._values` is a
Python `int`, so the expression is a plain `int`, which has no `.view`:

```python
>>> Boolean(True) ** 2
AttributeError: 'int' object has no attribute 'view'
>>> Boolean([True, False]) ** 2      # array case is fine
Scalar(1 0)
```

**Fix**: `np.asarray(...)` before `.view`, or build the result with
`np.int8(bool(...))` on the scalar path. Parametrize the existing power tests over both
shapeless and array operands.

### 1.8 `Vector.int(top=<int>)` raises `TypeError` — **medium** [confirmed]

`src/polymath/vector.py:296-305`. The nested `_as_tuple` helper broadcasts a scalar
argument using `len(top)`, but it is called on `top` itself first, when `top` is not yet a
tuple:

```python
>>> Vector([[1., 2.]]).int(top=5)
TypeError: object of type 'int' has no len()
```

The docstring documents `top` as a tuple, so this is arguably out of contract — but the
error message gives the caller nothing to work with, and the sibling `Scalar.int()` accepts
a plain int for the same parameter.

**Fix**: use `self._numer[0]` rather than `len(top)` inside `_as_tuple`, or validate `top`
explicitly and raise a message that names the parameter.

### 1.9 `Scalar.frac(recursive=...)` ignores its argument — **medium** [confirmed]

`src/polymath/scalar.py:293-324`. The parameter is documented and accepted, but the
constructor call passes `derivs=self._derivs` unconditionally:

```python
>>> s.frac(recursive=False).derivs
['t']            # expected []
```

**Fix**: `derivs=self._derivs if recursive else {}`. Worth grepping for the same pattern —
`recursive` is threaded through dozens of methods and this is the kind of slip that repeats.

### 1.10 `Scalar.exp(check=False)` raises the wrong exception — **medium** [confirmed]

`src/polymath/scalar.py:725-730` catches `(ValueError, TypeError)`, but NumPy signals
overflow with a `RuntimeWarning`, which `warnings.filterwarnings('error')` turns into a
`RuntimeWarning` exception:

```python
>>> Scalar(1e6).exp(check=False)
RuntimeWarning: overflow encountered in exp        # docstring promises ValueError
```

The sibling methods `sqrt`, `log`, `arcsin`, and `arccos` all catch `RuntimeWarning`
correctly; only `exp` is out of step.

**Fix**: catch `RuntimeWarning` (and keep `FloatingPointError` if NumPy's error state is
ever configured to raise).

### 1.11 `Qube.as_size_zero(axis=...)` collapses the wrong axis — **medium** [confirmed]

`src/polymath/qube.py:2610-2613`

```python
if axis == 0:
    indx = slice(0, 0)
else:
    indx = (Ellipsis, slice(0, 0))
```

Any axis other than 0 collapses the *last* axis:

```python
>>> Scalar(np.zeros((3,4,5))).as_size_zero(axis=1).shape
(3, 4, 0)        # documented behavior is (3, 0, 5)
```

Only `axis=0` and `axis=-1` behave as documented.

**Fix**: build the index as `axis % self._ndims * (slice(None),) + (slice(0, 0),)`, or use
`np.moveaxis`. Parametrize a test over every axis of a 3-D object.

### 1.12 Broken error-message f-strings — **low** [confirmed]

Two messages are missing (or misplacing) the `f` prefix and print their braces literally:

- `qube.py:1600` — `'{type(self).__name__} object; object is read-only'` renders as
  `derivative "t" cannot be replaced in {type(self).__name__} object; object is read-only`.
  (Note the *first* line of that same message is a correctly-formatted f-string, so the
  message is half-interpolated.)
- `unit.py:785` — `'fnon-integer power on unit "{old_power}"'`: the `f` slipped inside the
  quotes, and `old_power` is not a defined name, so making it a real f-string would raise
  `NameError`. The intended variable is presumably the pre-conversion `power`.

`unit.py:224` is a third message defect, this one semantic: `'unit is not incompatible with
an angle'` is raised precisely when the unit *is* incompatible.

**Fix**: correct all three. A `ruff` rule cannot catch these; a test that asserts on message
content (as `.claude/rules/python_testing.md` §7 requires) would.

---

## 2. Correctness risks found by inspection

These were not reproduced end-to-end but look wrong on reading.

- **`Vector.element_div()` derivative unit** (`vector.py:786`). `arg_inv_sq` holds
  `divisor**(-2)` but is constructed with `Unit.unit_power(arg._unit, -1)`. Observed
  behavior matches: dividing km by s yields a `d/dt` derivative labelled `km/s` where the
  quotient rule gives `km/s**2`. Values are right; the unit is off by one power.

- **`unshrink()` loses the shape when the shrunken object is fully masked**
  (`shrinker.py:147-148`). `masked_single().broadcast_to(shape)` uses the `shape=`
  parameter, which defaults to `()`. With the cache enabled the `'unshrunk'` entry rescues
  the correct shape, so the result depends on `Qube._DISABLE_CACHE` — which contradicts the
  `shrink()` docstring's promise that shrinking never changes results. Reproduced with
  `_DISABLE_CACHE = True`: `shrink`/`unshrink` of an all-masked length-6 Scalar returns
  shape `()` instead of `(6,)`.

- **`Qube.__init__` treats `nrank=0` as "unspecified"** (`qube.py:225`). `nrank = nrank or
  self._NRANK or 0` cannot distinguish an explicit `0` from `None`, so
  `Vector(np.ones(3), nrank=0).numer` returns `(3,)` rather than raising. The guard on line
  235 then passes because `nrank` has already been rewritten. The `drank = drank or 0` on
  the next line has the same shape but no observable consequence. Use `if nrank is None:`.

- **`Qube` violates the hash/eq invariant.** `__eq__` is attached after class creation
  (`extensions/__init__.py:79`), so Python never sets `__hash__ = None`. Two equal Qubes
  hash differently — `Scalar(1) == Scalar(1)` is `True` while `hash(Scalar(1)) !=
  hash(Scalar(1))` — and since Qube is mutable, a Qube used as a dict key can go stale.
  Either set `Qube.__hash__ = None` explicitly or document that Qubes must not be used as
  keys. (`Unit` has the opposite issue: it defines `__eq__` in the class body, so it is
  unhashable, which is fine but undocumented.)

- **`Qube._cache` entries are not marked read-only.** `antimask` (`qube.py:1296-1303`)
  caches a freshly allocated array and hands the same object to every caller. Nothing stops
  a caller mutating it, after which every later `antimask` read is wrong. `Qube._array_to_readonly`
  already exists and would cost nothing here.

- **`as_readonly()` mutates `self._cache` while iterating it** (`qube.py:2080-2082`).
  Reassigning existing keys is safe in CPython today, but combined with the `wod` aliasing
  in §1.5(a) the loop can recurse into a call that iterates the *same* dict. It terminates
  only because `_readonly` is set before the loop. Snapshot with `list(self._cache.items())`.

- **`Matrix.unitary()` compares against `False` by identity** (`matrix.py:436`):
  `elif self._mask is not False:` is `True` for `np.False_`, so the branch can `|=` a
  scalar `np.False_` into an array mask. Harmless today; fragile.

- **`Vector.clip_component()` assigns a Scalar object into a NumPy array**
  (`vector.py:1041`): `vector._values[axis] = upper` where the parallel lower-bound branch
  correctly uses `lower._values`. It happens to work because `Scalar` defines `__float__`.

---

## 3. Dead and unreachable code

- **`math_ops.__ipow__` is never bound to `Qube`.** It is defined at `math_ops.py:1216`
  but absent from `extensions/__init__.py`, so `x **= 2` falls back to Python's default
  `x = x ** 2` rebinding rather than the in-place semantics every other augmented operator
  in the file implements. This masks a real bug in the dead function: line 1230 reads
  `self.set_unit(self, result._unit)`, passing `self` as the *unit* and the unit as
  *override*. If the binding is ever added, that line will raise
  `ValueError: not a recognized unit`. Decide whether `**=` should be in-place; if yes,
  bind it and fix line 1230; if no, delete the function.

- **`math_ops._floordiv_by_number` is never called.** `__mul__`, `__truediv__`, and
  `__mod__` all have a `Qube._is_one_value(arg)` fast path; `__floordiv__` does not, so
  `x // 2` takes the slow route through `Scalar.as_scalar` and `mask_where_eq`, and the
  helper written for the fast path is unreachable. Either add the fast path (a two-line
  change that also removes an array allocation) or delete the helper.

- **`Vector.cross_product_as_matrix()` contains `self._values._shape`** (lines 630 and
  644) — NumPy arrays have `.shape`, not `._shape`. Both lines sit in `drank > 0` branches
  that `_disallow_denom()` makes unreachable three lines earlier, so this is a latent
  `AttributeError` guarding dead code. Delete the branches or fix the typo and drop the
  guard.

- **The `else` branches in `Scalar.max/min/argmax/argmin`** that handle
  `np.shape(mask) == ()` (e.g. `scalar.py:893-894`) are unreachable: they are inside the
  partially-masked branch, where `np.all(self._mask, axis=None)` is `False` by
  construction. `min` and `argmin` additionally set `mask = True` there while `max` and
  `argmax` do not — an inconsistency with no observable effect, which is itself evidence
  the code is dead.

- **`Matrix.solve()`** is 75 lines of commented-out code (`matrix.py:441-517`) carrying the
  note "Algorithm has been validated but code has not been tested". Git history is the right
  home for this; it also contains at least two bugs (`self._derivs[k]` inside a loop over
  `key`, and a reference to an undefined `shape`).

- **`Unit.__init__:64`**: `(numer, denom) = triple[:2]` is immediately overwritten by the
  next two lines.

---

## 4. Unresolved questions left in the code

`grep` finds nine `TODO`/`XXX` markers in `src/`. Three of them record open *correctness*
questions in shipped code, which is different from a style note:

- `polynomial.py:198` — `# XXX Code Rabbit claims that this math is not correct - check it`,
  sitting directly above the derivative propagation in `invert_line()`. Either the review
  bot is wrong (in which case remove the comment and add the test that proves it) or the
  derivative of an inverted linear polynomial is wrong in the released package.
- `quaternion.py:417` — `# TODO: what to do about divide by zero here?`
- `unit.py:553` and `581` — `# XXX This is not well-specified. Why do we only do this for
  new units?` on `mul_units`/`div_units` overwriting `result.name`.

`unit.py:856` and `870` (`# TODO What is the purpose of this check?`) mark two `raise`
statements whose reachability the author could not determine; both carry `# pragma: no
cover`. If they are genuinely unreachable, delete them; a `pragma: no cover` on a defensive
raise nobody understands is coverage laundering.

`matrix3.py:849` and `quaternion.py:472` (a `NotImplementedError` with a `TODO`) are
ordinary feature gaps and fine to leave, though the latter's message has a doubled space
mid-sentence.

---

## 5. Performance

> **Status: §5.1 and §5.2's `np.prod` item were fixed on 2026-08-09.** Indexing is now
> flat in the axis length (33,000 µs to 32 µs for the million-element case) and
> construction-bound arithmetic is about 2x faster. The measurements in the table below
> are the *pre-fix* baseline, kept as the record of what was wrong; see the "after"
> column. The `numbers` ABC item in §5.2, and all of §5.3 through §5.6, are still open.

Measured on this machine (Python 3.12, NumPy 2.5.2, arrays of 1000 elements unless noted):

| Operation | polymath | raw NumPy | ratio |
|---|---|---|---|
| Operation | before | after §5.1/§5.2 | raw NumPy |
|---|---|---|---|
| `Scalar + Scalar` | 22.6 µs | 10.4 µs | 0.65 µs |
| `Scalar() + float` (shapeless) | 10.1 µs | 9.9 µs | — |
| `Scalar(ndarray)` construction | 18.9 µs | 7.0 µs | — |
| `Vector3.cross` | 57.3 µs | 34.1 µs | 22.2 µs |
| `Vector3.norm` | 57.9 µs | 33.7 µs | — |
| `v + w` (Vector3) | 22.2 µs | 11.0 µs | — |
| Index 100 elements of a 10³ Scalar | 81 µs | 33 µs | 0.3 µs |
| Index 100 elements of a 10⁵ Scalar | 3,093 µs | 32 µs | 0.3 µs |
| Index 100 elements of a 10⁶ Scalar | 34,418 µs | 32 µs | 0.3 µs |
| Index 100 of a 10⁶ Scalar, masked | 34,965 µs | 658 µs | — |

The last three rows were the headline: indexing cost scaled with the size of the axis
rather than the number of indices. It is now flat for an unmasked index, and linear in C
rather than in Python objects when the index is masked.

### 5.1 `_prep_index` is O(axis length) on every array index — **highest-value fix**

`src/polymath/extensions/indexer.py:429-444`

```python
index_vals = index_vals % axis_length
if np.shape(mask_vals):
    antimask = np.logical_not(mask_vals)
    unused_set = (set(range(axis_length)) - set(index_vals[antimask]))
elif mask_vals:
    unused_set = ()
else:
    unused_set = (set(range(axis_length)) - set(index_vals.ravel()))

if unused_set:
    unused_index_value = unused_set.pop()
else:
    unused_index_value = -1

if any_masked:
    index_vals = index_vals.copy()
    index_vals[mask_vals] = unused_index_value
```

`unused_index_value` is used **only** inside `if any_masked:`, but the Python-set
construction that produces it runs unconditionally. For an unmasked index — by far the
common case — the function builds and discards a Python set with one entry per element of
the indexed axis. Hence 33 ms to index a million-element Scalar with 100 integers.

Two independent fixes, both small:

1. **Guard the whole block with `if any_masked:`.** This alone removes the cost from every
   unmasked index, which should bring the 10⁶ case to roughly the 10³ case's cost.
2. **Replace the set arithmetic with a NumPy occupancy test** for the case where it *is*
   needed:

   ```python
   used = np.zeros(axis_length, dtype=np.bool_)
   used[index_vals[antimask]] = True
   candidates = np.flatnonzero(~used)
   unused_index_value = int(candidates[0]) if candidates.size else -1
   ```

   That is O(axis_length) in C rather than in Python objects — roughly two orders of
   magnitude cheaper — and allocates one byte per element instead of a `PyObject*` plus a
   boxed int.

Also in this function: `index_vals % axis_length` allocates a new array on every index even
when all indices are already in range. Guard it with a cheap
`if np.any(index_vals < 0):` — negative indices are the minority case.

### 5.2 `Qube.__init__` dominates arithmetic, and a third of it is `np.prod`

`cProfile` over 20,000 `Scalar + Scalar` operations:

```text
   ncalls  tottime  cumtime  function
    20000    0.169    0.786  qube.py:133(__init__)          <- 85% of total runtime
    80000    0.101    0.249  numpy _wrapreduction
    80000    0.049    0.298  numpy prod                     <- 32% of total runtime
   600000    0.084    0.133  isinstance
   160000    0.022    0.049  abc.__instancecheck__
```

Two cheap, local wins:

- **Replace `np.prod` with `math.prod`** at `qube.py:276-279`. Four calls per construction,
  always on a small tuple of Python ints. Measured: `int(np.prod(()))` is 2.9 µs;
  `math.prod(())` is 0.08 µs — about **30x faster**. This one edit should remove roughly a
  third of construction cost. (`np.prod` also returns `1.0` for an empty tuple, which is why
  the `int(...)` wrappers are there; `math.prod` returns `1` and the wrappers can go.)
  `shaper.flatten:63` and `item_ops.reshape_numer:234` / `reshape_denom:328` use `np.prod`
  the same way.

- **Stop routing hot-path type checks through `numbers` ABCs.** `isinstance(x,
  numbers.Real)` is 0.41 µs versus 0.09 µs for `isinstance(x, (int, float))` — 4.4x — and
  the profile shows 160,000 ABC `__instancecheck__` calls per 20,000 additions.
  `Qube._is_one_value` (`qube.py:2253`) and the `numbers.Real` / `numbers.Integral` tests in
  `_as_values_and_mask`, `_dtype_and_value`, `_as_mask`, and the operator fast paths are the
  hot ones. Check `(int, float, np.number)` first and fall back to the ABC only if that
  misses, so third-party numeric types still work.

### 5.3 A fast internal constructor would pay for itself

Beyond the two micro-fixes, the structural issue is that every internal result goes through
the full public constructor. `Qube.__init__` re-derives everything the caller already knows:
`_as_values_and_mask` re-inspects the values, `_suitable_value` calls `_dtype_and_value`
*again* plus `_suitable_dtype` and `_suitable_numer`, `_suitable_mask` re-validates the
mask, and `_casted_to_dtype(default, dtype)` re-runs dtype inference for the default.

Internal call sites — `__add__`, `__sub__`, `_mul_by_scalar`, `_div_by_scalar`, `dot`,
`cross`, `outer`, `norm` — all construct from a NumPy array whose dtype and shape they
computed themselves. A `Qube._new_from_parts(cls, values, mask, *, nrank, drank, unit,
example)` that sets the ~20 attributes directly and skips validation would remove most of
the remaining 85%. `clone()` already demonstrates the pattern; this is the same idea applied
to results rather than copies. Keep the validating `__init__` as the public entry point.

### 5.4 Redundant passes over masks and values

Several methods walk the mask two or three times where once would do:

- `vector_ops._mean_or_sum:50,54` calls `np.any(arg._mask)` then `np.all(arg._mask)`, then
  `arg._values.copy()` followed by `new_values[arg._mask] = 0`. `np.where(mask, 0, values)`
  is one pass and one allocation instead of two.
- `Scalar.max/min/argmax/argmin` each do `np.any(self._mask)`, `np.all(self._mask)`,
  `self._values.copy()`, and a masked assignment — four passes. Computing
  `count = np.count_nonzero(mask)` once answers both the `any` and `all` questions.
- `Qube._find_corners` (`qube.py:1441-1452`) runs one `np.any` reduction per axis, i.e. it
  reads the whole antimask `ndims` times. For 2-D and 3-D backplanes that is 2-3 full
  passes where `np.argwhere` on the flattened antimask plus `np.unravel_index` needs one.
- `Qube.__str__` calls `np.any(self._mask)` and then, for masked objects, constructs a
  whole temporary Qube (`qube.py:2979`) just to format it. Fine for `repr`, but `__str__`
  is also used inside error paths.

### 5.5 Array-op level opportunities

- **`Qube.dot`** (`vector_ops.py:250`) computes `np.sum(array1 * array2, axis=-1)`, which
  materializes the full elementwise product before reducing. `np.einsum('...i,...i->...',
  a, b)` (or `np.matmul` for the matrix cases) avoids the temporary entirely — a
  meaningful memory win on large backplanes, not just a speed one. The
  `np.ascontiguousarray` calls on lines 246-247 also force two full copies; with `einsum`
  they become unnecessary.
- **`Qube.as_diagonal`** (`vector_ops.py:692-693`) fills the diagonal with a Python loop
  over the item length. `np.einsum('...ii->...i', new_values)[...] = rolled` writes the
  diagonal in one call.
- **`_cross_3x3`** (`vector_ops.py:545`) allocates `np.empty(a.shape)` with the default
  float64 dtype regardless of input dtype, so integer inputs silently produce float output
  and pay for the conversion. Use `np.result_type(a, b)`. The measured 2.6x gap versus
  `np.cross` for Vector3 is mostly constructor overhead (§5.2), but the extra
  `np.broadcast_arrays` copy contributes.
- **`Matrix.identity`** (`matrix.py:594-596`) builds the identity with a Python loop;
  `np.eye(size)` is one call. Same for `Qube.or_`/`and_` with three or more masks
  (`qube.py:913`), which recurse pairwise and allocate an intermediate array per step;
  `np.logical_or.reduce` on the array subset would do it in one.
- **`broadcaster.broadcasted_shape`** reimplements NumPy's broadcasting rules in pure
  Python and is called from `_prep_index` and `broadcast` on every operation.
  `np.broadcast_shapes(*shapes)` is a C implementation of exactly this.
- **`Unit.create_name`** runs a triple-nested loop over unit options (up to ~200 iterations
  with a `math.gcd` each) and is called from `get_name()` → `__str__` with no memoization.
  It is only 20 µs per call, but `__str__` runs in error paths and in every `repr` in a
  notebook. `functools.lru_cache` keyed on `(exponents, triple)` would make it free after
  the first call — the same treatment `Scalar._minval`/`_maxval` already get.
- **`Scalar.maximum`/`minimum`** (`scalar.py:1177-1181`) reduce via
  `result[antimask] = scalar[antimask]`, i.e. a full `__getitem__` + `__setitem__` round
  trip per argument. `np.where` on the values plus `Qube.and_`/`or_` on the masks stays in
  NumPy.

### 5.6 A note on legacy NumPy APIs

`np.rollaxis` appears 34 times across nine modules. NumPy has recommended `moveaxis` since
1.11; `rollaxis` is not deprecated but its argument convention (`start` semantics) is the
documented source of confusion, and the code already mixes both (`indexer.py` uses
`moveaxis`, `vector_ops.py` uses `rollaxis`). Converting is mechanical but touches enough
call sites that it deserves its own change with its own test run — worth doing while the
axis-manipulation code is fresh, not as a drive-by.

---

## 6. Consistency and API shape

- **`Scalar.as_index_and_mask` is keyword-only; `Vector.as_index_and_mask` is not**
  (`scalar.py:142` vs `vector.py:206`), and `Vector.as_index` calls it positionally. Same
  method name, same conceptual signature, two different calling conventions. Likewise
  `Scalar.int` has `*` before `remask` while `Vector.int` does not, and `Qube.as_int` /
  `as_bool` take `copy` and `builtins` positionally while `as_float` makes them keyword-only.
  `.claude/rules/python.md` §2 caps positional parameters at 5; these are all under the cap,
  so the rule does not force a change — but the asymmetry between sibling classes will keep
  producing small surprises.

- **`Qube.pickle` is bound to the *module*** (`extensions/__init__.py:143`), so
  `type(Qube.pickle)` is `module` and `obj.pickle` returns the pickler module rather than
  anything callable. The comment explains the motive (`help(Qube.pickle)` shows the module
  docstring), but it puts a non-callable, non-data attribute into the public namespace of
  every Qube. A module-level `polymath.pickling` documented in Sphinx would achieve the same
  without the surprise.

- **`extract_denom`'s docstring example is wrong** (`item_ops.py:56-58`): it claims that
  extracting from a Vector with shape `(3,)`, numer `(3,)`, denom `(3,)` returns shape `()`.
  Verified: the result has shape `(3,)`. Extracting a denominator axis does not touch the
  leading shape.

- **Other docstring/signature mismatches**: `Qube.filled` documents a `dtype` parameter it
  does not have (the parameter is `fill`, which is undocumented); `Qube._suitable_mask`
  documents `expand` where the parameter is `broadcast`; `Qube.__init__` promises
  `ValueError` for a disallowed unit but raises `TypeError` (`qube.py:233`).

- **`Qube.__getitem__` deliberately diverges from NumPy** in two ways that are not
  documented in the class docstring: non-consecutive array indices keep their axis position
  instead of moving to the front (`a[:, [0,1], :, [0,1]]` gives shape `(4,2,6)` where NumPy
  gives `(2,4,6)`), and a scalar boolean index does not add a leading axis
  (`Scalar(np.arange(3))[True].shape` is `(3,)` where NumPy gives `(1,3)`). Both look
  intentional and both are reasonable; they just need to be stated where a user indexing a
  Qube will find them.

- **Broad exception handling**: `indexer._prep_index` wraps its entire body in
  `except Exception as err: raise IndexError(err) from err`, which converts genuine
  programming errors (the `UnboundLocalError` of §1.4 would be caught here if it were inside
  the block) into `IndexError`. `.claude/rules/python.md` §2 asks for the smallest possible
  granularity. Narrow it to the conversions that can legitimately fail.

- **`pickler.py:88` uses a bare `assert` for a runtime invariant**
  (`assert sys.float_info.mant_dig == 53`). Assertions are removed under `python -O`, so the
  check silently disappears in exactly the deployment mode where a surprise float format
  would be most costly. Raise instead.

- **`pickler` keeps mutable module-level globals** (`_DEFAULT_PICKLE_DIGITS`,
  `_DEFAULT_PICKLE_REFERENCE`, `_PICKLE_DEBUG`) written by
  `set_default_pickle_digits()` with no locking, as do `Qube._PREFER_BUILTIN_TYPES`,
  `Qube._DISABLE_CACHE`, and `Qube._DISABLE_SHRINKING`. That is a defensible design for a
  numeric library, but the thread-safety guarantee (or absence of one) is not stated
  anywhere in the docs. One sentence in the `Qube` class docstring would settle it.

---

## 7. Configuration and tooling

- **No minimum version constraints on runtime dependencies.** `pyproject.toml` declares
  `dependencies = ["numpy", "rms-fpzip"]`. `.claude/rules/dependency_management.md` §3
  requires minimum versions ("e.g., `numpy>=2.2.0`"), and the code targets NumPy 2 behavior.
  A user on NumPy 1.20 gets an install that fails at runtime rather than at resolve time.

- **The `dev` extra depends on the package itself** — `dev = ["rms-polymath", ...,
  "rms-polymath[docs]"]`. It resolves harmlessly under `pip install -e ".[dev]"` but is
  circular; the `[docs]` self-reference is the only one that does real work, and
  `--group docs` or listing the docs deps inline would be clearer.

- **Two different coverage source specs**: `addopts` says `--cov=src` while
  `[tool.coverage.run] source = ["polymath"]`. They happen to agree because of the src
  layout, but a future `--cov` change would silently diverge from `fail_under`.

- **`filterwarnings` is not configured.** `.claude/rules/python_testing.md` §4 asks for
  `filterwarnings = ["error", ...]`. Given how much of this library's correctness rests on
  NumPy warning behavior — §1.10 is exactly a warnings bug, and `Scalar.__pow__`,
  `arcsin`, `sqrt`, `log`, `reciprocal`, and `Matrix.inverse` all manipulate the warnings
  filter — turning warnings into errors in the test suite would be unusually valuable here.

- **`scripts/run-all-checks.sh` runs `mypy src tests`** (line 381) while `CLAUDE.md` says
  "**Never run `mypy` on `src/`**". The check is disabled by default so nothing breaks
  today, but the script and the rule contradict each other; change the invocation to
  `mypy tests` so that enabling the flag does the documented thing.

- **CI and the check script have drifted apart** in one direction:
  `.claude/rules/environment.md` §2 requires CI to run exactly the script's enabled set. CI
  runs ruff, pyroma, sphinx, pymarkdown, pytest — which matches — but CI's `pymarkdown scan`
  covers `docs/ .claude/ README.md CONTRIBUTING.md` while the script's comment says it
  covers the same set; worth a one-line check that they stay in sync, since CLAUDE.md
  claims the local run is *stricter*.

- **No `pip audit` step**, which `.claude/rules/security.md` §2 and
  `dependency_management.md` §5 both call for. With only two runtime dependencies this is
  low-risk but cheap to add.

- **The PEP 561 story is incomplete.** `py.typed` is shipped and `src/polymath/__init__.pyi`
  exists, but it is 31 lines of re-exports and the stub's own docstring admits the classes
  "are still inferred from their implementation modules". Since `src/` carries no
  annotations by policy, downstream type checkers see a typed package whose entire public
  surface is untyped — arguably worse than shipping no marker, because it suppresses the
  "missing stubs" diagnostic that would otherwise tell users to expect nothing. Either
  commit to per-module stubs (starting with `qube.pyi` and `scalar.pyi`, which cover most
  of the surface) or drop `py.typed` until they exist.

---

## Recommended priorities

1. **Fix the six confirmed defects that produce wrong answers or exceptions**: §1.1
   (`as_fully_masked`), §1.2 (`Matrix.inverse` mutating input), §1.3 (`sort()` losing the
   mask), §1.4 (`__setitem__` `UnboundLocalError`), §1.6 (`np.ma.stack`), §1.5 (the three
   aliased-dict cases). Each is a few lines. Each needs a regression test — all six live in
   paths the current suite does not reach, which is the more important signal.

2. ~~**Make the two performance changes**: guard the `unused_set` computation in
   `_prep_index` (§5.1) and swap `np.prod` for `math.prod` in `Qube.__init__` (§5.2).~~
   Done 2026-08-09. Indexing is flat in the axis length and construction-bound arithmetic
   is roughly 2x faster; 420 randomized index cases confirm the semantics are byte-identical
   to the previous implementation.

3. **Clean up the remaining confirmed defects and the dead code**: §1.7-§1.12, then the
   §3 items — decide `__ipow__`'s fate, add the `__floordiv__` fast path or delete its
   orphaned helper, and remove `Matrix.solve`'s commented-out body.

4. **Resolve the correctness `XXX`s**, starting with `Polynomial.invert_line` (§4). An
   unanswered "this math may be wrong" note in a released numerical library is a liability
   regardless of whether it turns out to be right.

5. **Then consider the structural performance work** — the fast internal constructor
   (§5.3) and the `einsum` conversion in `dot` (§5.5). These are larger and want a
   benchmark harness committed alongside them, so that the next person can tell whether a
   change helped.
