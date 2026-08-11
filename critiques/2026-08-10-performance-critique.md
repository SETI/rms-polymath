# Performance critique: rms-polymath — 2026-08-10

Scope: the speed of `src/polymath/` on branch `mark-reorg` at commit `08fc6df`. This is a
follow-on to the performance section of `2026-08-09-code-critique.md`, whose two findings
(`_prep_index` and the `np.prod` calls in `Qube.__init__`) are both addressed. It looks
only at execution speed; correctness, documentation and the test suite are out of scope.

Every measurement below was taken against the project virtualenv on this machine, Python
3.12.0 with NumPy 2.5.2, on objects of 1000 elements unless stated otherwise, as the
minimum of three `timeit` repeats. Profiles were taken with `cProfile`; absolute times
under the profiler are inflated and are quoted only as proportions.

## Summary

> **Status: sections 1 through 5.2 were implemented on 2026-08-10**, each with its own
> commit, differential testing against the previous behavior, and measurements recorded in
> the commit message. Section 5.3 (`Quaternion.from_matrix3()`) and section 5.4 (the
> `Matrix3` pickling trade) were reviewed and deliberately left alone. Each section below
> carries a note recording what was done. The measurements in the text are the ones taken
> on 2026-08-09 and 2026-08-10 before the work, and are kept unedited as the record of
> that reading.

The branch is roughly **2.9x faster than `main`** across a spread of 41 operations
(geometric mean; median 2.7x), so the large structural wins are already banked. What
remains is concentrated in three themes, in descending order of value.

1. **One pathological path.** `mask_where()` with a replacement value costs about 80 us no
   matter how few items it replaces, because it copies the whole object and then routes
   through `__setitem__`. It sits under `Scalar.sqrt()`, `Scalar.log()`, `clip()`, every
   `Vector.mask_where_*` variant, and the divide-by-zero guard. The effect is a cliff:
   `Scalar.sqrt()` costs 11 us on clean data and **92 us as soon as a single value out of
   1000 is negative**. A direct implementation measures 3.1 us.
2. **Result construction still goes through the validating constructor** almost
   everywhere outside the four binary arithmetic operators. `Qube.__init__` costs 5.80 us
   against 1.32 us for `Qube._new_from_parts()`, and unary math, reductions, indexing,
   boolean operations and `cast()` all still pay the higher price. `cast()` is the worst
   of these because it is called immediately *after* the fast constructor in the
   `vector_ops` results, handing back most of what was just saved.
3. **A few NumPy-level choices.** `Matrix3 * Matrix3` spends 96% of its time in the
   generic contraction inside `dot()`, which `np.matmul` computes 2.4x faster; and
   `clone()` and `wod` iterate `self.__dict__`, which both costs a string scan per
   attribute and permanently slows attribute access on both objects involved.

Top three: §1, then §2.1 (`cast`), then §2.2 (the unary and reduction constructors).
Together they should be worth a factor of two or more on the operations that are still
slow, and §1 alone removes an 8x cliff from ordinary numerical code.

---

## 1. `mask_where(..., replace=...)` costs the same whether it replaces two items or a thousand — **highest value** [confirmed]

> **Done** in `6a98197`. A replacement that is a single unmasked item without
> derivatives is written into the value arrays directly, and a number replacing the
> items of a rank-zero object is cast rather than wrapped in an object. Verified
> against the previous path over 384 combinations. `mask_where()` with a replacement
> went from 80.4 to 7.2 us, and `Scalar.sqrt()` of an array holding two negative
> values from 91.7 to 16.9 us.

`src/polymath/extensions/mask_ops.py:79-85`

```python
    rep_mask = mask if replace._shape else True

    obj = self.copy(recursive=recursive)
    obj[mask] = replace[rep_mask]   # handles derivatives too!

    if remask:
        obj = obj.remask_or(mask, recursive=recursive)
```

The comment is accurate about derivatives, and that is the reason for the design, but the
price is steep. `self.copy()` duplicates the whole values array, the mask and every
derivative; `obj[mask] = ...` then enters the full `__setitem__` machinery in
`indexer.py`, which re-derives index information, clones again, and scans attribute names.

Replacing **2 items out of 1000** in a `Scalar`:

```text
mask_where(mask, replace=1.)      80.39 us
copy + assign + fast construct     3.09 us      (26x)
np.where(mask, 1., values)         1.23 us
```

The cost is independent of how many items match, so ordinary numerical code hits a cliff
whenever a single bad value appears:

```text
Scalar.sqrt(), no negative values           10.98 us
Scalar.sqrt(), 2 negative values of 1000    91.66 us      (8.3x)
np.sqrt on the same array                    0.74 us
```

The same cliff shows up in division: `a / b` costs 8.6 us when `b` has no zeros and
89.5 us when it has ten, and in `Scalar.log()`, `clip()`, and the four
`Vector.mask_where_le/ge/lt/gt` methods that pass a replacement through
(`src/polymath/vector.py:919, 943, 966, 990`).

**What a fix has to preserve.** The current behavior on derivatives is not incidental and
a fast path must reproduce it exactly:

```python
>>> a = Scalar([1., -2., 3.]);  a.insert_deriv('t', Scalar([10., 20., 30.]))
>>> b = a.mask_where_lt(0., replace=99.)
>>> b.values, b.mask, b.d_dt.values
(array([ 1., 99.,  3.]), array([False, True, False]), array([10.,  0., 30.]))
```

The replaced position takes the replacement value, the mask is set, and the derivative is
zeroed there (not merely masked).

**Fix**: add a fast path for the common shape of the problem — a replacement that is a
single unmasked constant with no derivatives of its own. Copy the values array, assign
into it under the mask, zero the same positions in each derivative, and build the result
with `_new_from_parts()`. Fall back to the existing `__setitem__` path when the
replacement is an array, carries a mask, or carries derivatives. The measured floor for
that fast path is 3.1 us, so the expected result is `Scalar.sqrt()` becoming
insensitive to whether its input contains negatives.

## 2. Result construction still runs the validating constructor

`Qube.__init__` performs type checking, dtype coercion, shape inference and mask
validation. `Qube._new_from_parts()` (`src/polymath/qube.py:1161`) skips all of it for
callers that have already computed the answer:

```text
Scalar(ndarray)                    5.80 us
Scalar._new_from_parts(ndarray)    1.32 us
np.sqrt on the same array          1.35 us
```

The fast constructor is used by ten call sites — the four binary operators and six
`vector_ops` results. Everything else still pays 5.80 us. Counting constructor calls per
operation:

```text
operation                     __init__  fast ctor
Scalar +                             0          1
Scalar /                             0          1
Scalar.sqrt() clean                  1          0
Scalar.sqrt() w/ negatives           4          0
Scalar.sin()                         1          0
Scalar.sum()                         2          0
Scalar.mean()                        2          0
Scalar.max()                         1          0
Scalar.clip()                        1          0
Vector3.norm()                       1          1
Vector3.unit()                       1          2
Vector3.cross()                      1          1
Vector3.dot()                        1          1
Matrix3 * Matrix3                    1          1
Matrix3 * Vector3                    1          1
Matrix3.to_euler()                   3          0
Quaternion.from_matrix3()            1          0
a[mask]                              2          0
a[10:900]                            1          0
Boolean &                            1          0
a.broadcast_to((5, N))               1          0
```

### 2.1 `cast()` re-runs the full constructor, immediately after the fast one [confirmed]

> **Done** in `690dd44`. `_castable_to()` decides whether the new class restricts the
> data type, unit, denominator or derivatives; when it does not, the fast constructor
> builds the result. Verified over 165 source and target combinations, including the
> ones that coerce and the ones that raise. A cast that re-types went from 6.9 to
> 2.3 us.

`src/polymath/qube.py:2796-2800`

```python
            # Construct the new object
            obj = Qube.__new__(cls)
            obj.__init__(self._values, self._mask, derivs=self._derivs,
                         example=self)
            return obj
```

This is why every row above that shows a fast constructor *also* shows an `__init__`. The
`vector_ops` results are built with `_new_from_parts()` and then immediately re-typed:

```python
    obj = Qube._new_from_parts(new_values, ...)
    obj = obj.cast(classes)
```

Measured:

```text
cast() when the class already matches    0.17 us      (early return)
cast() when it actually re-types         6.88 us
```

So `Vector3.dot()` costs 26.43 us of which the raw `np.einsum` contraction is 7.80 us, and
6.88 us of the remainder is a constructor re-run over values that were already validated
one line earlier.

**Fix**: `cast()` has, by construction, an object whose values, mask, unit and derivatives
are already valid — only `__class__` and the class-derived attributes need to change. It
can copy the attributes across directly (or, more cheaply still, `_new_from_parts()` into
the target class and move the derivatives over) instead of re-entering `__init__`. The
early-return path shows the floor is 0.17 us.

An alternative worth considering: let `_new_from_parts()` take the target class list and
resolve it internally, so the `vector_ops` sites never build an object of the wrong class
in the first place.

### 2.2 Unary math, reductions, indexing and boolean operations [confirmed]

> **Done** in `3ae8699`, along with section 5.2. The unary `Scalar` functions, the
> reductions, max and min, the logical operators, indexing and `broadcast_to()` all
> build their results with the fast constructor now. The reductions keep their
> `cast()`, because that is what makes `Boolean.sum()` a `Scalar`.

None of these use the fast constructor, and each is dominated by the one they do use:

```text
Scalar.sin()      one __init__ per call;  the ufunc itself is under 5% of the time
Scalar.sum()      two __init__ per call;  np.sum on the same array is ~1 us of 23 us
Scalar.mean()     two __init__ per call
Boolean &         one __init__ per call for what is a single np.logical_and
a[mask]           two __init__ per call
```

`Scalar.sum()` at 23 us for an operation whose NumPy core is about 1 us is the clearest
case. These call sites have all computed their values and know their rank, so they meet
`_new_from_parts()`'s contract as stated in its docstring.

**Fix**: convert them, in batches, checking each against the contract. The unary
`Scalar` methods are the easiest (rank is unchanged, the mask is passed through), the
reductions next, then the indexing results.

### 2.3 `Qube.__init__` still computes the four shape products [confirmed]

> **Done** in `7eb4a89`, and marginal as predicted: between nothing and 5%. Validated
> across 262005 constructions in the test suite.

Commit `dea0fa0` taught `_new_from_parts()` to take `_size`, `_isize`, `_nsize` and
`_dsize` from its `example` when the shapes carried through. `__init__` was left alone and
still calls `math.prod` four times per construction (`src/polymath/qube.py:305-308`); a
profile of `Scalar(ndarray)` shows 80000 `math.prod` calls for 20000 constructions.

`__init__` also accepts an `example`, and the same reasoning applies to it. This is a
small win on its own but it is nearly free, and it shrinks the cost of every construction
that §2.1 and §2.2 do not eliminate. The profile also shows 14 `isinstance` calls per
construction, which is worth a look while in the area.

## 3. The generic contraction in `dot()` is the cost of every matrix product [confirmed]

> **Done** in `ac8dc4a`. `matmul` for a matrix times a matrix, a direct subscript for
> a matrix times a vector, and the general path for everything else. One wrinkle the
> measurements below missed: `matmul` is much slower on strided input than on a
> contiguous copy, and a transposed matrix is always strided, so the operands are
> made contiguous first. `Matrix3 * Matrix3` went from 156 to 70 us, or to 85 us when
> one operand is a transpose.

`src/polymath/extensions/vector_ops.py:244-261`

All matrix products route through `dot()`, which is fully general: it reshapes both
operands so their numerator axes broadcast against each other, moves the contracted axis
last, and reduces with a single subscript.

```python
    array1 = arg1._values.reshape(shape1)
    array2 = arg2._values.reshape(shape2)
    array1 = np.moveaxis(array1, k1, -1)
    array2 = np.moveaxis(array2, k2, -1)
    new_values = np.einsum('...i,...i->...', array1, array2)
```

That generality is what makes one function serve every rank and denominator combination,
but for the two shapes that dominate real use it costs a great deal. Reproducing exactly
what `dot()` computes, against the specialized alternatives:

```text
Matrix3 x Matrix3
  whole operation                            163.57 us
  the contraction as written                 156.47 us   (96% of the operation)
  np.matmul(x, y)                             64.04 us   (2.4x faster)
  np.einsum('...ij,...jk->...ik', x, y)      146.89 us

Matrix3 x Vector3
  whole operation                             35.33 us
  the contraction as written                  21.01 us
  np.einsum('...ij,...j->...i', x, w)         13.83 us   (1.5x faster)
  np.matmul(x, w[..., np.newaxis])[..., 0]    30.73 us   (slower)
```

Both alternatives agree with the current result to 4.4e-16.

The reason the broadcast form is expensive is that it materializes the contraction over
the full outer product: for matrix times matrix it reduces over `N x 3 x 3 x 3` elements,
where `matmul` hands the same work to the BLAS-backed gufunc.

Note that `matmul` is the right answer only for matrix times matrix; for matrix times
vector it is *slower* than either einsum form, because of the reshape it needs. **This is
not a blanket substitution.** The comment at `vector_ops.py:258` explaining why `einsum`
was chosen over an explicit elementwise product remains correct for the general path.

**Fix**: keep `dot()` as the general implementation and dispatch to a specialized
contraction when the operands match a common shape and neither has a denominator: two
numerator axes on each side to `np.matmul`, and two against one to
`np.einsum('...ij,...j->...i', ...)`. That is worth about 90 us on `Matrix3 * Matrix3` and
about 7 us on `Matrix3 * Vector3`.

## 4. `clone()` and `wod` iterate `self.__dict__`

> **Done** in `915939c`. The attributes are listed on the class and copied by name,
> with a test asserting the list covers everything a constructed object carries.
> Objects made by either method now read their attributes as fast as fresh ones, and
> so do the objects they were made from. `clone()` went from 5.8 to 2.3 us.

`src/polymath/qube.py:1075-1083` and `1958-1967`

```python
        for attr, value in self.__dict__.items():
            if attr in ('_derivs', '_cache'):
                obj.__dict__[attr] = {}
            elif attr.startswith('d_d'):
                continue
```

Two costs, one obvious and one not.

The obvious one: 23 instance attributes means 23 `startswith` calls and a dict rebuild per
clone. `clone()` measures 5.77 us, 12.48 us with one derivative.

The subtle one: since Python 3.11 an object's attributes live in an inline values array,
and `LOAD_ATTR` specializes to a direct index. Touching `__dict__` explicitly materializes
a real dict and permanently loses that. Attribute access on the affected objects:

```text
freshly constructed Scalar     12.81 ns per access
a + b                          12.83 ns
a[0:10]                        12.93 ns
23-slot class, for reference   13.03 ns
a.clone()                      28.94 ns      (materialized)
a.wod                          28.96 ns      (materialized)
object carrying a derivative   28.68 ns      (first one to add a d_d* name)
```

Reading `self.__dict__` materializes the *source* as well, so a clone slows down both
objects for the rest of their lives. With 115 `clone()`/`wod` call sites in `src/`, a
meaningful fraction of live objects are on the slow path.

**Fix**: replace the `__dict__` scan with an explicit tuple of attribute names, copying
with `getattr`/`setattr` and deriving the `d_d*` names from `_derivs` rather than
discovering them by prefix. The measured end-to-end payoff from staying on the inline path
is modest — 12% on the cheapest operations, 0-5% on anything doing real array work — so
this ranks below §1 and §2, but it also removes the per-attribute string scan.

This item also explains why `__slots__` is not worth pursuing: the inline-values path
already matches a slotted class (12.81 ns against 13.03 ns), so slots would buy only 88
bytes per instance, while breaking the pickle format, which is literally the instance dict
(`pickler.py:893` returns `clone.__dict__`, `pickler.py:922` assigns it back), and
breaking the dynamically-named `d_d*` derivative attributes.

## 5. Smaller items

### 5.1 `norm()` squares into a temporary

> **Done** in `c74473d`. `Vector3.norm()` went from 21.4 to 15.4 us.

`src/polymath/extensions/vector_ops.py:341`

```python
    new_values = np.sqrt(np.sum(arg._values**2, axis=k1))
```

`arg._values**2` allocates a full `(N, 3)` temporary that `np.sum` then reduces away.
Contracting instead avoids it:

```text
np.sqrt(np.sum(x**2, axis=-1))               15.42 us
np.sqrt(np.einsum('...i,...i->...', x, x))    6.01 us      (2.6x)
```

The two agree to 4.4e-16. This is the same change already made for the dot product at
`vector_ops.py:261`, and it accounts for over half of `Vector3.norm()`'s 26.80 us.

`norm()` takes an arbitrary `axis`, so the substitution applies directly only when the
contracted axis is last; the general case needs a `moveaxis` first, exactly as `dot()`
already does.

### 5.2 `Matrix3.to_euler()` builds three objects

> **Done** in `3ae8699`. The three `Scalar`s are built with the fast constructor;
> `to_euler()` went from 76 to 63 us.

75.27 us, with three `__init__` calls and the body of `matrix3.py:721` accounting for half
the profile. It returns three `Scalar`s, so three constructions is the floor unless the
method is restructured, but they can be the cheap kind (§2.2).

### 5.3 `Quaternion.from_matrix3()` is the slowest single conversion

> **Not done**, as recommended. It is correct and well tested, and the vectorization
> it would take is not worth the risk against the value.

267.55 us, of which the function body is 73% — four Python-level branches each doing
boolean-mask fancy indexing, plus an `argmax` and two reductions. It is also the only
operation in the comparison that is slower than `main` (0.93x), which bought the identity
matrix converting correctly. It could likely be vectorized further, but it is correct and
well tested now; treat it as a low priority and change it only with the finite-difference
tests in `tests/test_quaternion_matrix3.py` as the guard.

### 5.4 `Matrix3` pickling trades CPU for size

> **Not done**, and nothing to do: this records an intended trade rather than a
> defect.

Writing a 1000-element `Matrix3` costs 1118 us against `main`'s 460 us, and produces 24233
bytes against 68276. That is the intended trade from the quaternion encoding, and the
`_QUATERNION_PICKLE_CUTOFF` constant in `matrix3.py` is the knob if the balance is wrong
for a given workload. Noted here so it is not mistaken for a regression.

## Recommended priorities

1. **§1** — the `mask_where` replacement fast path. One function, a measured 26x on the
   path itself, and it removes an 8x cliff from `sqrt`, `log`, `clip` and division on
   ordinary data. Highest value by a wide margin.
2. **§2.1** — stop `cast()` re-entering `__init__`. One function, ~6.7 us off every
   `norm`, `dot`, `cross` and matrix product.
3. **§2.2** — move unary math, reductions and indexing onto `_new_from_parts()`. More
   call sites, but each is mechanical and independently testable.
4. **§3** — specialize the contraction in `dot()` for the two common matrix shapes.
   Worth about 90 us on `Matrix3 * Matrix3`.
5. **§2.3** — the shape products in `__init__`, mirroring what `_new_from_parts()` already
   does.
6. **§4** — the `__dict__` scans in `clone()` and `wod`.
7. **§5** — the remaining items, as opportunity allows.

A note on method: each of these was found by profiling, and each proposed fix has a
measured floor quoted beside it. Any of them should be re-measured after implementation
rather than assumed, and the differential-testing approach used for
`mask_where_eq` in commit `08fc6df` — comparing the new implementation against the old
across a grid of dtypes, ranks, masks and arguments — is the right guard for §1 and §2.1
in particular, where the derivative and mask semantics are easy to get subtly wrong.
