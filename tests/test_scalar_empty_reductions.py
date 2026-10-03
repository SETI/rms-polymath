##########################################################################################
# tests/test_scalar_empty_reductions.py
# Unit tests for reductions over objects with no elements
##########################################################################################

import numpy as np
import pytest

from polymath import Boolean, Scalar, Unit, Vector

# (shape, axis, result shape) for an empty Scalar; the last case collapses an axis of
# nonzero length, so the result is itself empty.
SHAPES = [
    ((0,), 0, ()),
    ((0,), None, ()),
    ((0, 3), 0, (3,)),
    ((3, 0), 1, (3,)),
    ((2, 0, 3), (0, 2), (0,)),
    ((3, 0), 0, (0,)),
]
SHAPE_IDS = ['1d_axis0', '1d_all', '0x3_axis0', '3x0_axis1', '2x0x3_axes02', '3x0_axis0']

# argmin() and argmax() accept a single axis only, so they skip the tuple-axis case.
UNDEFINED = [(method, *case)
             for method in ('mean', 'min', 'max', 'median', 'argmin', 'argmax')
             for case in SHAPES
             if method in ('mean', 'min', 'max', 'median') or not isinstance(case[1], tuple)]
UNDEFINED_IDS = [f'{c[0]}-{c[1]}-{c[2]}' for c in UNDEFINED]


@pytest.mark.parametrize(('shape', 'axis', 'result_shape'), SHAPES, ids=SHAPE_IDS)
def test_scalar_empty_reductions_sum_is_zero(shape: tuple[int, ...],
                                             axis: int | tuple[int, ...] | None,
                                             result_shape: tuple[int, ...]) -> None:
    """sum() over no elements is zero and unmasked, with the axes removed."""

    b = Scalar(np.empty(shape)).sum(axis=axis)
    assert type(b) is Scalar
    assert b.shape == result_shape
    assert np.all(b.values == 0.)
    assert not np.any(b.mask)


def test_scalar_empty_reductions_sum_ignores_the_mask() -> None:
    """sum() of an empty, fully masked Scalar is zero and unmasked."""

    b = Scalar(np.empty((0,)), mask=True).sum(axis=0)
    assert b == 0.
    assert not b.mask


def test_scalar_empty_reductions_sum_of_ints_is_int_zero() -> None:
    """sum() of an empty integer Scalar is the integer zero."""

    b = Scalar(np.empty((0,), dtype='int')).sum()
    assert type(b.values) is int
    assert b.values == 0


def test_scalar_empty_reductions_sum_of_vector_is_zero_vector() -> None:
    """sum() of an empty Vector is the zero Vector with the same item shape."""

    b = Vector(np.empty((0, 3))).sum()
    assert type(b) is Vector
    assert b.shape == ()
    assert np.all(b.values == [0., 0., 0.])
    assert not b.mask


def test_scalar_empty_reductions_sum_keeps_unit_and_derivative() -> None:
    """sum() of an empty Scalar keeps its unit and sums its derivative to zero."""

    a = Scalar(np.empty((0,)), unit=Unit.KM)
    a.insert_deriv('t', Scalar(np.empty((0,))))
    b = a.sum()
    assert b.units == Unit.KM
    assert b.d_dt == 0.
    assert not b.d_dt.mask


def test_scalar_empty_reductions_sum_as_builtin() -> None:
    """sum() of an empty Scalar with builtins=True is the float 0."""

    b = Scalar(np.empty((0,))).sum(builtins=True)
    assert type(b) is float
    assert b == 0.


def test_scalar_empty_reductions_boolean_count_is_zero() -> None:
    """Boolean.sum() of an empty Boolean counts zero, for True and for False values."""

    a = Boolean(np.empty((0,), dtype='bool'))
    assert a.sum(builtins=True) == 0
    assert a.sum(value=False, builtins=True) == 0


@pytest.mark.parametrize(('method', 'shape', 'axis', 'result_shape'), UNDEFINED,
                         ids=UNDEFINED_IDS)
def test_scalar_empty_reductions_undefined_results_are_masked(
        method: str, shape: tuple[int, ...], axis: int | tuple[int, ...] | None,
        result_shape: tuple[int, ...]) -> None:
    """Reductions with no value over no elements are fully masked, with the axes removed."""

    b = getattr(Scalar(np.empty(shape)), method)(axis=axis)
    assert type(b) is Scalar
    assert b.shape == result_shape
    assert np.all(b.mask)


def test_scalar_empty_reductions_masked_result_as_builtin() -> None:
    """max() of an empty Scalar with builtins=True returns the `masked` value."""

    assert Scalar(np.empty((0,))).max(builtins=True, masked=-1) == -1


def test_scalar_empty_reductions_mean_keeps_unit_and_masks_derivative() -> None:
    """mean() of an empty Scalar keeps its unit and masks its derivative."""

    a = Scalar(np.empty((0,)), unit=Unit.KM)
    a.insert_deriv('t', Scalar(np.empty((0,))))
    b = a.mean()
    assert b.units == Unit.KM
    assert b.mask
    assert b.d_dt.mask


def test_scalar_empty_reductions_mean_of_vector_is_masked_vector() -> None:
    """mean() of an empty Vector is a masked Vector with the same item shape."""

    b = Vector(np.empty((0, 3))).mean()
    assert type(b) is Vector
    assert b.shape == ()
    assert b.item == (3,)
    assert b.mask


@pytest.mark.parametrize(('shape', 'axis'), [((0,), 0), ((3, 0), 0), ((3, 0), 1)])
def test_scalar_empty_reductions_sort_returns_the_array_unchanged(
        shape: tuple[int, ...], axis: int) -> None:
    """sort() of an empty Scalar returns an empty Scalar of the same shape."""

    b = Scalar(np.empty(shape)).sort(axis=axis)
    assert type(b) is Scalar
    assert b.shape == shape
