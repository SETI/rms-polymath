##########################################################################################
# tests/test_qube_mask_nans_infs.py: Qube.mask_nans_infs()
#
# This is the method that keeps NaNs and infinities out of PolyMath objects, so it is the
# one place where a test must supply such a value.
##########################################################################################

import numpy as np
import pytest

from polymath import Boolean, Matrix, Matrix3, Scalar, Vector3

NAN = float('nan')
INF = float('inf')


def test_qube_mask_nans_infs_masks_a_nan() -> None:
    """An element holding a NaN is masked."""

    obj = Scalar([1., NAN, 3.])
    obj.mask_nans_infs()
    assert np.array_equal(obj.mask, [False, True, False])


def test_qube_mask_nans_infs_masks_a_positive_infinity() -> None:
    """An element holding a positive infinity is masked."""

    obj = Scalar([1., INF, 3.])
    obj.mask_nans_infs()
    assert np.array_equal(obj.mask, [False, True, False])


def test_qube_mask_nans_infs_masks_a_negative_infinity() -> None:
    """An element holding a negative infinity is masked."""

    obj = Scalar([1., -INF, 3.])
    obj.mask_nans_infs()
    assert np.array_equal(obj.mask, [False, True, False])


def test_qube_mask_nans_infs_replaces_with_the_default() -> None:
    """A disallowed value is replaced by _default_for(), which is 1 for a Scalar."""

    obj = Scalar([1., NAN, 3.])
    obj.mask_nans_infs()
    assert obj.vals[1] == 1.


def test_qube_mask_nans_infs_returns_this_object() -> None:
    """The object returned is this object, not a copy."""

    obj = Scalar([1., NAN])
    assert obj.mask_nans_infs() is obj


def test_qube_mask_nans_infs_modifies_in_place() -> None:
    """The change is visible through the original reference."""

    obj = Scalar([1., NAN])
    Scalar.mask_nans_infs(obj)
    assert obj.mask[1]


def test_qube_mask_nans_infs_keeps_the_finite_values() -> None:
    """A finite value is left exactly as it was."""

    obj = Scalar([1., NAN, 3.])
    obj.mask_nans_infs()
    assert np.array_equal(obj.vals[[0, 2]], [1., 3.])


def test_qube_mask_nans_infs_of_a_clean_object_masks_nothing() -> None:
    """An object holding no disallowed value is left unmasked."""

    obj = Scalar([1., 2., 3.])
    obj.mask_nans_infs()
    assert not np.any(obj.mask)


def test_qube_mask_nans_infs_preserves_an_existing_mask() -> None:
    """An element already masked stays masked."""

    obj = Scalar([1., 2., NAN], [True, False, False])
    obj.mask_nans_infs()
    assert np.array_equal(obj.mask, [True, False, True])


def test_qube_mask_nans_infs_masks_a_whole_item() -> None:
    """One disallowed value anywhere in an item masks that element."""

    obj = Vector3([[1., 2., 3.], [4., NAN, 6.], [7., 8., 9.]])
    obj.mask_nans_infs()
    assert np.array_equal(obj.mask, [False, True, False])


def test_qube_mask_nans_infs_replaces_a_whole_item() -> None:
    """The entire item is replaced, not just the disallowed value."""

    obj = Vector3([[1., 2., 3.], [4., NAN, 6.]])
    obj.mask_nans_infs()
    assert np.array_equal(obj.vals[1], [1., 1., 1.])


def test_qube_mask_nans_infs_uses_the_subclass_default() -> None:
    """A Matrix3 item is replaced by the identity matrix, its _DEFAULT_VALUE."""

    obj = Matrix3([np.identity(3), np.full((3, 3), NAN)])
    obj.mask_nans_infs()
    assert np.array_equal(obj.vals[1], np.identity(3))


def test_qube_mask_nans_infs_of_a_shapeless_scalar() -> None:
    """A shapeless Scalar is masked entirely."""

    obj = Scalar(NAN)
    obj.mask_nans_infs()
    assert obj.mask


def test_qube_mask_nans_infs_replaces_a_shapeless_scalar() -> None:
    """A shapeless Scalar takes the default value."""

    obj = Scalar(NAN)
    obj.mask_nans_infs()
    assert obj.vals == 1.


def test_qube_mask_nans_infs_of_a_shapeless_vector() -> None:
    """A shapeless Vector3 with one disallowed component is masked entirely."""

    obj = Vector3([1., INF, 3.])
    obj.mask_nans_infs()
    assert obj.mask


def test_qube_mask_nans_infs_replaces_a_shapeless_vector() -> None:
    """A shapeless Vector3 takes the default item."""

    obj = Vector3([1., INF, 3.])
    obj.mask_nans_infs()
    assert np.array_equal(obj.vals, [1., 1., 1.])


def test_qube_mask_nans_infs_leaves_a_derivative_unmasked() -> None:
    """Derivatives are not examined, so a disallowed value in one is left in place."""

    obj = Scalar([1., 2., 3.])
    obj.insert_deriv('t', Scalar([NAN, 5., 6.]))
    obj.mask_nans_infs()
    assert not np.any(obj.d_dt.mask)


def test_qube_mask_nans_infs_leaves_a_derivative_value_alone() -> None:
    """A disallowed value in a derivative is not replaced either."""

    obj = Scalar([1., 2., 3.])
    obj.insert_deriv('t', Scalar([NAN, 5., 6.]))
    obj.mask_nans_infs()
    assert np.isnan(obj.d_dt.vals[0])


def test_qube_mask_nans_infs_does_not_mask_a_derivative_of_a_masked_element() -> None:
    """Masking an element of the object leaves its derivative unmasked."""

    obj = Scalar([NAN, 2., 3.])
    obj.insert_deriv('t', Scalar([4., 5., 6.]))
    obj.mask_nans_infs()
    assert not np.any(obj.d_dt.mask)


def test_qube_mask_nans_infs_keeps_its_derivatives() -> None:
    """The derivatives survive the operation."""

    obj = Scalar([NAN, 2., 3.])
    obj.insert_deriv('t', Scalar([4., 5., 6.]))
    obj.mask_nans_infs()
    assert 't' in obj.derivs


def test_qube_mask_nans_infs_handles_a_denominator() -> None:
    """An object with a denominator is masked by element."""

    values = np.ones((2, 3, 3, 2))
    values[1, 0, 0, 0] = NAN
    obj = Matrix(values, drank=1)
    obj.mask_nans_infs()
    assert np.array_equal(obj.mask, [False, True])


def test_qube_mask_nans_infs_leaves_a_boolean_alone() -> None:
    """A Boolean cannot hold a disallowed value, so it is untouched."""

    obj = Boolean([True, False])
    obj.mask_nans_infs()
    assert not np.any(obj.mask)


def test_qube_mask_nans_infs_leaves_an_integer_alone() -> None:
    """An integer Scalar cannot hold a disallowed value, so it is untouched."""

    obj = Scalar([1, 2, 3])
    obj.mask_nans_infs()
    assert not np.any(obj.mask)


def test_qube_mask_nans_infs_accepts_a_clean_read_only_object() -> None:
    """Nothing needs to change, so a read-only object is returned untouched."""

    obj = Scalar([1., 2.]).as_readonly()
    assert not np.any(obj.mask_nans_infs().mask)


def test_qube_mask_nans_infs_rejects_a_read_only_object() -> None:
    """A read-only object cannot be modified in place."""

    obj = Scalar([1., NAN]).as_readonly()
    with pytest.raises(ValueError, match='read-only'):
        obj.mask_nans_infs()

##########################################################################################
