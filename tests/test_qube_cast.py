##########################################################################################
# tests/test_qube_cast.py: Tests of Qube.cast
##########################################################################################

import numpy as np
import pytest

from polymath import Boolean, Matrix3, Qube, Scalar, Unit, Vector, Vector3


def test_qube_cast_to_the_same_class_returns_the_object() -> None:
    """An object already of the requested class is returned unchanged."""

    np.random.seed(6011)

    a = Vector3(np.random.randn(5, 3))

    assert a.cast(Vector3) is a


def test_qube_cast_to_an_incompatible_class_returns_the_object() -> None:
    """An object is returned unchanged when no listed class fits its numerator."""

    np.random.seed(6011)

    a = Vector3(np.random.randn(5, 3))

    assert a.cast(Matrix3) is a


def test_qube_cast_selects_the_first_suitable_class() -> None:
    """The first class in the list whose numerator fits is the one selected."""

    np.random.seed(6011)

    a = Vector(np.random.randn(5, 3))
    b = a.cast((Matrix3, Vector3, Vector))

    assert type(b) is Vector3


def test_qube_cast_preserves_the_values_and_the_mask() -> None:
    """A cast copies the values and the mask across unchanged."""

    np.random.seed(6011)

    values = np.random.randn(5, 3)
    mask = np.array([True, False, False, True, False])
    a = Vector(values, mask)
    b = a.cast(Vector3)

    assert np.all(b.values == values)
    assert np.all(b.mask == mask)
    assert b.shape == (5,)
    assert b.numer == (3,)


def test_qube_cast_preserves_the_unit() -> None:
    """A cast to a class that allows units keeps the unit."""

    np.random.seed(6011)

    a = Vector(np.random.randn(5, 3), unit=Unit.KM)
    b = a.cast(Vector3)

    assert b.unit_ == Unit.KM


def test_qube_cast_preserves_the_derivatives() -> None:
    """A cast carries the derivatives across."""

    np.random.seed(6011)

    deriv = np.random.randn(5, 3)
    a = Vector(np.random.randn(5, 3))
    a.insert_deriv('t', Vector(deriv))
    b = a.cast(Vector3)

    assert ('t' in b.derivs)
    assert np.all(b.d_dt.values == deriv)


def test_qube_cast_preserves_readonly_status() -> None:
    """A cast of a read-only object is read-only."""

    np.random.seed(6011)

    a = Vector(np.random.randn(5, 3)).as_readonly()

    assert a.cast(Vector3).readonly


def test_qube_cast_of_a_writable_object_is_writable() -> None:
    """A cast of a writable object is writable."""

    np.random.seed(6011)

    a = Vector(np.random.randn(5, 3))

    assert not a.cast(Vector3).readonly


def test_qube_cast_coerces_an_integer_object_to_a_float_class() -> None:
    """A class that disallows integers receives the values coerced to floats."""

    a = Vector(np.arange(6).reshape(2, 3))
    b = a.cast(Vector3)

    assert type(b) is Vector3
    assert b.is_float()
    assert b.values[1, 2] == 5.


def test_qube_cast_to_a_class_without_derivatives_is_rejected() -> None:
    """A class that disallows derivatives cannot receive an object that has them."""

    a = Scalar([1., 2.])
    a.insert_deriv('t', Scalar([3., 4.]))

    with pytest.raises(ValueError, match='derivatives are disallowed'):
        a.cast(Boolean)


def test_qube_cast_does_not_alter_the_source() -> None:
    """A cast leaves the object it was applied to unchanged."""

    np.random.seed(6011)

    a = Vector(np.random.randn(5, 3))
    a.insert_deriv('t', Vector(np.random.randn(5, 3)))
    a.cast(Vector3)

    assert type(a) is Vector
    assert ('t' in a.derivs)


def test_qube_cast_of_a_rank_zero_object_to_scalar() -> None:
    """A rank-zero object built by the fast constructor casts to a Scalar."""

    np.random.seed(6011)

    values = np.random.randn(5)
    a = Qube._new_from_parts(values, False, nrank=0)
    b = a.cast(Scalar)

    assert type(b) is Scalar
    assert np.all(b.values == values)


##########################################################################################
