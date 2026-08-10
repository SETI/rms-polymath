##########################################################################################
# tests/test_qube_unit.py
##########################################################################################

import numpy as np
import pytest

from polymath import Boolean, Matrix3, Quaternion, Scalar, Unit


def test_qube_unit_classes_for_which_units_are_not_allowed() -> None:
    """Classes for which units are not allowed."""

    a = Scalar((1.,2.,3.))
    assert a.units == None
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.KM)
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.CM)
    assert a.units == Unit.CM
    assert np.all(a.values == (1,2,3))
    with pytest.raises(ValueError):
        a.set_unit(Unit.DEG)   # incompatible
    a.set_unit(Unit.M)
    assert a.units == Unit.M
    assert np.all(a.values == (1,2,3))
    a = a.as_readonly()
    assert a.readonly
    with pytest.raises(ValueError):
        a.set_unit(Unit.KM)
    a.set_unit(Unit.KM, override=True)
    assert a.readonly
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))

    a = Matrix3([(1,0,0),(0,1,0),(0,0,1)])
    with pytest.raises(TypeError):
        a.set_unit(Unit.KM)
    a = Quaternion((1,0,0,0))
    with pytest.raises(TypeError):
        a.set_unit(Unit.KM)
    a = Boolean([True, False])
    with pytest.raises(TypeError):
        a.set_unit(Unit.KM)

    ##################################################################################
    # without_unit(self, recursive=True)
    ##################################################################################
    a = Scalar((1.,2.,3.), unit=Unit.KM)
    b = a.without_unit()
    assert a.units == Unit.KM
    assert b.units == None
    assert np.all(a.values == b.values)
    assert a.readonly == False
    assert b.readonly == False
    a = a.as_readonly()
    assert a.readonly == True
    b = a.without_unit()
    assert b.readonly == True
    assert b.units == None
    assert np.all(b.values == (1,2,3))

    ##################################################################################
    # into_unit(self, recursive=True)
    ##################################################################################
    a = Scalar((1.,2.,3.))
    assert a.units == None
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.M)
    assert a.units == Unit.M
    assert np.all(a.values == (1,2,3))
    vals = a.into_unit()
    assert np.all(vals == (1000, 2000, 3000))
    vals = a.into_unit(recursive=True)
    assert np.all(vals[0] == (1000, 2000, 3000))
    assert (vals[1] == {})
    a = Scalar((1.,2.,3.), unit=Unit.M)
    da_dt = Scalar((4., 5., 6.), unit=Unit.CM/Unit.S)
    a.insert_deriv('t', da_dt)
    vals = a.into_unit(recursive=False)
    assert np.all(vals == (1000, 2000, 3000))
    vals = a.into_unit(recursive=True)
    assert np.all(vals[0] == (1000, 2000, 3000))
    assert set(vals[1].keys()) == {'t'}
    assert np.all(vals[1]['t'] == (400000, 500000, 600000))

    a_nd = Scalar(np.random.rand(2, 3, 4), unit=Unit.M)
    vals = a_nd.into_unit()
    assert vals.shape == (2, 3, 4)
    expected = a_nd.values * 1000  # KM to M conversion
    assert np.allclose(vals, expected)

    a_unitless = Scalar((1., 2., 3.))
    vals = a_unitless.into_unit()
    assert np.all(vals == (1., 2., 3.))

    a_km = Scalar((1., 2., 3.), unit=Unit.KM)
    vals = a_km.into_unit()
    assert np.all(vals == (1., 2., 3.))

    ##################################################################################
    # confirm_unit(self, unit)
    ##################################################################################

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    result = a.confirm_unit(Unit.M)
    assert result == a

    result = a.confirm_unit(Unit.KM)
    assert result == a

    a_unitless = Scalar((1., 2., 3.))
    result = a_unitless.confirm_unit(None)
    assert result == a_unitless

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    with pytest.raises(ValueError):
        a.confirm_unit(Unit.DEG)

    a = Scalar((1., 2., 3.), unit=Unit.S)
    with pytest.raises(ValueError):
        a.confirm_unit(Unit.KM)

    a_nd = Scalar(np.random.rand(2, 3, 4), unit=Unit.M)
    result = a_nd.confirm_unit(Unit.CM)
    assert result == a_nd

    a_unitless = Scalar((1., 2., 3.))
    result = a_unitless.confirm_unit(None)
    assert result == a_unitless

    ##################################################################################
    # is_unitless(self)
    ##################################################################################

    a = Scalar((1., 2., 3.))
    assert a.is_unitless()

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    assert not a.is_unitless()

    a = Scalar((1., 2., 3.), unit=Unit.DEG)
    assert not a.is_unitless()

    a = Scalar((1., 2., 3.), unit=Unit.S)
    assert not a.is_unitless()

    a_nd = Scalar(np.random.rand(2, 3, 4))
    assert a_nd.is_unitless()
    a_nd = Scalar(np.random.rand(2, 3, 4), unit=Unit.M)
    assert not a_nd.is_unitless()

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    assert not a.is_unitless()
    a.set_unit(None)
    assert a.is_unitless()

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    b = a.without_unit()
    assert b.is_unitless()

    ##################################################################################
    # Additional comprehensive tests for set_unit
    ##################################################################################

    a_nd = Scalar(np.random.rand(2, 3, 4))
    a_nd.set_unit(Unit.KM)
    assert a_nd.units == Unit.KM
    assert a_nd.shape == (2, 3, 4)

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    a.set_unit(None)
    assert a.units == None
    assert a.is_unitless()

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    a.set_unit(Unit.M)
    assert a.units == Unit.M

    assert np.all(a.values == (1., 2., 3.))

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    a = a.as_readonly()
    with pytest.raises(ValueError):
        a.set_unit(Unit.M)


def test_qube_unit_test_with_read_only_object_and_override_true() -> None:
    """Test with read-only object and override=True."""

    a = Scalar((1.,2.,3.))
    assert a.units == None
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.KM)
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.CM)
    assert a.units == Unit.CM
    assert np.all(a.values == (1,2,3))
    with pytest.raises(ValueError):
        a.set_unit(Unit.DEG)   # incompatible
    a.set_unit(Unit.M)
    assert a.units == Unit.M
    assert np.all(a.values == (1,2,3))
    a = a.as_readonly()
    assert a.readonly
    with pytest.raises(ValueError):
        a.set_unit(Unit.KM)
    a.set_unit(Unit.KM, override=True)
    assert a.readonly
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    a = a.as_readonly()
    a.set_unit(Unit.M, override=True)
    assert a.units == Unit.M

    ##################################################################################
    # Additional comprehensive tests for without_unit
    ##################################################################################


def test_qube_unit_test_with_n_d_arrays() -> None:
    """Test with n-D arrays."""

    a = Scalar((1.,2.,3.))
    assert a.units == None
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.KM)
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.CM)
    assert a.units == Unit.CM
    assert np.all(a.values == (1,2,3))
    with pytest.raises(ValueError):
        a.set_unit(Unit.DEG)   # incompatible
    a.set_unit(Unit.M)
    assert a.units == Unit.M
    assert np.all(a.values == (1,2,3))
    a = a.as_readonly()
    assert a.readonly
    with pytest.raises(ValueError):
        a.set_unit(Unit.KM)
    a.set_unit(Unit.KM, override=True)
    assert a.readonly
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))

    a_nd = Scalar(np.random.rand(2, 3, 4), unit=Unit.KM)
    b_nd = a_nd.without_unit()
    assert b_nd.units == None
    assert b_nd.shape == (2, 3, 4)
    assert np.all(a_nd.values == b_nd.values)


def test_qube_unit_test_with_recursive_false_should_strip_derivatives() -> None:
    """Test with recursive=False (should strip derivatives)."""

    a = Scalar((1.,2.,3.))
    assert a.units == None
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.KM)
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.CM)
    assert a.units == Unit.CM
    assert np.all(a.values == (1,2,3))
    with pytest.raises(ValueError):
        a.set_unit(Unit.DEG)   # incompatible
    a.set_unit(Unit.M)
    assert a.units == Unit.M
    assert np.all(a.values == (1,2,3))
    a = a.as_readonly()
    assert a.readonly
    with pytest.raises(ValueError):
        a.set_unit(Unit.KM)
    a.set_unit(Unit.KM, override=True)
    assert a.readonly
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    da_dt = Scalar((4., 5., 6.), unit=Unit.M/Unit.S)
    a.insert_deriv('t', da_dt)
    b = a.without_unit(recursive=False)
    assert b.units == None
    assert len(b.derivs) == 0


def test_qube_unit_test_with_recursive_true_should_keep_derivatives_and_strip_t() -> None:
    """Test with recursive=True (should keep derivatives and strip their units)."""

    a = Scalar((1.,2.,3.))
    assert a.units == None
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.KM)
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.CM)
    assert a.units == Unit.CM
    assert np.all(a.values == (1,2,3))
    with pytest.raises(ValueError):
        a.set_unit(Unit.DEG)   # incompatible
    a.set_unit(Unit.M)
    assert a.units == Unit.M
    assert np.all(a.values == (1,2,3))
    a = a.as_readonly()
    assert a.readonly
    with pytest.raises(ValueError):
        a.set_unit(Unit.KM)
    a.set_unit(Unit.KM, override=True)
    assert a.readonly
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    da_dt = Scalar((4., 5., 6.), unit=Unit.M/Unit.S)
    a.insert_deriv('t', da_dt)
    b = a.without_unit(recursive=True)
    assert b.units == None
    assert len(b.derivs) == 1
    assert 't' in b.derivs

    assert b.derivs['t'].units == None


def test_qube_unit_test_that_original_object_is_unchanged() -> None:
    """Test that original object is unchanged."""

    a = Scalar((1.,2.,3.))
    assert a.units == None
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.KM)
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.CM)
    assert a.units == Unit.CM
    assert np.all(a.values == (1,2,3))
    with pytest.raises(ValueError):
        a.set_unit(Unit.DEG)   # incompatible
    a.set_unit(Unit.M)
    assert a.units == Unit.M
    assert np.all(a.values == (1,2,3))
    a = a.as_readonly()
    assert a.readonly
    with pytest.raises(ValueError):
        a.set_unit(Unit.KM)
    a.set_unit(Unit.KM, override=True)
    assert a.readonly
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    b = a.without_unit()
    assert a.units == Unit.KM
    assert b.units == None


def test_qube_unit_test_with_read_only_object() -> None:
    """Test with read-only object."""

    a = Scalar((1.,2.,3.))
    assert a.units == None
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.KM)
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.CM)
    assert a.units == Unit.CM
    assert np.all(a.values == (1,2,3))
    with pytest.raises(ValueError):
        a.set_unit(Unit.DEG)   # incompatible
    a.set_unit(Unit.M)
    assert a.units == Unit.M
    assert np.all(a.values == (1,2,3))
    a = a.as_readonly()
    assert a.readonly
    with pytest.raises(ValueError):
        a.set_unit(Unit.KM)
    a.set_unit(Unit.KM, override=True)
    assert a.readonly
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))

    a = Scalar((1., 2., 3.), unit=Unit.KM)
    a = a.as_readonly()
    b = a.without_unit()
    assert b.readonly
    assert b.units == None

    ##################################################################################
    # Additional comprehensive tests for into_unit
    ##################################################################################


def test_qube_unit_test_with_angle_units_values_are_in_standard_units_radians_i() -> None:
    """Test with angle units # Values are in standard units (radians), into_unit converts to degrees."""

    a = Scalar((1.,2.,3.))
    assert a.units == None
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.KM)
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.CM)
    assert a.units == Unit.CM
    assert np.all(a.values == (1,2,3))
    with pytest.raises(ValueError):
        a.set_unit(Unit.DEG)   # incompatible
    a.set_unit(Unit.M)
    assert a.units == Unit.M
    assert np.all(a.values == (1,2,3))
    a = a.as_readonly()
    assert a.readonly
    with pytest.raises(ValueError):
        a.set_unit(Unit.KM)
    a.set_unit(Unit.KM, override=True)
    assert a.readonly
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))

    a = Scalar(np.array([np.pi/2, np.pi, 3*np.pi/2]), unit=Unit.DEG)
    vals = a.into_unit()
    expected = np.array([90., 180., 270.])
    assert np.allclose(vals, expected)


def test_qube_unit_test_with_time_units_values_are_in_standard_units_seconds_in() -> None:
    """Test with time units # Values are in standard units (seconds), into_unit converts to minutes."""

    a = Scalar((1.,2.,3.))
    assert a.units == None
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.KM)
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.CM)
    assert a.units == Unit.CM
    assert np.all(a.values == (1,2,3))
    with pytest.raises(ValueError):
        a.set_unit(Unit.DEG)   # incompatible
    a.set_unit(Unit.M)
    assert a.units == Unit.M
    assert np.all(a.values == (1,2,3))
    a = a.as_readonly()
    assert a.readonly
    with pytest.raises(ValueError):
        a.set_unit(Unit.KM)
    a.set_unit(Unit.KM, override=True)
    assert a.readonly
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))

    a = Scalar(np.array([3600., 7200., 10800.]), unit=Unit.MIN)
    vals = a.into_unit()
    expected = np.array([60., 120., 180.])
    assert np.allclose(vals, expected)


def test_qube_unit_test_with_recursive_true_and_multiple_derivatives() -> None:
    """Test with recursive=True and multiple derivatives."""

    a = Scalar((1.,2.,3.))
    assert a.units == None
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.KM)
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.CM)
    assert a.units == Unit.CM
    assert np.all(a.values == (1,2,3))
    with pytest.raises(ValueError):
        a.set_unit(Unit.DEG)   # incompatible
    a.set_unit(Unit.M)
    assert a.units == Unit.M
    assert np.all(a.values == (1,2,3))
    a = a.as_readonly()
    assert a.readonly
    with pytest.raises(ValueError):
        a.set_unit(Unit.KM)
    a.set_unit(Unit.KM, override=True)
    assert a.readonly
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))

    a = Scalar((1., 2., 3.), unit=Unit.M)
    da_dt = Scalar((4., 5., 6.), unit=Unit.CM/Unit.S)
    da_dx = Scalar((7., 8., 9.), unit=Unit.M/Unit.KM)
    a.insert_deriv('t', da_dt)
    a.insert_deriv('x', da_dx)
    vals = a.into_unit(recursive=True)
    assert np.all(vals[0] == (1000, 2000, 3000))
    assert set(vals[1].keys()) == {'t', 'x'}

    assert np.allclose(vals[1]['t'], (400000, 500000, 600000))

    assert np.allclose(vals[1]['x'], (7000, 8000, 9000))


def test_qube_unit_test_with_n_d_arrays_and_recursive_true() -> None:
    """Test with n-D arrays and recursive=True."""

    a = Scalar((1.,2.,3.))
    assert a.units == None
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.KM)
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))
    a.set_unit(Unit.CM)
    assert a.units == Unit.CM
    assert np.all(a.values == (1,2,3))
    with pytest.raises(ValueError):
        a.set_unit(Unit.DEG)   # incompatible
    a.set_unit(Unit.M)
    assert a.units == Unit.M
    assert np.all(a.values == (1,2,3))
    a = a.as_readonly()
    assert a.readonly
    with pytest.raises(ValueError):
        a.set_unit(Unit.KM)
    a.set_unit(Unit.KM, override=True)
    assert a.readonly
    assert a.units == Unit.KM
    assert np.all(a.values == (1,2,3))

    a_nd = Scalar(np.random.rand(2, 3, 4), unit=Unit.M)
    da_dt_nd = Scalar(np.random.rand(2, 3, 4), unit=Unit.CM/Unit.S)
    a_nd.insert_deriv('t', da_dt_nd)
    vals = a_nd.into_unit(recursive=True)
    assert vals[0].shape == (2, 3, 4)
    assert vals[1]['t'].shape == (2, 3, 4)


##########################################################################################
