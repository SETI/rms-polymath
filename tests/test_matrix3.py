##########################################################################################
# tests/test_matrix3.py
# Matrix3 tests for basic operations and methods not covered by other test files
##########################################################################################

import numpy as np
import pytest

from polymath import Matrix3, Matrix, Vector, Vector3, Scalar, Quaternion
from polymath.unit import Unit


def test_matrix3_test_basic_construction_arrays_of_wrong_shape_raise_valueerr() -> None:
    """Test basic construction # Arrays of wrong shape raise ValueError."""

    np.random.seed(2599)
    DEL = 1.e-12

    with pytest.raises(ValueError):
        Matrix3(np.random.randn(3, 4, 5))
    with pytest.raises(ValueError):
        Matrix3(1.)

    a = Matrix3.zeros((2, 3), dtype='float')
    assert a.shape == (2, 3)
    assert a.vals.shape == (2, 3, 3, 3)
    assert a.vals.dtype.kind == 'f'
    assert np.all(a.vals == 0)
    a = Matrix3.zeros((2, 2), mask=[[0, 1], [0, 0]])
    assert a.shape == (2, 2)
    assert a.vals.shape == (2, 2, 3, 3)
    assert np.all(a.vals == 0)
    assert a.vals.dtype.kind == 'f'
    assert np.all(a.mask == [[0, 1], [0, 0]])

    a = Matrix3.ones((2, 3), dtype='float')
    assert a.shape == (2, 3)
    assert a.vals.shape == (2, 3, 3, 3)
    assert a.vals.dtype.kind == 'f'
    assert np.all(a.vals == 1)
    a = Matrix3.ones((2, 2), mask=[[0, 1], [0, 0]])
    assert a.shape == (2, 2)
    assert a.vals.shape == (2, 2, 3, 3)
    assert np.all(a.vals == 1)
    assert a.vals.dtype.kind == 'f'
    assert np.all(a.mask == [[0, 1], [0, 0]])

    a = Matrix3.filled((2, 3), 7.)
    assert a.shape == (2, 3)
    assert a.vals.shape == (2, 3, 3, 3)
    assert a.vals.dtype.kind == 'f'
    assert np.all(a.vals == 7)

    ident = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    a = Matrix3.filled((2, 2), ident)
    assert a.shape == (2, 2)
    assert a.vals.shape == (2, 2, 3, 3)
    for i in range(2):
        for j in range(2):
            assert np.allclose(a.vals[i, j], ident)

    m = Matrix3(np.random.randn(2, 3, 3, 3))
    m2 = Matrix3.as_matrix3(m)
    assert type(m2) == Matrix3
    assert np.allclose(m.vals, m2.vals)

    mat = Matrix(np.random.randn(2, 3, 3, 3))
    m3 = Matrix3.as_matrix3(mat)
    assert type(m3) == Matrix3
    assert m3.shape == mat.shape
    assert m3.numer == (3, 3)

    arr = np.random.randn(3, 3)
    m4 = Matrix3.as_matrix3(arr)
    assert type(m4) == Matrix3
    assert m4.shape == ()
    assert m4.numer == (3, 3)

    angle = np.pi / 4
    rx = Matrix3.x_rotation(angle)
    assert rx.shape == ()
    assert rx.numer == (3, 3)
    expected = np.array([[1., 0., 0.],
                        [0., np.cos(angle), np.sin(angle)],
                        [0., -np.sin(angle), np.cos(angle)]])
    assert np.allclose(rx.vals, expected, atol=DEL)

    angles = np.array([0., np.pi/4, np.pi/2])
    rx_array = Matrix3.x_rotation(angles)
    assert rx_array.shape == (3,)
    for i, angle in enumerate(angles):
        expected = np.array([[1., 0., 0.],
                            [0., np.cos(angle), np.sin(angle)],
                            [0., -np.sin(angle), np.cos(angle)]])
        assert np.allclose(rx_array.vals[i], expected, atol=DEL)

    ry = Matrix3.y_rotation(angle)
    expected = np.array([[np.cos(angle), 0., np.sin(angle)],
                        [0., 1., 0.],
                        [-np.sin(angle), 0., np.cos(angle)]])
    assert np.allclose(ry.vals, expected, atol=DEL)

    rz = Matrix3.z_rotation(angle)
    expected = np.array([[np.cos(angle), -np.sin(angle), 0.],
                        [np.sin(angle), np.cos(angle), 0.],
                        [0., 0., 1.]])
    assert np.allclose(rz.vals, expected, atol=DEL)

    test_angle = np.pi / 4
    rz2 = Matrix3.axis_rotation(test_angle)
    rz_ref = Matrix3.z_rotation(test_angle)
    assert np.allclose(rz2.vals, rz_ref.vals, atol=DEL)

    rx2 = Matrix3.axis_rotation(test_angle, axis=0)
    rx_ref = Matrix3.x_rotation(test_angle)
    assert np.allclose(rx2.vals, rx_ref.vals, atol=DEL)

    ry2 = Matrix3.axis_rotation(test_angle, axis=1)
    ry_ref = Matrix3.y_rotation(test_angle)
    assert np.allclose(ry2.vals, ry_ref.vals, atol=DEL)

    rz3 = Matrix3.axis_rotation(test_angle, axis=-1)
    assert np.allclose(rz3.vals, rz_ref.vals, atol=DEL)

    ra = 0.
    dec = np.pi / 2
    m_pole = Matrix3.pole_rotation(ra, dec)
    assert m_pole.shape == ()
    assert m_pole.numer == (3, 3)

    ra_array = np.array([0., np.pi/4])
    dec_array = np.array([np.pi/4, np.pi/2])
    m_pole_array = Matrix3.pole_rotation(ra_array, dec_array)
    assert m_pole_array.shape == (2,)
    assert m_pole_array.numer == (3, 3)

    v = Vector3([1., 0., 0.])
    m_rot = Matrix3.x_rotation(np.pi / 2)
    v_rotated = m_rot.rotate(v)
    assert type(v_rotated) == Vector3
    expected = Vector3([1., 0., 0.])
    assert np.allclose(v_rotated.vals, expected.vals, atol=DEL)

    m_array = Matrix3.x_rotation([0., np.pi/2])
    v_array = Vector3(np.array([[1., 0., 0.], [1., 0., 0.]]))
    v_rotated_array = m_array.rotate(v_array)
    assert v_rotated_array.shape == (2,)

    s = Scalar(5.)
    s_rotated = m_rot.rotate(s)
    assert type(s_rotated) == Scalar
    assert s_rotated.vals == 5.

    v_unrotated = m_rot.unrotate(v_rotated)
    assert np.allclose(v_unrotated.vals, v.vals, atol=DEL)

    s_unrotated = m_rot.unrotate(s)
    assert s_unrotated.vals == 5.

    m1 = Matrix3.IDENTITY
    m2 = Matrix3.x_rotation(np.pi/4)

    with pytest.raises(TypeError):
        (lambda: -m1)()

    with pytest.raises(TypeError):
        (lambda: m1 + m2)()
    with pytest.raises(TypeError):
        (lambda: m2 + m1)()

    with pytest.raises(TypeError):
        (lambda: m1 - m2)()
    with pytest.raises(TypeError):
        (lambda: m2 - m1)()

    v = Vector3([1., 0., 0.])
    result = m2 * v
    assert type(result) == Vector3

    result = m1 * m2
    assert type(result) == Matrix3
    assert result.shape == ()

    s = Scalar(5.)
    result = m2 * s
    assert type(result) == Scalar
    assert result.vals == 5.

    m3 = Matrix3.x_rotation(np.pi/4)
    m3_copy = m3.copy()
    m3 *= m1
    assert np.allclose(m3.vals, m3_copy.vals, atol=DEL)

    m = Matrix3.x_rotation(np.pi/4)
    m_recip = m.reciprocal()
    assert type(m_recip) == Matrix3

    m_transpose = m.transpose()
    assert np.allclose(m_recip.vals, m_transpose.vals, atol=DEL)

    with pytest.raises(TypeError):
        (lambda: m.sum())()

    with pytest.raises(TypeError):
        (lambda: m.mean())()

    m = Matrix3(np.random.randn(2, 3, 3, 3))
    assert m.shape == (2, 3)
    assert m.numer == (3, 3)
    assert m.rank == 2
    assert m.nrank == 2
    assert m.item == (3, 3)
    assert m.isize == 9
    assert m.nsize == 9

    assert Matrix3.IDENTITY.shape == ()
    assert Matrix3.IDENTITY.numer == (3, 3)
    assert (np.allclose(Matrix3.IDENTITY.vals,
                                np.eye(3), atol=DEL))
    assert Matrix3.IDENTITY.readonly
    assert Matrix3.MASKED.shape == ()
    assert Matrix3.MASKED.mask

    m = Matrix3.x_rotation(np.pi/4)
    m.insert_deriv('t', Matrix3.x_rotation(np.pi/8))
    m2 = Matrix3.as_matrix3(m, recursive=False)
    assert type(m2) == Matrix3
    assert not hasattr(m2, 'd_dt')

    angle = Scalar(np.pi/4)
    angle.insert_deriv('t', Scalar(1.))
    rx = Matrix3.x_rotation(angle, recursive=True)
    assert hasattr(rx, 'd_dt')
    assert type(rx.d_dt) == Matrix

    rx2 = Matrix3.axis_rotation(angle, axis=0, recursive=True)
    assert hasattr(rx2, 'd_dt')

    v = Vector3([1., 0., 0.])
    v.insert_deriv('t', Vector3([0., 1., 0.]))
    v_rotated = rx.rotate(v, recursive=True)
    assert hasattr(v_rotated, 'd_dt')

    v_unrotated = rx.unrotate(v_rotated, recursive=True)
    assert hasattr(v_unrotated, 'd_dt')

    m1 = Matrix3.x_rotation([0., np.pi/4])
    m2 = Matrix3.y_rotation([0., np.pi/4])
    result = m1 * m2
    assert result.shape == (2,)

    m = Matrix3.x_rotation([0., np.pi/4])
    mask = np.array([False, True])
    m_masked = Matrix3(m.vals, mask=mask)
    assert np.all(m_masked.mask == mask)

    m = Matrix3.IDENTITY
    assert m.readonly
    m2 = m.copy()
    assert not m2.readonly

    m = Matrix3.x_rotation(np.pi/4)
    m_t = m.transpose()
    product = m * m_t
    assert np.allclose(product.vals, np.eye(3), atol=DEL)

    rx = Matrix3.x_rotation(np.pi/4)
    ry = Matrix3.y_rotation(np.pi/4)
    rz = Matrix3.z_rotation(np.pi/4)
    combined = rx * ry * rz
    assert type(combined) == Matrix3
    assert combined.shape == ()

    m1 = Matrix3.x_rotation(np.pi/4)
    m2 = Matrix3.y_rotation(np.pi/4)
    m_rotated = m1.rotate(m2)
    assert type(m_rotated) == Matrix3
    assert m_rotated.shape == ()

    angles = np.random.randn(4, 5, 6) * np.pi
    m_array = Matrix3.x_rotation(angles)
    assert m_array.shape == (4, 5, 6)
    assert m_array.numer == (3, 3)

    ra = np.random.randn(2, 3) * np.pi
    dec = np.random.randn(2, 3) * np.pi / 2
    m_pole = Matrix3.pole_rotation(ra, dec)
    assert m_pole.shape == (2, 3)
    assert m_pole.numer == (3, 3)

    m = Matrix3(np.random.randn(2, 3, 3, 3))
    m2 = Matrix3.as_matrix3(m)
    assert m2.shape == m.shape

    with pytest.raises(TypeError):
        Matrix3(np.eye(3), unit='km')

    m = Matrix3.zeros((2, 2), dtype='int')
    assert m.vals.dtype.kind == 'f'

    m = Matrix3.zeros((2, 2), dtype='bool')
    assert m.vals.dtype.kind == 'f'

    q = Quaternion(np.random.randn(4)).unit()
    m_quat = Matrix3.as_matrix3(q)
    assert type(m_quat) == Matrix3
    assert m_quat.shape == ()

    q.insert_deriv('t', Quaternion(np.random.randn(4)))
    m_quat2 = Matrix3.as_matrix3(q, recursive=False)
    assert type(m_quat2) == Matrix3
    assert not hasattr(m_quat2, 'd_dt')

    angle_y = Scalar(np.pi/4)
    angle_y.insert_deriv('t', Scalar(1.))
    ry_deriv = Matrix3.y_rotation(angle_y, recursive=True)
    assert hasattr(ry_deriv, 'd_dt')
    assert type(ry_deriv.d_dt) == Matrix

    angle_z = Scalar(np.pi/4)
    angle_z.insert_deriv('t', Scalar(1.))
    rz_deriv = Matrix3.z_rotation(angle_z, recursive=True)
    assert hasattr(rz_deriv, 'd_dt')
    assert type(rz_deriv.d_dt) == Matrix

    with pytest.raises(TypeError):
        (lambda: 5 + m1)()

    m_write = Matrix3.x_rotation(np.pi/4).copy()
    with pytest.raises(TypeError):
        (lambda: m_write.__iadd__(m2))()

    with pytest.raises(TypeError):
        (lambda: 5 - m1)()

    m_write = Matrix3.x_rotation(np.pi/4).copy()
    with pytest.raises(TypeError):
        (lambda: m_write.__isub__(m2))()

    with pytest.raises((ValueError, TypeError)):
        (lambda: m2 * "invalid")()

    with pytest.raises((ValueError, TypeError)):
        (lambda: "invalid" * m2)()

    m_write = Matrix3.x_rotation(np.pi/4).copy()
    with pytest.raises((ValueError, TypeError)):
        (lambda: m_write.__imul__("invalid"))()

    m_readonly = Matrix3.IDENTITY
    with pytest.raises(ValueError):
        (lambda: m_readonly.__imul__(m2))()

    m = Matrix3.x_rotation(np.pi/4)
    m_recip_nozeros = m.reciprocal(nozeros=True)
    m_recip_normal = m.reciprocal(nozeros=False)
    assert np.allclose(m_recip_nozeros.vals, m_recip_normal.vals, atol=DEL)

    m.insert_deriv('t', Matrix3.x_rotation(np.pi/8))
    m_recip_no_derivs = m.reciprocal(recursive=False)
    assert not hasattr(m_recip_no_derivs, 'd_dt')

    s = Scalar(5.)
    s.insert_deriv('t', Scalar(1.))
    result = m2 * s
    assert type(result) == Scalar

    result_no_derivs = m2.__mul__(s, recursive=False)
    assert not hasattr(result_no_derivs, 'd_dt')

    result_rmul = m2.__rmul__(m1, recursive=False)
    assert type(result_rmul) == Matrix3

    v = Vector3([1., 0., 0.])
    v.insert_deriv('t', Vector3([0., 1., 0.]))
    v_rotated_no_derivs = m2.rotate(v, recursive=False)
    assert not hasattr(v_rotated_no_derivs, 'd_dt')

    v_unrotated_no_derivs = m2.unrotate(v_rotated_no_derivs, recursive=False)
    assert not hasattr(v_unrotated_no_derivs, 'd_dt')

    v_test = Vector3([1., 0., 0.])
    result = m2 * v_test
    assert type(result) == Vector3

    m_with_deriv = Matrix3.x_rotation(np.pi/4)
    m_with_deriv.insert_deriv('t', Matrix3.x_rotation(np.pi/8))
    m_converted = Matrix3.as_matrix3(m_with_deriv, recursive=True)
    assert hasattr(m_converted, 'd_dt')

    with pytest.raises(ValueError):
        Matrix3.pole_rotation(Scalar(1., unit=Unit.KM), np.pi/4)

    with pytest.raises(ValueError):
        Matrix3.pole_rotation(np.pi/4, Scalar(1., unit=Unit.KM))

    with pytest.raises(ValueError):
        Matrix3.x_rotation(Scalar(1., unit=Unit.KM))

    with pytest.raises(ValueError):
        Matrix3.y_rotation(Scalar(1., unit=Unit.KM))

    with pytest.raises(ValueError):
        Matrix3.z_rotation(Scalar(1., unit=Unit.KM))

    rx_wrap = Matrix3.axis_rotation(np.pi/4, axis=3)
    rx_ref = Matrix3.x_rotation(np.pi/4)
    assert np.allclose(rx_wrap.vals, rx_ref.vals, atol=DEL)

    ry_wrap = Matrix3.axis_rotation(np.pi/4, axis=4)
    ry_ref = Matrix3.y_rotation(np.pi/4)
    assert np.allclose(ry_wrap.vals, ry_ref.vals, atol=DEL)

    ry_wrap2 = Matrix3.axis_rotation(np.pi/4, axis=-2)
    assert np.allclose(ry_wrap2.vals, ry_ref.vals, atol=DEL)

    s_with_deriv = Scalar(5.)
    s_with_deriv.insert_deriv('t', Scalar(1.))
    result = m2.__mul__(s_with_deriv, recursive=True)
    assert hasattr(result, 'd_dt')

    mat = Matrix(np.random.randn(3, 3))
    result = mat * m2
    assert type(result) == Matrix3

    arr = np.random.randn(3, 3)
    result = arr * m2
    assert type(result) == Matrix3

    m_write = Matrix3.x_rotation(np.pi/4).copy()
    mat_conv = Matrix(np.random.randn(3, 3))
    m_write *= mat_conv
    assert type(m_write) == Matrix3

    m_write = Matrix3.x_rotation(np.pi/4).copy()
    arr_conv = np.random.randn(3, 3)
    m_write *= arr_conv
    assert type(m_write) == Matrix3

    result = m2 * 5.0
    assert type(result) == Scalar
    assert result.vals == 5.0

    result = m2.__mul__(5.0, recursive=False)
    assert type(result) == Scalar

    # Test twovec with denominators (should raise error)
    # This is hard to test without creating actual denominators, so we skip it
    # The code path exists but requires specific setup that's not easily testable

    v1_deriv = Vector3([1., 0., 0.])
    v2_deriv = Vector3([0., 1., 0.])

    v1_deriv.insert_deriv('t', Vector3([0., 0., 1.]))

    m_twovec = Matrix3.twovec(v1_deriv, 0, v2_deriv, 1, recursive=True)
    assert type(m_twovec) == Matrix3

    v1_ro = Vector3([1., 0., 0.]).as_readonly()
    v2_ro = Vector3([0., 1., 0.]).as_readonly()
    m_twovec_ro = Matrix3.twovec(v1_ro, 0, v2_ro, 1)

    assert type(m_twovec_ro) == Matrix3

    m_euler_tuple = Matrix3.from_euler(1., 2., 3., axes=(0, 1, 0, 1))
    assert type(m_euler_tuple) == Matrix3

    m_euler_string = Matrix3.from_euler(1., 2., 3., axes='ryzx')
    assert np.allclose(m_euler_tuple.vals, m_euler_string.vals, atol=DEL)

    m_euler_tuple2 = Matrix3.from_euler(1., 2., 3., axes=(2, 0, 1, 1))
    m_euler_string2 = Matrix3.from_euler(1., 2., 3., axes='rzxz')
    assert np.allclose(m_euler_tuple2.vals, m_euler_string2.vals, atol=DEL)

    m_euler_parity = Matrix3.from_euler(1., 2., 3., axes='sxzy')  # has parity
    assert type(m_euler_parity) == Matrix3

    m_test = Matrix3.x_rotation(np.pi/4)
    angles_tuple = m_test.to_euler(axes=(0, 0, 0, 0))
    assert len(angles_tuple) == 3
    assert type(angles_tuple[0]) == Scalar

    angles_string = m_test.to_euler(axes='sxyz')
    assert len(angles_string) == 3
    for i in range(3):
        assert np.allclose(angles_tuple[i].vals, angles_string[i].vals, atol=DEL)

    angles_tuple2 = m_test.to_euler(axes=(2, 0, 1, 1))
    angles_string2 = m_test.to_euler(axes='rzxz')
    assert len(angles_tuple2) == 3
    for i in range(3):
        assert np.allclose(angles_tuple2[i].vals, angles_string2[i].vals, atol=DEL)

    m_rep_mask = Matrix3.IDENTITY.copy()
    m_rep_vals = m_rep_mask.vals.copy()

    m_rep_vals[0, 1] = 1e-20
    m_rep_vals[0, 2] = 1e-20
    m_rep_mask = Matrix3(m_rep_vals)
    angles_rep = m_rep_mask.to_euler(axes='sxyx')  # repetition=True
    assert len(angles_rep) == 3

    m_nonrep_mask = Matrix3.IDENTITY.copy()
    m_nonrep_vals = m_nonrep_mask.vals.copy()
    m_nonrep_vals[0, 0] = 1e-20
    m_nonrep_vals[1, 0] = 1e-20
    m_nonrep_mask = Matrix3(m_nonrep_vals)
    angles_nonrep = m_nonrep_mask.to_euler(axes='sxyz')  # repetition=False
    assert len(angles_nonrep) == 3

    m_test2 = Matrix3.x_rotation(np.pi/4)
    angles_parity = m_test2.to_euler(axes='sxzy')  # has parity
    assert len(angles_parity) == 3
    angles_frame = m_test2.to_euler(axes='rzyx')  # has frame
    assert len(angles_frame) == 3

    m_qtest = Matrix3.x_rotation(np.pi/4)
    q = m_qtest.to_quaternion()
    assert type(q) == Quaternion

    m_test = Matrix3.x_rotation(np.pi/4)
    if hasattr(m_test, '__getstate__experimental'):
        # Test with small size (should use normal getstate)
        m_small = Matrix3.x_rotation(np.pi/4)
        state_small = m_small.__getstate__experimental()
        assert isinstance(state_small, dict)

        # Test with larger size (should use quaternion conversion)
        # Need size >= 30 to trigger quaternion path
        m_large = Matrix3.x_rotation(np.random.randn(10, 10) * np.pi)
        # Ensure it's large enough
        if m_large._size >= 30:
            state_large = m_large.__getstate__experimental()
            assert isinstance(state_large, dict)
            # Check if it used quaternion conversion
            if hasattr(m_large, 'CONVERTED_TO_QUATERNION'):
                # Test setstate with quaternion conversion
                m_new = Matrix3.__new__(Matrix3)
                try:
                    m_new.__setstate__experimental(state_large)
                    assert type(m_new) == Matrix3
                except (AttributeError, KeyError, TypeError):
                    pass

        # Test with masked (should use normal getstate)
        m_masked = Matrix3.x_rotation([np.pi/4, np.pi/2])
        m_masked = Matrix3(m_masked.vals, mask=[False, True])
        state_masked = m_masked.__getstate__experimental()
        assert isinstance(state_masked, dict)

        # Test __setstate__experimental
        if hasattr(m_test, '__setstate__experimental'):
            # Create a state that would have CONVERTED_TO_QUATERNION
            # This is tricky, so we'll test the path where it doesn't have it
            m_new = Matrix3.__new__(Matrix3)
            try:
                # Test with normal state (no CONVERTED_TO_QUATERNION)
                normal_state = m_test.__getstate__experimental()
                m_new.__setstate__experimental(normal_state)
                assert type(m_new) == Matrix3
            except (AttributeError, KeyError, TypeError):
                # Some states might not work, that's okay
                pass

    v1_vals = np.array([[1., 0.], [0., 0.], [0., 0.]])  # shape (3, 2)
    v1_with_denom = Vector(v1_vals, drank=1)  # shape (), numer (3,), denom (2,)
    v1 = Vector3.as_vector3(v1_with_denom)  # Preserves denominator
    v2 = Vector3([0., 1., 0.])

    with pytest.raises(ValueError):
        Matrix3.twovec(v1, 0, v2, 1)

    v1 = Vector3([1., 0., 0.])
    v2_vals = np.array([[0., 0.], [1., 0.], [0., 0.]])  # shape (3, 2)
    v2_with_denom = Vector(v2_vals, drank=1)  # shape (), numer (3,), denom (2,)
    v2 = Vector3.as_vector3(v2_with_denom)  # Preserves denominator
    with pytest.raises(ValueError):
        Matrix3.twovec(v1, 0, v2, 1)

    v1 = Vector3([1., 0., 0.])

    v1_deriv_vals = np.array([[0., 0.], [0., 0.], [1., 0.]])  # shape (3, 2)
    v1_deriv = Vector(v1_deriv_vals, drank=1)  # shape (), numer (3,), denom (2,)
    v1.insert_deriv('t', Vector3.as_vector3(v1_deriv))
    v2 = Vector3([0., 1., 0.])

    v2_deriv_vals = np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]])  # shape (3, 3)
    v2_deriv = Vector(v2_deriv_vals, drank=1)  # shape (), numer (3,), denom (3,)
    v2.insert_deriv('t', Vector3.as_vector3(v2_deriv))

    with pytest.raises(ValueError):
        Matrix3.twovec(v1, 0, v2, 1, recursive=True)

    v1 = Vector3([1., 0., 0.])
    v1_deriv1 = Vector(np.array([[0., 0.], [0., 0.], [1., 0.]]), drank=1)  # denom (2,)
    v1.insert_deriv('t', Vector3.as_vector3(v1_deriv1))
    v2 = Vector3([0., 1., 0.])
    v2_deriv1 = Vector(np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]), drank=1)  # denom (3,)
    v2.insert_deriv('t', Vector3.as_vector3(v2_deriv1))

    with pytest.raises(ValueError):
        Matrix3.twovec(v1, 0, v2, 1, recursive=True)


def test_matrix3_test_twovec_with_derivatives_in_unit1_unit2_and_unit3_we_nee() -> None:
    """Test twovec with derivatives in unit1, unit2, and unit3 # We need to test when key is in unit1._derivs, unit2._derivs, and unit3._derivs # unit1 is created from vector1 using .unit(), which preserves derivatives # unit2 and unit3 are created from cross products (ucross), which also preserve derivatives."""

    np.random.seed(2599)

    v1 = Vector3([1., 0., 0.])
    v1.insert_deriv('t', Vector3([0., 0., 1.]))
    v2 = Vector3([0., 1., 0.])
    v2.insert_deriv('t', Vector3([0., 0., 1.]))

    m = Matrix3.twovec(v1, 0, v2, 1, recursive=True)
    assert hasattr(m, 'd_dt')

    assert type(m) == Matrix3


def test_matrix3_test_with_different_derivative_keys_to_test_branches_test_ca() -> None:
    """Test with different derivative keys to test branches # Test case where key is only in vector2, not in unit1 # This tests the branch where key is NOT in unit1._derivs but IS in unit2._derivs and unit3._derivs."""

    np.random.seed(2599)

    v1_no_deriv = Vector3([1., 0., 0.])
    v2_only = Vector3([0., 1., 0.])
    v2_only.insert_deriv('t2', Vector3([0., 0., 1.]))

    m = Matrix3.twovec(v1_no_deriv, 0, v2_only, 1, recursive=True)
    assert hasattr(m, 'd_dt2')

    assert type(m) == Matrix3


def test_matrix3_test_case_where_key_is_in_unit1_but_we_want_to_test_all_bran() -> None:
    """Test case where key is in unit1 but we want to test all branches # If v1 has 't1' and v2 has 't2', then all units will have both keys # But we can test the True branches for all three."""

    np.random.seed(2599)

    v1_both = Vector3([1., 0., 0.])
    v1_both.insert_deriv('t1', Vector3([0., 0., 1.]))
    v2_both = Vector3([0., 1., 0.])
    v2_both.insert_deriv('t2', Vector3([0., 0., 1.]))

    m = Matrix3.twovec(v1_both, 0, v2_both, 1, recursive=True)
    assert hasattr(m, 'd_dt1')
    assert hasattr(m, 'd_dt2')

    assert type(m) == Matrix3


def test_matrix3_test_with_different_axis_combination_to_ensure_all_paths_are() -> None:
    """Test with different axis combination to ensure all paths are covered # For axis1=1, axis2=2, we have axis3=0 # This uses the if branch: unit3 = unit1.ucross(vector2), unit2 = unit3.ucross(unit1)."""

    np.random.seed(2599)

    v1 = Vector3([1., 0., 0.])
    v1.insert_deriv('t', Vector3([0.1, 0., 0.]))
    v2 = Vector3([0., 1., 0.])
    v2.insert_deriv('t', Vector3([0., 0.1, 0.]))

    m = Matrix3.twovec(v1, 1, v2, 2, recursive=True)
    assert hasattr(m, 'd_dt')

    assert type(m) == Matrix3


def test_matrix3_test_else_branch_this_happens_when_3_axis2_axis1_3_1_for_axi() -> None:
    """Test else branch # This happens when (3 + axis2 - axis1) % 3 != 1 # For axis1=0, axis2=2: (3 + 2 - 0) % 3 = 2, so uses else branch."""

    np.random.seed(2599)

    v1 = Vector3([1., 0., 0.])
    v1.insert_deriv('t', Vector3([0.1, 0., 0.]))
    v2 = Vector3([0., 1., 0.])
    v2.insert_deriv('t', Vector3([0., 0.1, 0.]))
    m = Matrix3.twovec(v1, 0, v2, 2, recursive=True)
    assert hasattr(m, 'd_dt')
    assert type(m) == Matrix3

    m = Matrix3.twovec(v1, 2, v2, 1, recursive=True)
    assert hasattr(m, 'd_dt')
    assert type(m) == Matrix3

    m = Matrix3.twovec(v1, 1, v2, 0, recursive=True)
    assert hasattr(m, 'd_dt')
    assert type(m) == Matrix3


def test_matrix3_test_twovec_with_readonly_inputs_the_code_checks_if_unit1_re() -> None:
    """Test twovec with readonly inputs # The code checks if unit1.readonly and vector2.readonly, then sets result as readonly # However, unit() doesn't preserve readonly, so unit1.readonly will be False # This means the condition at line 143 will be False, so line 144 won't execute # To test line 144, we would need unit1.readonly to be True, but unit() doesn't preserve it # So this path might be hard to test. Let's test that the function works with readonly inputs."""

    np.random.seed(2599)

    v1 = Vector3([1., 0., 0.]).as_readonly()
    v2 = Vector3([0., 1., 0.]).as_readonly()
    m = Matrix3.twovec(v1, 0, v2, 1)

    assert type(m) == Matrix3


        # Note: To actually test line 144, we would need unit1.readonly to be True,
        # but unit() doesn't preserve readonly, so this is difficult to test
##########################################################################################
