##########################################################################################
# tests/test_vector3_operations.py
# Vector3 instance methods: coordinate conversions, transformations, and vector operations
##########################################################################################

import numpy as np

from polymath import Scalar, Vector3, Matrix


def test_vector3_operations_test_to_ra_dec_length_method() -> None:
    """Test to_ra_dec_length method."""

    np.random.seed(2599)

    v24 = Vector3([1., 0., 0.])
    ra24, dec24, length24 = v24.to_ra_dec_length()
    assert type(ra24) == Scalar
    assert type(dec24) == Scalar
    assert type(length24) == Scalar
    assert np.allclose(ra24.vals, 0., atol=1e-10)
    assert np.allclose(dec24.vals, 0., atol=1e-10)
    assert np.allclose(length24.vals, 1., atol=1e-10)


def test_vector3_operations_test_to_cylindrical_method() -> None:
    """Test to_cylindrical method."""

    np.random.seed(2599)

    v29 = Vector3([1., 0., 0.])
    radius29, longitude29, z29 = v29.to_cylindrical()
    assert type(radius29) == Scalar
    assert type(longitude29) == Scalar
    assert type(z29) == Scalar
    assert np.allclose(radius29.vals, 1., atol=1e-10)
    assert np.allclose(longitude29.vals, 0., atol=1e-10)
    assert np.allclose(z29.vals, 0., atol=1e-10)


def test_vector3_operations_test_longitude_method() -> None:
    """Test longitude method."""

    np.random.seed(2599)

    v31 = Vector3([1., 0., 0.])
    lon31 = v31.longitude()
    assert type(lon31) == Scalar
    assert np.allclose(lon31.vals, 0., atol=1e-10)
    v32 = Vector3([0., 1., 0.])
    lon32 = v32.longitude()
    assert np.allclose(lon32.vals, np.pi/2, atol=1e-10)


def test_vector3_operations_test_latitude_method() -> None:
    """Test latitude method."""

    np.random.seed(2599)

    v34 = Vector3([1., 0., 0.])
    lat34 = v34.latitude()
    assert type(lat34) == Scalar
    assert np.allclose(lat34.vals, 0., atol=1e-10)
    v35 = Vector3([0., 0., 1.])
    lat35 = v35.latitude()
    assert np.allclose(lat35.vals, np.pi/2, atol=1e-10)


def test_vector3_operations_test_spin_method() -> None:
    """Test spin method."""

    np.random.seed(2599)

    v37 = Vector3([1., 0., 0.])
    pole = Vector3([0., 0., 1.])  # z-axis
    angle = Scalar(np.pi/2)
    v37_spun = v37.spin(pole, angle)
    assert type(v37_spun) == Vector3

    assert np.allclose(v37_spun.vals, [0., 1., 0.], atol=1e-10)


def test_vector3_operations_test_spin_with_angle_none_uses_pole_magnitude_via_arcsin() -> None:
    """Test spin with angle=None (uses pole magnitude via arcsin)."""

    np.random.seed(2599)

    v38 = Vector3([1., 0., 0.])

    pole38 = Vector3([0., 0., 1.])  # magnitude is 1.0, arcsin(1.0) = pi/2
    v38_spun = v38.spin(pole38)
    assert type(v38_spun) == Vector3

    assert np.allclose(v38_spun.vals, [0., 1., 0.], atol=1e-10)


def test_vector3_operations_test_offset_angles_method() -> None:
    """Test offset_angles method."""

    np.random.seed(2599)

    v40 = Vector3([1., 0., 0.])
    v41 = Vector3([0., 1., 0.])
    lon_off, lat_off = v40.offset_angles(v41)
    assert type(lon_off) == Scalar
    assert type(lat_off) == Scalar

    assert np.isfinite(lon_off.vals)
    assert np.isfinite(lat_off.vals)


def test_vector3_operations_test_inherited_methods_from_vector_to_scalar() -> None:
    """Test inherited methods from Vector - to_scalar."""

    np.random.seed(2599)

    v44 = Vector3(np.random.randn(4, 1, 5, 3))
    s44 = v44.to_scalar(0)
    assert type(s44) == Scalar
    assert s44.shape == v44.shape

    scalars44 = v44.to_scalars()
    assert len(scalars44) == 3
    assert type(scalars44[0]) == Scalar
    assert scalars44[0].shape == v44.shape


def test_vector3_operations_test_as_column() -> None:
    """Test as_column."""

    np.random.seed(2599)

    v45 = Vector3([1., 2., 3.])
    m45 = v45.as_column()
    assert type(m45) == Matrix
    assert m45.numer == (3, 1)
    assert np.allclose(m45.vals[..., 0], [1., 2., 3.])


def test_vector3_operations_test_as_row() -> None:
    """Test as_row."""

    np.random.seed(2599)

    v46 = Vector3([1., 2., 3.])
    m46 = v46.as_row()
    assert type(m46) == Matrix
    assert m46.numer == (1, 3)
    assert np.allclose(m46.vals[0, :], [1., 2., 3.])


def test_vector3_operations_test_as_diagonal() -> None:
    """Test as_diagonal."""

    np.random.seed(2599)

    v47 = Vector3([1., 2., 3.])
    m47 = v47.as_diagonal()
    assert type(m47) == Matrix
    assert m47.numer == (3, 3)
    assert np.allclose(m47.vals[0, 0], 1.)
    assert np.allclose(m47.vals[1, 1], 2.)
    assert np.allclose(m47.vals[2, 2], 3.)


def test_vector3_operations_test_dot() -> None:
    """Test dot."""

    np.random.seed(2599)

    v48 = Vector3([1., 2., 3.])
    v49 = Vector3([4., 5., 6.])
    dot48 = v48.dot(v49)
    assert type(dot48) == Scalar

    assert np.allclose(dot48.vals, 32.)


def test_vector3_operations_test_norm() -> None:
    """Test norm."""

    np.random.seed(2599)

    v52 = Vector3([3., 4., 0.])
    norm52 = v52.norm()
    assert type(norm52) == Scalar

    assert np.allclose(norm52.vals, 5.)


def test_vector3_operations_test_unit() -> None:
    """Test unit."""

    np.random.seed(2599)

    v54 = Vector3([3., 4., 0.])
    unit54 = v54.unit()
    assert type(unit54) == Vector3

    assert np.allclose(unit54.vals, [0.6, 0.8, 0.], atol=1e-10)
    assert np.allclose(unit54.norm().vals, 1., atol=1e-10)


def test_vector3_operations_test_cross() -> None:
    """Test cross."""

    np.random.seed(2599)

    v56 = Vector3([1., 0., 0.])
    v57 = Vector3([0., 1., 0.])
    cross56 = v56.cross(v57)
    assert type(cross56) == Vector3

    assert np.allclose(cross56.vals, [0., 0., 1.], atol=1e-10)


def test_vector3_operations_test_ucross() -> None:
    """Test ucross."""

    np.random.seed(2599)

    v60 = Vector3([1., 0., 0.])
    v61 = Vector3([0., 1., 0.])
    ucross60 = v60.ucross(v61)
    assert type(ucross60) == Vector3

    assert np.allclose(ucross60.vals, [0., 0., 1.], atol=1e-10)
    assert np.allclose(ucross60.norm().vals, 1., atol=1e-10)


def test_vector3_operations_test_outer() -> None:
    """Test outer."""

    np.random.seed(2599)

    v62 = Vector3([1., 2., 3.])
    v63 = Vector3([4., 5., 6.])
    outer62 = v62.outer(v63)
    assert type(outer62) == Matrix

    assert outer62.numer == (3, 3)


def test_vector3_operations_test_perp() -> None:
    """Test perp."""

    np.random.seed(2599)

    v64 = Vector3([1., 1., 0.])
    v65 = Vector3([1., 0., 0.])
    perp64 = v64.perp(v65)
    assert type(perp64) == Vector3

    assert np.allclose(perp64.vals, [0., 1., 0.], atol=1e-10)


def test_vector3_operations_test_proj() -> None:
    """Test proj."""

    np.random.seed(2599)

    v66 = Vector3([1., 1., 0.])
    v67 = Vector3([1., 0., 0.])
    proj66 = v66.proj(v67)
    assert type(proj66) == Vector3

    assert np.allclose(proj66.vals, [1., 0., 0.], atol=1e-10)


def test_vector3_operations_test_sep() -> None:
    """Test sep."""

    np.random.seed(2599)

    v68 = Vector3([1., 0., 0.])
    v69 = Vector3([0., 1., 0.])
    sep68 = v68.sep(v69)
    assert type(sep68) == Scalar

    assert np.allclose(sep68.vals, np.pi/2, atol=1e-10)


def test_vector3_operations_test_cross_product_as_matrix() -> None:
    """Test cross_product_as_matrix."""

    np.random.seed(2599)

    v72 = Vector3([1., 2., 3.])
    m72 = v72.cross_product_as_matrix()
    assert type(m72) == Matrix
    assert m72.numer == (3, 3)

    v73 = Vector3([4., 5., 6.])
    cross72 = v72.cross(v73)
    m72_v73 = m72 * v73
    assert np.allclose(m72_v73.vals, cross72.vals, atol=1e-10)


def test_vector3_operations_test_element_mul() -> None:
    """Test element_mul."""

    np.random.seed(2599)

    v75 = Vector3([1., 2., 3.])
    v76 = Vector3([4., 5., 6.])
    elem_mul75 = v75.element_mul(v76)
    assert type(elem_mul75) == Vector3

    assert np.allclose(elem_mul75.vals, [4., 10., 18.])


def test_vector3_operations_test_element_div() -> None:
    """Test element_div."""

    np.random.seed(2599)

    v79 = Vector3([4., 10., 18.])
    v80 = Vector3([4., 5., 6.])
    elem_div79 = v79.element_div(v80)
    assert type(elem_div79) == Vector3

    assert np.allclose(elem_div79.vals, [1., 2., 3.], atol=1e-10)


def test_vector3_operations_test_abs_norm() -> None:
    """Test __abs__ (norm)."""

    np.random.seed(2599)

    v83 = Vector3([3., 4., 0.])
    abs83 = abs(v83)
    assert type(abs83) == Scalar
    assert np.allclose(abs83.vals, 5.)


##########################################################################################
