##########################################################################################
# tests/test_vector3_advanced.py
# Vector3 advanced tests: n-D arrays, round-trips, type preservation
##########################################################################################

import numpy as np

from polymath import Scalar, Vector3


def test_vector3_advanced_test_n_d_arrays() -> None:
    """Test n-D arrays."""

    np.random.seed(2599)

    v5 = Vector3(np.random.randn(2, 3, 3))
    assert v5.shape == (2, 3)
    assert v5.item == (3,)
    assert v5.vals.shape == (2, 3, 3)


def test_vector3_advanced_test_higher_dimensional_arrays() -> None:
    """Test higher-dimensional arrays."""

    np.random.seed(2599)

    v6 = Vector3(np.random.randn(4, 5, 6, 3))
    assert v6.shape == (4, 5, 6)
    assert v6.item == (3,)
    assert v6.vals.shape == (4, 5, 6, 3)


def test_vector3_advanced_test_from_ra_dec_length_with_n_d_inputs() -> None:
    """Test from_ra_dec_length with n-D inputs."""

    np.random.seed(2599)

    ra_2d = Scalar([[0., np.pi/2], [np.pi, 3*np.pi/2]])
    dec_2d = Scalar([[0., 0.], [0., 0.]])
    v23 = Vector3.from_ra_dec_length(ra_2d, dec_2d, 2.)
    assert v23.shape == (2, 2)

    assert np.allclose(v23.vals[0, 0], [2., 0., 0.], atol=1e-10)


def test_vector3_advanced_test_to_ra_dec_length_with_n_d() -> None:
    """Test to_ra_dec_length with n-D."""

    np.random.seed(2599)

    v25 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
    ra25, dec25, length25 = v25.to_ra_dec_length()
    assert ra25.shape == (2, 2)
    assert dec25.shape == (2, 2)
    assert length25.shape == (2, 2)


def test_vector3_advanced_test_from_cylindrical_with_n_d_inputs() -> None:
    """Test from_cylindrical with n-D inputs."""

    np.random.seed(2599)

    radius_2d = Scalar([[1., 2.], [3., 4.]])
    longitude_2d = Scalar([[0., np.pi/2], [np.pi, 3*np.pi/2]])
    v28 = Vector3.from_cylindrical(radius_2d, longitude_2d, 0.)
    assert v28.shape == (2, 2)


def test_vector3_advanced_test_to_cylindrical_with_n_d() -> None:
    """Test to_cylindrical with n-D."""

    np.random.seed(2599)

    v30 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
    radius30, longitude30, z30 = v30.to_cylindrical()
    assert radius30.shape == (2, 2)
    assert longitude30.shape == (2, 2)
    assert z30.shape == (2, 2)


def test_vector3_advanced_test_longitude_with_n_d() -> None:
    """Test longitude with n-D."""

    np.random.seed(2599)

    v33 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[-1., 0., 0.], [0., -1., 0.]]]))
    lon33 = v33.longitude()
    assert lon33.shape == (2, 2)


def test_vector3_advanced_test_latitude_with_n_d() -> None:
    """Test latitude with n-D."""

    np.random.seed(2599)

    v36 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
    lat36 = v36.latitude()
    assert lat36.shape == (2, 2)


def test_vector3_advanced_test_spin_with_n_d() -> None:
    """Test spin with n-D."""

    np.random.seed(2599)

    v39 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
    pole39 = Vector3([0., 0., 1.])
    angle39 = Scalar(np.pi/2)
    v39_spun = v39.spin(pole39, angle39)
    assert v39_spun.shape == (2, 2)


def test_vector3_advanced_test_offset_angles_with_n_d() -> None:
    """Test offset_angles with n-D."""

    np.random.seed(2599)

    v42 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
    v43 = Vector3([1., 0., 0.])
    lon_off2, lat_off2 = v42.offset_angles(v43)
    assert lon_off2.shape == (2, 2)
    assert lat_off2.shape == (2, 2)


def test_vector3_advanced_test_dot_with_n_d() -> None:
    """Test dot with n-D."""

    np.random.seed(2599)

    v50 = Vector3(np.random.randn(4, 1, 5, 3))
    v51 = Vector3(np.random.randn(8, 5, 3))
    dot50 = v50.dot(v51)

    assert dot50.shape == (4, 8, 5)


def test_vector3_advanced_test_norm_with_n_d() -> None:
    """Test norm with n-D."""

    np.random.seed(2599)

    v53 = Vector3(np.random.randn(2, 3, 3))
    norm53 = v53.norm()
    assert norm53.shape == (2, 3)


def test_vector3_advanced_test_unit_with_n_d() -> None:
    """Test unit with n-D."""

    np.random.seed(2599)

    v55 = Vector3(np.random.randn(2, 3, 3))
    unit55 = v55.unit()
    assert unit55.shape == (2, 3)


def test_vector3_advanced_test_cross_with_n_d() -> None:
    """Test cross with n-D."""

    np.random.seed(2599)

    v58 = Vector3(np.random.randn(4, 1, 5, 3))
    v59 = Vector3(np.random.randn(8, 5, 3))
    cross58 = v58.cross(v59)

    assert cross58.shape == (4, 8, 5)


def test_vector3_advanced_test_cross_product_as_matrix_with_n_d() -> None:
    """Test cross_product_as_matrix with n-D."""

    np.random.seed(2599)

    v74 = Vector3(np.random.randn(2, 3, 3))
    m74 = v74.cross_product_as_matrix()
    assert m74.shape == (2, 3)
    assert m74.numer == (3, 3)


def test_vector3_advanced_test_element_mul_with_n_d() -> None:
    """Test element_mul with n-D."""

    np.random.seed(2599)

    v77 = Vector3(np.random.randn(2, 3, 3))
    v78 = Vector3(np.random.randn(2, 3, 3))
    elem_mul77 = v77.element_mul(v78)
    assert elem_mul77.shape == (2, 3)


def test_vector3_advanced_test_element_div_with_n_d() -> None:
    """Test element_div with n-D."""

    np.random.seed(2599)

    v81 = Vector3(np.random.randn(2, 3, 3))
    v82 = Vector3(np.random.randn(2, 3, 3))
    elem_div81 = v81.element_div(v82)
    assert elem_div81.shape == (2, 3)


def test_vector3_advanced_test_sep_with_n_d() -> None:
    """Test sep with n-D."""

    np.random.seed(2599)

    v70 = Vector3(np.random.randn(2, 3, 3))
    v71 = Vector3(np.random.randn(2, 3, 3))
    sep70 = v70.sep(v71)
    assert sep70.shape == (2, 3)


def test_vector3_advanced_test_complex_n_d_case() -> None:
    """Test complex n-D case."""

    np.random.seed(2599)

    v87 = Vector3(np.random.randn(3, 4, 5, 6, 3))
    assert v87.shape == (3, 4, 5, 6)
    assert v87.item == (3,)
    assert v87.vals.shape == (3, 4, 5, 6, 3)


def test_vector3_advanced_test_that_operations_preserve_type() -> None:
    """Test that operations preserve type."""

    np.random.seed(2599)

    v88 = Vector3([1., 2., 3.])
    v89 = Vector3([4., 5., 6.])
    v_result = v88 + v89
    assert type(v_result) == Vector3
    v_result2 = v88 * 2.
    assert type(v_result2) == Vector3


def test_vector3_advanced_test_round_trip_conversions() -> None:
    """Test round-trip conversions."""

    np.random.seed(2599)

    v90 = Vector3([1., 2., 3.])
    ra90, dec90, length90 = v90.to_ra_dec_length()
    v90_recon = Vector3.from_ra_dec_length(ra90, dec90, length90)
    assert np.allclose(v90.vals, v90_recon.vals, atol=1e-10)
    v91 = Vector3([1., 2., 3.])
    radius91, longitude91, z91 = v91.to_cylindrical()
    v91_recon = Vector3.from_cylindrical(radius91, longitude91, z91)
    assert np.allclose(v91.vals, v91_recon.vals, atol=1e-10)


def test_vector3_advanced_test_n_d_round_trip() -> None:
    """Test n-D round-trip."""

    np.random.seed(2599)

    v92 = Vector3(np.random.randn(2, 3, 3))
    ra92, dec92, length92 = v92.to_ra_dec_length()
    v92_recon = Vector3.from_ra_dec_length(ra92, dec92, length92)
    assert v92_recon.shape == (2, 3)
    assert np.allclose(v92.vals, v92_recon.vals, atol=1e-10)


##########################################################################################
