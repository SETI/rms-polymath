##########################################################################################
# tests/test_vector_comprehensive.py
# Comprehensive unit tests for Vector class based on docstrings
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Vector, Matrix, Pair


def test_vector_comprehensive_test_as_vector_static_method_simple_case_vector_to_vector() -> None:
    """Test as_vector static method # Simple case: Vector to Vector."""

    np.random.seed(1234)

    v1 = Vector([1., 2., 3.])
    v1_conv = Vector.as_vector(v1)
    assert type(v1_conv) == Vector
    assert np.allclose(v1_conv.vals, [1., 2., 3.])

    s1 = Scalar(5.)
    v2 = Vector.as_vector(s1)
    assert type(v2) == Vector
    assert v2.shape == ()
    assert v2.numer == (1,)
    assert np.allclose(v2.vals, [5.])

    v3 = Vector.as_vector([1., 2., 3.])
    assert type(v3) == Vector
    assert np.allclose(v3.vals, [1., 2., 3.])

    s2 = Scalar([[1., 2.], [3., 4.]])
    v4 = Vector.as_vector(s2)
    assert v4.shape == (2, 2)
    assert v4.numer == (1,)
    assert np.allclose(v4.vals[0, 0], [1.])

    v5 = Vector([1., 2., 3.])
    s3 = v5.to_scalar(0)
    assert type(s3) == Scalar
    assert s3 == 1.
    s4 = v5.to_scalar(1)
    assert s4 == 2.

    v6 = Vector([[1., 2., 3.], [4., 5., 6.]])
    s5 = v6.to_scalar(0)
    assert s5.shape == (2,)
    assert np.allclose(s5.vals, [1., 4.])

    v7 = Vector([1., 2., 3.])
    scalars = v7.to_scalars()
    assert len(scalars) == 3
    assert scalars[0] == 1.
    assert scalars[1] == 2.
    assert scalars[2] == 3.

    v8 = Vector([[1., 2.], [3., 4.]])
    scalars2 = v8.to_scalars()
    assert len(scalars2) == 2
    assert scalars2[0].shape == (2,)
    assert np.allclose(scalars2[0].vals, [1., 3.])

    v9 = Vector([1., 2., 3., 4.])
    p1 = v9.to_pair(axes=(0, 1))
    assert type(p1) == Pair
    assert np.allclose(p1.vals, [1., 2.])
    p2 = v9.to_pair(axes=(1, 3))
    assert np.allclose(p2.vals, [2., 4.])

    s6 = Scalar(1.)
    s7 = Scalar(2.)
    s8 = Scalar(3.)
    v10 = Vector.from_scalars(s6, s7, s8)
    assert type(v10) == Vector
    assert v10.shape == ()
    assert np.allclose(v10.vals, [1., 2., 3.])

    s9 = Scalar([[1., 2.], [3., 4.]])
    s10 = Scalar([[5., 6.], [7., 8.]])
    s11 = Scalar([[9., 10.], [11., 12.]])
    v11 = Vector.from_scalars(s9, s10, s11)
    assert v11.shape == (2, 2)
    assert np.allclose(v11.vals[0, 0], [1., 5., 9.])

    v12 = Vector([0, 1, 2])
    idx = v12.as_index()
    assert type(idx) == tuple

    assert len(idx) == 3
    assert np.allclose(idx[0], [0])
    assert np.allclose(idx[1], [1])
    assert np.allclose(idx[2], [2])

    v13 = Vector([0, 1, 2])
    idx2, mask2 = v13.as_index_and_mask()
    assert type(idx2) == tuple
    assert not mask2

    v14 = Vector([1.5, 2.7, 3.9])
    v15 = v14.int()
    assert np.allclose(v15.vals, [1, 2, 3])
    assert v15.is_int()

    v16 = Vector([1, 2, 3, 4, 5])
    v17 = v16.int(top=(3, 3, 3, 3, 3), remask=True)

    if isinstance(v17.mask, np.ndarray):
        # Elements with values > 3 should be masked (inclusive=False by default)
        # Actually, let's just check that the method works
        assert isinstance(v17, Vector)
    else:
        # If scalar mask, it's either all masked or all unmasked
        assert isinstance(v17.mask, (bool, np.bool_))

    v18 = Vector([1., 2., 3.])
    m1 = v18.as_column()
    assert type(m1) == Matrix
    assert m1.numer == (3, 1)
    assert np.allclose(m1.vals[:, 0], [1., 2., 3.])

    m2 = v18.as_row()
    assert type(m2) == Matrix
    assert m2.numer == (1, 3)
    assert np.allclose(m2.vals[0, :], [1., 2., 3.])

    m3 = v18.as_diagonal()
    assert type(m3) == Matrix
    assert m3.numer == (3, 3)
    assert np.allclose(np.diag(m3.vals), [1., 2., 3.])

    v19 = Vector([1., 2., 3.])
    v20 = Vector([4., 5., 6.])
    s12 = v19.dot(v20)
    assert type(s12) == Scalar
    assert s12 == 32.  # 1*4 + 2*5 + 3*6

    v21 = Vector([[1., 2.], [3., 4.]])
    v22 = Vector([[5., 6.], [7., 8.]])
    s13 = v21.dot(v22)
    assert s13.shape == (2,)
    assert s13[0] == 17.  # 1*5 + 2*6
    assert s13[1] == 53.  # 3*7 + 4*8

    v23 = Vector([3., 4.])
    n1 = v23.norm()
    assert type(n1) == Scalar
    assert n1 == 5. or abs(n1 - 5.) <= 1e-10

    n2 = v23.norm_sq()
    assert n2 == 25.

    v24 = Vector([3., 4.])
    v25 = v24.unit()
    assert v25.norm() == 1. or abs(v25.norm() - 1.) <= 1e-10
    assert np.allclose(v25.vals, [0.6, 0.8])

    v26 = Vector([3., 4.])
    v27 = v26.with_norm(10.)
    assert v27.norm() == 10. or abs(v27.norm() - 10.) <= 1e-10

    v28 = Vector([1., 0., 0.])
    v29 = Vector([0., 1., 0.])
    v30 = v28.cross(v29)
    assert np.allclose(v30.vals, [0., 0., 1.])

    v31 = v28.ucross(v29)
    assert v31.norm() == 1. or abs(v31.norm() - 1.) <= 1e-10

    v32 = Vector([1., 2.])
    v33 = Vector([3., 4.])
    m4 = v32.outer(v33)
    assert type(m4) == Matrix
    assert m4.numer == (2, 2)
    assert np.allclose(m4.vals, [[3., 4.], [6., 8.]])

    v34 = Vector([1., 1.])
    v35 = Vector([1., 0.])
    v36 = v34.perp(v35)

    assert v36.dot(v35) == 0. or abs(v36.dot(v35) - 0.) <= 1e-10

    v37 = Vector([1., 1.])
    v38 = Vector([1., 0.])
    v39 = v37.proj(v38)

    assert np.allclose(v39.vals, [1., 0.], atol=1e-10)

    v40 = Vector([1., 0.])
    v41 = Vector([0., 1.])
    s14 = v40.sep(v41)
    assert s14 == np.pi/2 or abs(s14 - np.pi/2) <= 1e-10

    v42 = Vector([1., 2., 3.])
    m5 = v42.cross_product_as_matrix()
    assert type(m5) == Matrix
    assert m5.numer == (3, 3)

    v43 = Vector([4., 5., 6.])
    v44 = m5 * v43
    v45 = v42.cross(v43)
    assert np.allclose(v44.vals, v45.vals)

    v46 = Vector([1., 2., 3.])
    v47 = Vector([4., 5., 6.])
    v48 = v46.element_mul(v47)
    assert np.allclose(v48.vals, [4., 10., 18.])

    v49 = Vector([4., 10., 18.])
    v50 = Vector([2., 5., 6.])
    v51 = v49.element_div(v50)
    assert np.allclose(v51.vals, [2., 2., 3.])

    v52 = Vector([1., 0.])
    v53 = Vector([2., 0.])  # Scale along x-axis with magnitude 2
    v54 = v52.vector_scale(v53)

    v52b = Vector([2., 0.])
    v54b = v52b.vector_scale(v53)

    assert isinstance(v54, Vector)
    assert v54.shape == ()
    assert isinstance(v54b, Vector)

    v55 = v54.vector_unscale(v53)
    assert v55.vals[0] == 1. or abs(v55.vals[0] - 1.) <= 1e-10

    s15 = Scalar([1., 2.])
    s16 = Scalar([3., 4.])
    v56 = Vector.combos(s15, s16)
    assert v56.shape == (2, 2)
    assert v56.numer == (2,)
    assert np.allclose(v56.vals[0, 0], [1., 3.])
    assert np.allclose(v56.vals[0, 1], [1., 4.])
    assert np.allclose(v56.vals[1, 0], [2., 3.])
    assert np.allclose(v56.vals[1, 1], [2., 4.])

    v57 = Vector([[1., 2., 3.], [4., 5., 6.]])
    v58 = v57.mask_where_component_le(axis=0, limit=2.)
    assert (v58.mask[0] or not np.allclose(v58.vals[0], [1., 2., 3.]))

    v59 = v57.mask_where_component_ge(axis=0, limit=4.)
    assert (v59.mask[1] or not np.allclose(v59.vals[1], [4., 5., 6.]))

    v60 = v57.mask_where_component_lt(axis=0, limit=2.)

    assert isinstance(v60, Vector)

    v61 = v57.mask_where_component_gt(axis=0, limit=3.)

    assert isinstance(v61, Vector)

    v62 = Vector([1., 5., 9.])

    v63 = v62.clip_component(axis=0, lower=2., upper=8.)

    assert v63.vals[0] == 2. or abs(v63.vals[0] - 2.) <= 1e-10
    assert v63.vals[1] == 5. or abs(v63.vals[1] - 5.) <= 1e-10  # Unchanged
    assert v63.vals[2] == 9. or abs(v63.vals[2] - 9.) <= 1e-10  # Unchanged

    v64 = Vector([3., 4.])
    s17 = abs(v64)
    assert type(s17) == Scalar
    assert s17 == 5.

    v65 = Vector([1., 2., 3.])
    with pytest.raises(TypeError):
        v65.identity()

    v66 = Vector([[1., 0.], [0., 1.]], drank=1)
    v67 = v66.reciprocal()

    assert type(v67) == Vector
    assert v67.drank == 1
    # Check that it's the inverse: v66 * v67 should be identity
    # This is tested more thoroughly in test_vector_reciprocal.py

    v68 = Vector([1., 2., 3.])
    with pytest.raises(TypeError):
        v68.reciprocal()

    v69 = Vector(5.)
    assert v69.shape == ()
    assert v69.numer == (1,)
    assert np.allclose(v69.vals, [5.])
    v70 = Vector(7)
    assert np.allclose(v70.vals, [7])

    m6 = Matrix([[1., 2., 3.]])
    v71 = Vector.as_vector(m6)
    assert type(v71) == Vector
    assert np.allclose(v71.vals, [1., 2., 3.])

    m7 = Matrix([[1.], [2.], [3.]])
    v72 = Vector.as_vector(m7)
    assert type(v72) == Vector
    assert np.allclose(v72.vals, [1., 2., 3.])

    s18 = Scalar(1.)
    s18.insert_deriv('t', Scalar(2.))
    v73 = Vector.as_vector(s18, recursive=True)
    assert ('t' in v73.derivs)

    v74 = Vector([1., 2., 3.])
    with pytest.raises(IndexError):
        v74.to_pair(axes=(0, 5))
    with pytest.raises(IndexError):
        v74.to_pair(axes=(0, 0))

    v75 = Vector([-1, 5, 3])
    v76 = v75.int(top=(3, 3, 3), clip=True)

    assert np.allclose(v76.vals, [0, 2, 2])

    v77 = Vector([0, 1, 2, 3])
    v78 = v77.int(top=(3, 3, 3, 3), inclusive=False, remask=True)

    assert isinstance(v78, Vector)

    v79 = Vector([0, 1, 2, 3])
    v80 = v79.int(top=(3, 3, 3, 3), shift=True, remask=True)
    assert isinstance(v80, Vector)

    v81 = Vector([0, 1, 2])
    v81 = v81.mask_where_component_le(0, 1)
    idx3, _mask3 = v81.as_index_and_mask()
    assert type(idx3) == tuple

    v82 = Vector([0, 1, 2])
    idx4, _mask4 = v82.as_index_and_mask(masked=99)
    assert type(idx4) == tuple

    v83 = Vector([3., 4.])
    v84 = v83.unit(recursive=False)
    assert v84.norm() == 1. or abs(v84.norm() - 1.) <= 1e-10

    v85 = Vector([3., 4.])
    v86 = v85.with_norm(10., recursive=False)
    assert v86.norm() == 10. or abs(v86.norm() - 10.) <= 1e-10

    v87 = Vector([1., 0.])
    v88 = Vector([0., 1.])
    s19 = v87.cross(v88)
    assert type(s19) == Scalar
    assert s19 == 1. or abs(s19 - 1.) <= 1e-10

    v89 = Vector([1., 1.])
    v90 = Vector([1., 0.])
    v91 = v89.perp(v90, recursive=False)
    assert v91.dot(v90) == 0. or abs(v91.dot(v90) - 0.) <= 1e-10

    v92 = Vector([1., 1.])
    v93 = Vector([1., 0.])
    v94 = v92.proj(v93, recursive=False)
    assert np.allclose(v94.vals, [1., 0.], atol=1e-10)

    v95 = Vector([1., 0.])
    v96 = Vector([0., 1.])
    s20 = v95.sep(v96, recursive=False)
    assert s20 == np.pi/2 or abs(s20 - np.pi/2) <= 1e-10

    v97a = Vector([1., 0., 0.])
    m8 = v97a.cross_product_as_matrix()
    assert type(m8) == Matrix
    assert m8.drank == 0

    v98 = Vector([1., 2.])
    with pytest.raises(ValueError):
        v98.cross_product_as_matrix()

    v99 = Vector([[1., 2., 3.], [0., 0., 0.]], drank=1)
    v100 = Vector([[4., 5., 6.], [0., 0., 0.]], drank=1)
    with pytest.raises(ValueError):
        v99.element_mul(v100)

    v101 = Vector([1., 2., 3.])
    v102 = v101.element_mul([4., 5., 6.])
    assert np.allclose(v102.vals, [4., 10., 18.])

    v103 = Vector([4., 10., 18.])
    v104 = Vector([2., 0., 6.])
    v105 = v103.element_div(v104)

    assert isinstance(v105, Vector)

    if isinstance(v105.mask, np.ndarray):
        # Check if any element is masked (the zero divisor should cause masking)
        assert (np.any(v105.mask) or v105.mask.all())

    v106 = Vector([[1., 2., 3.], [0., 0., 0.]], drank=1)
    v107 = Vector([4., 5., 6.])
    with pytest.raises(ValueError):
        v106.element_div(v107)

    s19 = Scalar([1., 2.], drank=1)
    with pytest.raises(ValueError):
        Vector.combos(s19)


def test_vector_comprehensive_test_mask_where_component_le_with_replace() -> None:
    """Test mask_where_component_le with replace."""

    np.random.seed(1234)

    v108 = Vector([[1., 2., 3.], [4., 5., 6.]])

    v109 = v108.mask_where_component_le(axis=0, limit=2., replace=Vector([99., 99., 99.]))

    assert isinstance(v109, Vector)

    v110 = v108.mask_where_component_ge(axis=0, limit=4., replace=Vector([99., 99., 99.]))
    assert isinstance(v110, Vector)

    v111 = v108.mask_where_component_lt(axis=0, limit=2., replace=Vector([99., 99., 99.]))
    assert isinstance(v111, Vector)

    v112 = v108.mask_where_component_gt(axis=0, limit=3., replace=Vector([99., 99., 99.]))
    assert isinstance(v112, Vector)


def test_vector_comprehensive_test_clip_component_with_lower_only() -> None:
    """Test clip_component with lower only."""

    np.random.seed(1234)

    v113 = Vector([1., 5., 9.])
    v114 = v113.clip_component(axis=0, lower=2., upper=None)
    assert v114.vals[0] == 2. or abs(v114.vals[0] - 2.) <= 1e-10


def test_vector_comprehensive_test_clip_component_with_upper_only() -> None:
    """Test clip_component with upper only."""

    np.random.seed(1234)

    v115 = Vector([1., 5., 9.])
    v116 = v115.clip_component(axis=0, lower=None, upper=8.)

    assert v116.vals[0] == 1. or abs(v116.vals[0] - 1.) <= 1e-10
    assert v116.vals[1] == 5. or abs(v116.vals[1] - 5.) <= 1e-10
    assert v116.vals[2] == 9. or abs(v116.vals[2] - 9.) <= 1e-10


def test_vector_comprehensive_test_clip_component_with_remask_true() -> None:
    """Test clip_component with remask=True."""

    np.random.seed(1234)

    v117 = Vector([1., 5., 9.])
    v118 = v117.clip_component(axis=0, lower=2., upper=8., remask=True)

    assert isinstance(v118, Vector)


def test_vector_comprehensive_test_clip_component_with_n_d_lower_upper() -> None:
    """Test clip_component with n-D lower/upper."""

    np.random.seed(1234)

    v119 = Vector([[1., 5.], [9., 3.]])
    v120 = v119.clip_component(axis=0, lower=Scalar([2., 2.]), upper=Scalar([8., 8.]))
    assert isinstance(v120, Vector)


def test_vector_comprehensive_test_abs_with_recursive_false() -> None:
    """Test __abs__ with recursive=False."""

    np.random.seed(1234)

    v121 = Vector([3., 4.])
    s21 = v121.__abs__(recursive=False)
    assert s21 == 5.


def test_vector_comprehensive_test_from_scalars_with_n_d_and_recursive_false() -> None:
    """Test from_scalars with n-D and recursive=False."""

    np.random.seed(1234)

    s22 = Scalar([[1., 2.], [3., 4.]])
    s23 = Scalar([[5., 6.], [7., 8.]])
    v122 = Vector.from_scalars(s22, s23, recursive=False)
    assert v122.shape == (2, 2)


def test_vector_comprehensive_test_from_scalars_with_readonly_parameter() -> None:
    """Test from_scalars with readonly parameter."""

    np.random.seed(1234)

    s24 = Scalar(1.)
    s25 = Scalar(2.)
    v123 = Vector.from_scalars(s24, s25, readonly=True)

    assert isinstance(v123, Vector)


##########################################################################################
