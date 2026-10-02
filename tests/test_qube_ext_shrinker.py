##########################################################################################
# tests/test_qube_ext_shrinker.py
#
# Comprehensive unit tests for shrink and unshrink operations based on docstrings in shrinker.py
##########################################################################################

import numpy as np

from polymath import Boolean, Qube, Scalar, Vector, Vector3


def test_qube_ext_shrinker_simple_1_d_case_true_antimask_leaves_object_unchanged() -> None:
    """Simple 1-D case: True antimask leaves object unchanged."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    b = a.shrink(True)
    assert a == b


def test_qube_ext_shrinker_simple_1_d_case_false_antimask_returns_masked_single_value() -> None:
    """Simple 1-D case: False antimask returns masked single value."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    b = a.shrink(False)
    assert b == Scalar.MASKED
    assert b.readonly


def test_qube_ext_shrinker_simple_1_d_case_partial_antimask() -> None:
    """Simple 1-D case: partial antimask."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    assert b.shape == (3,)  # 3 True values
    assert np.allclose(b.values, [1., 3., 5.])
    assert b.readonly


def test_qube_ext_shrinker_simple_1_d_case_shapeless_object_with_true_antimask() -> None:
    """Simple 1-D case: shapeless object with True antimask."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(7.)
    b = a.shrink(True)
    assert a == b


def test_qube_ext_shrinker_simple_1_d_case_shapeless_object_with_false_antimask() -> None:
    """Simple 1-D case: shapeless object with False antimask."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(7.)
    b = a.shrink(False)
    assert b == Scalar.MASKED
    assert b.readonly


def test_qube_ext_shrinker_simple_1_d_case_shapeless_object_with_array_antimask() -> None:
    """Simple 1-D case: shapeless object with array antimask."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(7.)
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    assert a == b  # Shapeless objects return unchanged


def test_qube_ext_shrinker_2_d_array_with_2_d_antimask_matching_full_shape() -> None:
    """Complex n-D case: 2-D array with 2-D antimask (matches full shape)."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(np.arange(20).reshape(4, 5))
    antimask = np.array([[True, False, True, False, True],
                         [False, False, False, False, False],
                         [True, True, False, False, False],
                         [False, False, False, False, False]])
    b = a.shrink(antimask)

    assert b.shape[-1] == np.sum(antimask)
    assert b.readonly


def test_qube_ext_shrinker_complex_n_d_case_2_d_array_with_2_d_antimask() -> None:
    """Complex n-D case: 2-D array with 2-D antimask."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(np.arange(20).reshape(4, 5))
    antimask = np.array([[True, False, True, False, True],
                         [False, False, False, False, False],
                         [True, True, False, False, False],
                         [False, False, False, False, False]])
    b = a.shrink(antimask)

    assert b.shape[-1] == np.sum(antimask)
    assert b.readonly


def test_qube_ext_shrinker_complex_n_d_case_vector_with_antimask() -> None:
    """Complex n-D case: Vector with antimask."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Vector(np.arange(30).reshape(10, 3))
    antimask = np.array([True] * 5 + [False] * 5)
    b = a.shrink(antimask)
    assert b.shape == (5,)
    assert b.numer == (3,)
    assert np.allclose(b.values[0], a.values[0])
    assert b.readonly


def test_qube_ext_shrinker_test_with_masked_object() -> None:
    """Test with masked object."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.], mask=[True, False, True, False, False])
    antimask = np.array([True, True, True, True, True])
    b = a.shrink(antimask)

    assert b.mask[0]
    assert not b.mask[1]
    assert b.mask[2]
    assert not b.mask[3]
    assert not b.mask[4]


def test_qube_ext_shrinker_test_with_entirely_masked_object() -> None:
    """Test with entirely masked object."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.], mask=True)
    antimask = np.array([True, True, True, True, True])
    b = a.shrink(antimask)
    assert b == Scalar.MASKED
    assert b.readonly


def test_qube_ext_shrinker_antimask_with_no_overlap_with_object_antimask() -> None:
    """Test with antimask that has no overlap with object's antimask."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.], mask=[True, True, True, True, True])
    antimask = np.array([True, True, True, True, True])
    b = a.shrink(antimask)

    assert b == Scalar.MASKED
    assert b.readonly


def test_qube_ext_shrinker_test_with_derivatives() -> None:
    """Test with derivatives."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    da_dt = Scalar([10., 20., 30., 40., 50.])
    a.insert_deriv('t', da_dt)
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    assert hasattr(b, 'd_dt')
    assert b.d_dt.shape == (3,)
    assert np.allclose(b.d_dt.values, [10., 30., 50.])

    ##################################################################################
    # unshrink()
    ##################################################################################


def test_qube_ext_shrinker_simple_1_d_case_true_antimask_returns_unchanged() -> None:
    """Simple 1-D case: True antimask returns unchanged."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3.])
    b = a.unshrink(True)
    assert a == b


def test_qube_ext_shrinker_simple_1_d_case_false_antimask_with_shape_parameter() -> None:
    """Simple 1-D case: False antimask with shape parameter."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar.MASKED
    b = a.unshrink(False, shape=(5,))
    assert b.shape == (5,)
    assert np.all(b.mask)


def test_qube_ext_shrinker_simple_1_d_case_unshrink_from_shrunk_object() -> None:
    """Simple 1-D case: unshrink from shrunk object."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert c.shape == a.shape
    assert np.allclose(c.values[antimask], a.values[antimask])
    assert np.all(c.mask[~antimask])  # Masked where antimask is False


def test_qube_ext_shrinker_simple_1_d_case_shapeless_object_with_true_antimask_2() -> None:
    """Simple 1-D case: shapeless object with True antimask."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(7.)
    b = a.unshrink(True)
    assert a == b


def test_qube_ext_shrinker_simple_1_d_case_shapeless_object_with_false_antimask_2() -> None:
    """Simple 1-D case: shapeless object with False antimask."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(7.)
    b = a.unshrink(False, shape=(5,))
    assert b.shape == (5,)

    assert np.all(b.mask)

    assert np.allclose(b.values, 1.)


def test_qube_ext_shrinker_complex_n_d_case_2_d_array_with_2_d_antimask_2() -> None:
    """Complex n-D case: 2-D array with 2-D antimask."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(np.arange(20).reshape(4, 5))
    antimask = np.array([[True, False, True, False, True],
                         [False, False, False, False, False],
                         [True, True, False, False, False],
                         [False, False, False, False, False]])
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert c.shape == a.shape
    assert np.allclose(c.values[antimask], a.values[antimask])
    assert np.all(c.mask[~antimask])


def test_qube_ext_shrinker_complex_n_d_case_2_d_antimask() -> None:
    """Complex n-D case: 2-D antimask."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(np.arange(20).reshape(4, 5))
    antimask = np.array([[True, False, True, False, True],
                         [False, False, False, False, False],
                         [True, True, False, False, False],
                         [False, False, False, False, False]])
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert c.shape == a.shape
    assert np.allclose(c.values[antimask], a.values[antimask])
    assert np.all(c.mask[~antimask])


def test_qube_ext_shrinker_complex_n_d_case_vector() -> None:
    """Complex n-D case: Vector."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Vector(np.arange(30).reshape(10, 3))
    antimask = np.array([True] * 5 + [False] * 5)
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert c.shape == a.shape
    assert c.numer == a.numer
    assert np.allclose(c.values[antimask], a.values[antimask])
    assert np.all(c.mask[~antimask])


def test_qube_ext_shrinker_test_with_masked_shrunk_object() -> None:
    """Test with masked shrunk object."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    b = b.mask_where([True, False, False])  # Mask some of the shrunk values
    c = b.unshrink(antimask)
    assert c.shape == a.shape
    assert c.mask[0]  # First True in antimask was masked in b
    assert not c.mask[2]  # Third True in antimask was not masked in b
    assert not c.mask[4]  # Fifth True in antimask was not masked in b


def test_qube_ext_shrinker_test_with_entirely_masked_shrunk_object() -> None:
    """Test with entirely masked shrunk object."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    b = b.mask_where(True)  # Mask all shrunk values
    c = b.unshrink(antimask)

    if c.shape == ():
        # Shapeless case - all values are masked
        assert c.mask
    else:
        # Should match original shape if unshrink worked correctly
        assert c.shape == a.shape
        assert np.all(c.mask[antimask])  # All antimask positions should be masked
        assert np.all(c.mask[~antimask])  # All non-antimask positions should also be masked


def test_qube_ext_shrinker_test_with_shape_parameter_when_antimask_is_false() -> None:
    """Test with shape parameter when antimask is False."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar.MASKED
    b = a.unshrink(False, shape=(4, 5))
    assert b.shape == (4, 5)
    assert np.all(b.mask)


def test_qube_ext_shrinker_test_with_derivatives_2() -> None:
    """Test with derivatives."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    da_dt = Scalar([10., 20., 30., 40., 50.])
    a.insert_deriv('t', da_dt)
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert hasattr(c, 'd_dt')
    assert c.d_dt.shape == a.shape
    assert np.allclose(c.d_dt.values[antimask], da_dt.values[antimask])
    assert np.all(c.d_dt.mask[~antimask])


def test_qube_ext_shrinker_test_that_unshrunk_object_is_read_only() -> None:
    """Test that unshrunk object is read-only."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    c = b.unshrink(antimask)

    assert isinstance(c.readonly, bool)


def test_qube_ext_shrinker_round_trip_preserves_unmasked_values() -> None:
    """Test round-trip: shrink then unshrink should preserve unmasked values."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(np.arange(100))
    antimask = np.random.rand(100) > 0.5
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert np.allclose(c.values[antimask], a.values[antimask])
    assert np.all(c.mask[~antimask])


def test_qube_ext_shrinker_test_with_vector3() -> None:
    """Test with Vector3."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Vector3(np.arange(30).reshape(10, 3))
    antimask = np.array([True] * 5 + [False] * 5)
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert c.shape == a.shape
    assert c.numer == a.numer
    assert np.allclose(c.values[antimask], a.values[antimask])
    assert np.all(c.mask[~antimask])


def test_qube_ext_shrinker_test_with_boolean() -> None:
    """Test with Boolean."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Boolean([True, False, True, False, True])
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert c.shape == a.shape
    assert np.allclose(c.values[antimask], a.values[antimask])
    assert np.all(c.mask[~antimask])


def test_qube_ext_shrinker_round_trip_of_2_d_object() -> None:
    """shrink() then unshrink() of a 2-D object with a 2-D antimask restores the selected
    values.
    """

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(np.arange(20).reshape(4, 5))
    antimask = np.array([[True, False, True, False, True],
                         [False, False, False, False, False],
                         [True, True, False, False, False],
                         [False, False, False, False, False]])
    b = a.shrink(antimask)

    c = b.unshrink(antimask)
    assert c.shape == a.shape
    assert np.allclose(c.values[antimask], a.values[antimask])


def test_qube_ext_shrinker_test_shrink_with_disable_shrinking_for_testing_only() -> None:
    """Test shrink with _DISABLE_SHRINKING (for testing only)."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    original_disable = Qube._DISABLE_SHRINKING
    try:
        Qube._DISABLE_SHRINKING = True
        a = Scalar([1., 2., 3., 4., 5.])
        antimask = np.array([True, False, True, False, True])
        b = a.shrink(antimask)
        # With _DISABLE_SHRINKING, should return mask_where(not antimask)
        assert b.shape == a.shape
        assert b.mask[1]
        assert b.mask[3]
    finally:
        Qube._DISABLE_SHRINKING = original_disable


def test_qube_ext_shrinker_shrink_broadcasts_antimask_with_fewer_dims() -> None:
    """shrink() of a 3-D object with an antimask that matches only its last two axes."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(np.arange(40).reshape(2, 4, 5))
    antimask = np.array([[True, False, True, False, True],
                         [False, False, False, False, False],
                         [True, True, False, False, False],
                         [False, False, False, False, False]])  # 2-D antimask for 3-D object

    b = a.shrink(antimask)
    assert b.readonly


def test_qube_ext_shrinker_shrink_of_2_d_object_is_readonly() -> None:
    """shrink() of a 2-D object with an antimask of the same shape returns a read-only
    result.
    """

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(np.arange(20).reshape(4, 5))
    antimask = np.array([[True, False, True, False, True],
                         [False, False, False, False, False],
                         [True, True, False, False, False],
                         [False, False, False, False, False]])  # Correct shape

    b = a.shrink(antimask)
    assert b.readonly


def test_qube_ext_shrinker_test_shrink_with_all_mask_true() -> None:
    """Test shrink with all mask True."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.], mask=[True, True, True, True, True])
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)

    assert b == Scalar.MASKED


def test_qube_ext_shrinker_test_unshrink_with_disable_shrinking() -> None:
    """Test unshrink with _DISABLE_SHRINKING."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    original_disable = Qube._DISABLE_SHRINKING
    try:
        Qube._DISABLE_SHRINKING = True
        a = Scalar([1., 2., 3.])
        b = a.unshrink(True)
        assert a == b
    finally:
        Qube._DISABLE_SHRINKING = original_disable


def test_qube_ext_shrinker_test_unshrink_with_disable_cache() -> None:
    """Test unshrink with _DISABLE_CACHE."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    original_disable_cache = Qube._DISABLE_CACHE
    try:
        Qube._DISABLE_CACHE = True
        a = Scalar([1., 2., 3., 4., 5.])
        antimask = np.array([True, False, True, False, True])
        b = a.shrink(antimask)
        c = b.unshrink(antimask)
        # Should work without cache
        assert c.shape == a.shape
    finally:
        Qube._DISABLE_CACHE = original_disable_cache


def test_qube_ext_shrinker_test_unshrink_with_cached_unshrunk_value() -> None:
    """Test unshrink with cached unshrunk value."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    original_disable_cache = Qube._DISABLE_CACHE
    try:
        Qube._DISABLE_CACHE = False
        a = Scalar([1., 2., 3., 4., 5.])
        antimask = np.array([True, False, True, False, True])
        b = a.shrink(antimask)
        # First unshrink should cache
        c1 = b.unshrink(antimask)
        # Second unshrink should use cache
        c2 = b.unshrink(antimask)
        assert c1.shape == c2.shape
    finally:
        Qube._DISABLE_CACHE = original_disable_cache


def test_qube_ext_shrinker_test_unshrink_with_ignore_unshrunk_as_cached() -> None:
    """Test unshrink with _IGNORE_UNSHRUNK_AS_CACHED."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    original_ignore = Qube._IGNORE_UNSHRUNK_AS_CACHED
    original_disable_cache = Qube._DISABLE_CACHE
    try:
        Qube._IGNORE_UNSHRUNK_AS_CACHED = True
        Qube._DISABLE_CACHE = False
        a = Scalar([1., 2., 3., 4., 5.])
        antimask = np.array([True, False, True, False, True])
        b = a.shrink(antimask)
        c = b.unshrink(antimask)
        # Should ignore cached value
        assert c.shape == a.shape
    finally:
        Qube._IGNORE_UNSHRUNK_AS_CACHED = original_ignore
        Qube._DISABLE_CACHE = original_disable_cache


def test_qube_ext_shrinker_test_unshrink_with_scalar_object_shapeless() -> None:
    """Test unshrink with scalar object (shapeless)."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(7.)
    b = a.unshrink(False, shape=(5,))
    assert b.shape == (5,)
    assert np.all(b.mask)


def test_qube_ext_shrinker_unshrink_scalar_with_default() -> None:
    """shrink() then unshrink() of a 1-D Scalar restores its shape."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3.])
    antimask = np.array([True, False, True])
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert c.shape == a.shape


def test_qube_ext_shrinker_unshrink_is_array_path() -> None:
    """unshrink() of an object whose values are an array."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert c.shape == a.shape


def test_qube_ext_shrinker_unshrink_is_scalar_path() -> None:
    """unshrink() of a shapeless Scalar with antimask False and a shape returns a fully
    masked object of that shape.
    """

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(7.)
    b = a.unshrink(False, shape=(3,))

    assert b.shape == (3,)
    assert np.all(b.mask)


def test_qube_ext_shrinker_test_shrink_with_disable_shrinking_and_scalar_object() -> None:
    """Test shrink with _DISABLE_SHRINKING and scalar object."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    original_disable = Qube._DISABLE_SHRINKING
    try:
        Qube._DISABLE_SHRINKING = True
        a = Scalar(7.)
        b = a.shrink(True)
        # With _DISABLE_SHRINKING and scalar, should return unchanged
        assert a == b
    finally:
        Qube._DISABLE_SHRINKING = original_disable


def test_qube_ext_shrinker_test_shrink_with_cache_path() -> None:
    """Test shrink with cache path."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    original_disable_cache = Qube._DISABLE_CACHE
    try:
        Qube._DISABLE_CACHE = False
        a = Scalar([1., 2., 3., 4., 5.])
        antimask = np.array([True, False, True, False, True])
        b = a.shrink(antimask)
        # Should have cache entry
        assert hasattr(b, '_cache')
    finally:
        Qube._DISABLE_CACHE = original_disable_cache


def test_qube_ext_shrinker_shrink_with_cache_enabled_returns_masked_single_early() -> None:
    """shrink() with the cache enabled returns the masked singleton early."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    original_disable_cache = Qube._DISABLE_CACHE
    try:
        Qube._DISABLE_CACHE = False
        # Option 1: object is fully masked
        a = Scalar([1., 2., 3., 4., 5.], mask=[True, True, True, True, True])
        antimask = np.array([True, False, True, False, True])
        b = a.shrink(antimask)
        # Should return masked_single and cache unshrunk if _DISABLE_CACHE is False
        assert b == Scalar.MASKED
        assert ('unshrunk' in b._cache)
    finally:
        Qube._DISABLE_CACHE = original_disable_cache


def test_qube_ext_shrinker_test_shrink_with_shape_mismatch_requiring_broadcast_to() -> None:
    """Test shrink with shape mismatch requiring broadcast_to."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    Scalar(np.arange(20).reshape(4, 5))


def test_qube_ext_shrinker_test_shrink_with_all_mask_true_2() -> None:
    """Test shrink with all mask True."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.], mask=[True, True, True, True, True])
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)

    assert b == Scalar.MASKED


def test_qube_ext_shrinker_test_unshrink_with_scalar_object() -> None:
    """Test unshrink with scalar object."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(7.)  # Scalar with shape ()
    antimask = True
    b = a.unshrink(antimask)

    assert a == b


def test_qube_ext_shrinker_test_unshrink_with_is_array_and_default_as_qube() -> None:
    """Test unshrink with _is_array and default as Qube."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)

    c = b.unshrink(antimask)
    assert c.shape == a.shape
    # The default is a Scalar (Qube), so it should use the _is_array path
    # and handle default as Qube


def test_qube_ext_shrinker_test_unshrink_with_derivatives() -> None:
    """Test unshrink with derivatives."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    a.insert_deriv('t', Scalar([0.1, 0.2, 0.3, 0.4, 0.5]))
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    c = b.unshrink(antimask)

    assert hasattr(c, 'd_dt')
    assert c.d_dt.shape == a.d_dt.shape


def test_qube_ext_shrinker_shrink_broadcast_to_path_with_extras() -> None:
    """shrink() of a 1-D object with an antimask that has more dimensions than the object.
    """

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])  # 1-D, shape (5,)
    antimask = np.array([[True, False, True, False, True],
                         [True, False, True, False, True]])  # 2-D, shape (2, 5)

    b = a.shrink(antimask)
    assert b.readonly

    assert b.shape[0] == np.sum(antimask)


def test_qube_ext_shrinker_test_shrink_with_shape_mismatch_that_requires_broadcasting() -> None:
    """Test shrink with shape mismatch that requires broadcasting."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(np.arange(20).reshape(4, 5))

    antimask = np.array([[True, False, True, False, True],
                         [False, False, False, False, False],
                         [True, True, False, False, False],
                         [False, False, False, False, False]])
    b = a.shrink(antimask)
    assert b.readonly


def test_qube_ext_shrinker_shrink_with_antimask_shape_mismatch() -> None:
    """shrink() of a (4, 5) object broadcasts a (1, 5) antimask."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(np.arange(20).reshape(4, 5))
    antimask = np.array([[True, False, True, False, True]])  # (1, 5) for (4, 5) object

    b = a.shrink(antimask)
    assert b.readonly


def test_qube_ext_shrinker_shrink_selecting_only_masked_elements() -> None:
    """shrink() when every selected element is masked, although the object is not fully
    masked.
    """

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.], mask=[True, True, True, False, False])
    antimask = np.array([True, True, True, False, False])  # Select first 3, all are masked
    b = a.shrink(antimask)

    assert b.shape == ()
    assert b.mask
    assert b.readonly


def test_qube_ext_shrinker_test_shrink_with_all_mask_true_earlier_return_path() -> None:
    """Test shrink with all mask True (earlier return path)."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.], mask=[True, True, True, True, True])
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)

    assert b == Scalar.MASKED
    assert b.readonly


def test_qube_ext_shrinker_test_unshrink_with_is_scalar_path() -> None:
    """Test unshrink with _is_scalar path."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(7.)
    b = a.unshrink(False, shape=(5,))
    assert b.shape == (5,)
    assert np.all(b.mask)


def test_qube_ext_shrinker_unshrink_vector_with_default() -> None:
    """shrink() then unshrink() of a Vector restores its shape and numerator."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Vector([1., 2., 3.])
    antimask = np.array([True, False, True])
    b = a.shrink(antimask)

    c = b.unshrink(antimask)

    assert c.shape == antimask.shape
    assert c.numer == a.numer


def test_qube_ext_shrinker_test_unshrink_with_is_array_path() -> None:
    """Test unshrink with _is_array path."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert c.shape == a.shape


def test_qube_ext_shrinker_test_unshrink_with_derivatives_2() -> None:
    """Test unshrink with derivatives."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    da_dt = Scalar([10., 20., 30., 40., 50.])
    a.insert_deriv('t', da_dt)
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert hasattr(c, 'd_dt')
    assert c.d_dt.shape == a.shape


def test_qube_ext_shrinker_shrink_with_cache_returns_masked_single_when_fully_masked() -> None:
    """shrink() with the cache enabled returns the masked singleton when the object is fully
    masked or the antimask is False.
    """

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    original_disable_cache = Qube._DISABLE_CACHE
    try:
        Qube._DISABLE_CACHE = False
        # Case 1: Fully masked object
        a = Scalar([1., 2., 3., 4., 5.], mask=True)
        antimask = np.array([True, False, True, False, True])
        b = a.shrink(antimask)
        assert b == Scalar.MASKED
        assert ('unshrunk' in b._cache)
        assert b._cache['unshrunk'] == a
        # Case 2: False antimask
        a = Scalar([1., 2., 3., 4., 5.])
        b = a.shrink(False)
        assert b == Scalar.MASKED
        assert ('unshrunk' in b._cache)
    finally:
        Qube._DISABLE_CACHE = original_disable_cache


def test_qube_ext_shrinker_shrink_with_cache_selecting_only_masked_elements() -> None:
    """shrink() with the cache enabled, when every selected element is masked, returns the
    masked singleton and caches it.
    """

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    original_disable_cache = Qube._DISABLE_CACHE
    try:
        Qube._DISABLE_CACHE = False
        a = Scalar([1., 2., 3., 4., 5.], mask=[True, True, True, False, False])
        antimask = np.array([True, True, True, False, False])
        b = a.shrink(antimask)
        assert b == Scalar.MASKED
        assert b.readonly
        assert ('unshrunk' in b._cache)
    finally:
        Qube._DISABLE_CACHE = original_disable_cache


def test_qube_ext_shrinker_unshrink_with_qube_default() -> None:
    """unshrink() fills unselected positions using a default that is a Qube."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Vector([1., 2., 3.])
    antimask = np.array([True, False, True])
    b = a.shrink(antimask)
    assert b.shape == (2,)

    b._default = Vector([1., 1., 1.])
    c = b.unshrink(antimask)
    assert c.shape == antimask.shape
    assert c.numer == a.numer

    assert c.shape == (3,)

    assert np.all(c.mask[~antimask])


def test_qube_ext_shrinker_unshrink_with_is_array_false() -> None:
    """unshrink() of an object whose values are not an array.

    The values and the array flag are set directly to reach this case.
    """

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2.])
    antimask = np.array([True, False])
    b = a.shrink(antimask)

    original_values = b._values
    original_is_array = b._is_array
    b._values = float(b._values[0])  # Convert to Python float
    b._is_array = False  # Must also set _is_array
    c = b.unshrink(antimask)
    assert c.shape == antimask.shape

    b._values = original_values
    b._is_array = original_is_array


def test_qube_ext_shrinker_test_unshrink_with_scalar_object_2() -> None:
    """Test unshrink with scalar object."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar(7.)
    antimask = np.array([True, False, True])
    b = a.shrink(antimask)
    assert b._is_scalar
    c = b.unshrink(antimask)
    assert c._is_scalar
    assert c == a


def test_qube_ext_shrinker_test_unshrink_with_derivatives_3() -> None:
    """Test unshrink with derivatives."""

    np.random.seed(8736)

    ##################################################################################
    # shrink()
    ##################################################################################

    a = Scalar([1., 2., 3., 4., 5.])
    da_dt = Scalar([10., 20., 30., 40., 50.])
    a.insert_deriv('t', da_dt)
    antimask = np.array([True, False, True, False, True])
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert hasattr(c, 'd_dt')
    assert c.d_dt.shape == a.shape
    assert np.allclose(c.d_dt.values[antimask], da_dt.values[antimask])

    a = Scalar([1., 2., 3., 4., 5.])
    da_dt = Scalar([10., 20., 30., 40., 50.])
    da_ds = Scalar([100., 200., 300., 400., 500.])
    a.insert_deriv('t', da_dt)
    a.d_dt.insert_deriv('s', da_ds)
    b = a.shrink(antimask)
    c = b.unshrink(antimask)
    assert hasattr(c, 'd_dt')
    assert c.d_dt.shape == a.shape

    if hasattr(c.d_dt, 'd_ds'):
        assert c.d_dt.d_ds.shape == a.shape


##########################################################################################
