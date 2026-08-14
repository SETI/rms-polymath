##########################################################################################
# tests/test_vector_int.py
##########################################################################################

import numpy as np
import pytest

from polymath import Pair, Scalar, Unit, Vector, Vector3


def test_vector_int_int_input() -> None:
    """int input."""

    np.random.seed(5394)

    a = Vector(np.arange(30).reshape(10,3))
    b = a.int()
    assert a is b
    a = Vector3(np.arange(30).reshape(10,3), unit=Unit.KM)
    with pytest.raises(ValueError) as cm:
        b = a.int()
    assert str(cm.value) == 'Vector3.int() unit is not permitted: km'
    a = Pair(np.arange(60).reshape(10,2,3), drank=1)
    with pytest.raises(ValueError) as cm:
        b = a.int()
    assert str(cm.value) == 'Pair.int() does not support denominators'
    a = Pair(np.arange(-40.,40.).reshape(-1,2)/10.)
    b = a.int()
    assert np.all(b.vals == np.floor(a.vals))
    assert b.is_int()
    assert not b.mask
    a = Pair(np.arange(-40.,40.).reshape(-1,2)/10.)
    b = a.int(remask=True)
    assert np.all(b.vals == np.floor(a.vals))
    assert b.is_int()
    assert np.all(b.vals[b.mask] < 0)
    assert np.all(b.vals[~b.mask] >= 0)


def test_vector_int_top_2() -> None:
    """top = 2."""

    np.random.seed(5394)

    a = Pair(np.arange(-40.,40.).reshape(-1,2)/10.)
    a.int(top=(2,3))

    # TBD!

    ##################################################################################
    # Additional coverage tests
    ##################################################################################


def test_vector_int_test_int_with_top_none_and_negative_values_clip_true() -> None:
    """Test int() with top=None and negative values, clip=True."""

    np.random.seed(5394)

    a = Vector([-1., 2., 3.])
    b = a.int(top=None, clip=True)
    assert b.values[0] == 0
    assert b.values[1] == 2
    assert b.values[2] == 3


def test_vector_int_test_vector_scale_with_recursive_false() -> None:
    """Test vector_scale with recursive=False."""

    np.random.seed(5394)

    v = Vector([1., 0., 0.])
    factor = Vector([2., 0., 0.])
    result = v.vector_scale(factor, recursive=False)
    assert type(result) == Vector


def test_vector_int_test_combos_with_all_int_scalars() -> None:
    """Test combos with all int scalars."""

    np.random.seed(5394)

    s1 = Scalar([1, 2])
    s2 = Scalar([3, 4])
    v = Vector.combos(s1, s2)
    assert v.shape == (2, 2)
    assert v.numer == (2,)
    assert v.is_int()


def test_vector_int_a_single_top_applies_to_every_component() -> None:
    """int() accepts one `top` value and applies it to every component."""

    v = Vector([[1., 7.]])
    assert list(v.int(top=5, clip=True).values[0]) == [1, 4]
    assert list(v.int(top=(5, 9), clip=True).values[0]) == [1, 7]


def test_vector_int_a_top_of_the_wrong_length_is_rejected() -> None:
    """int() rejects a `top` sequence whose length does not match the item shape."""

    with pytest.raises(ValueError, match='top does not match item shape'):
        Vector([[1., 7.]]).int(top=(5, 9, 11))


##########################################################################################


def test_vector_int_options_are_keyword_only() -> None:
    """int() takes remask, clip, inclusive and shift by keyword, as Scalar.int() does."""

    v = Vector([[1.6, 2.4]])
    assert v.int(remask=False).values.tolist() == [[1, 2]]

    with pytest.raises(TypeError, match='positional argument'):
        v.int(None, True)


def test_vector_as_index_and_mask_options_are_keyword_only() -> None:
    """as_index_and_mask() takes purge and masked by keyword, as Scalar's does."""

    v = Vector([[1, 2]])
    index, mask = v.as_index_and_mask(purge=False, masked=None)
    assert len(index) == 2
    assert mask is False

    with pytest.raises(TypeError, match='positional argument'):
        v.as_index_and_mask(True)
