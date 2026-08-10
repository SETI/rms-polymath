##########################################################################################
# tests/test_qube_new_from_parts.py
##########################################################################################

import numpy as np
import pytest

from polymath import Matrix, Qube, Scalar, Unit, Vector, Vector3


def _attrs(obj: Qube) -> dict:
    """Every shape and type attribute that the two constructors both determine."""

    return {name: getattr(obj, name)
            for name in ('_shape', '_ndims', '_rank', '_nrank', '_drank', '_item',
                         '_numer', '_denom', '_size', '_isize', '_nsize', '_dsize',
                         '_readonly', '_is_array', '_is_scalar', '_unit')}


@pytest.mark.parametrize(('shape', 'nrank', 'drank'), [
    ((), 0, 0),
    ((5,), 0, 0),
    ((2, 3), 0, 0),
    ((5,), 1, 0),
    ((2, 3), 1, 0),
    ((5,), 1, 1),
    ((), 2, 0),
    ((4,), 2, 1),
])
def test_qube_new_from_parts_matches_the_constructor(shape: tuple, nrank: int,
                                                     drank: int) -> None:
    """The fast constructor derives the same shape attributes as __init__()."""

    rng = np.random.default_rng(11)
    item = (3,) * nrank + (2,) * drank
    values = rng.normal(size=shape + item)
    if not shape and not item:
        values = float(values)

    fast = Qube._new_from_parts(values, False, nrank=nrank, drank=drank, unit=Unit.KM)
    slow = Qube(values, False, nrank=nrank, drank=drank, unit=Unit.KM)

    assert _attrs(fast) == _attrs(slow)
    assert np.all(np.asarray(fast.values) == np.asarray(slow.values))


def test_qube_new_from_parts_reduces_a_numpy_scalar() -> None:
    """A NumPy scalar is stored as a Python scalar, as the constructor stores it."""

    obj = Qube._new_from_parts(np.float64(2.5), False, nrank=0)
    assert type(obj.values) is float
    assert obj._is_scalar


def test_qube_new_from_parts_reduces_a_shapeless_array() -> None:
    """A zero-dimensional array is stored as a Python scalar."""

    obj = Qube._new_from_parts(np.array(7.5), False, nrank=0)
    assert type(obj.values) is float
    assert obj.values == 7.5
    assert obj.shape == ()


def test_qube_new_from_parts_broadcasts_a_narrow_mask() -> None:
    """A mask narrower than the values is broadcast to the leading shape."""

    values = np.zeros((4, 3))
    obj = Qube._new_from_parts(values, np.array([True, False, True, False]), nrank=1)

    assert obj.shape == (4,)
    assert obj.mask.shape == (4,)

    wide = Qube._new_from_parts(np.zeros((4, 5)), np.array([[True], [False],
                                                            [True], [False]]), nrank=0)
    assert wide.mask.shape == (4, 5)
    assert list(wide.mask[:, 0]) == [True, False, True, False]


def test_qube_new_from_parts_takes_the_default_from_a_matching_example() -> None:
    """The default is reused when the example has the same item shape and dtype."""

    example = Vector3(np.zeros((4, 3)))
    obj = Qube._new_from_parts(np.ones((4, 3)), False, nrank=1, example=example)
    assert obj._default is example._default


def test_qube_new_from_parts_recomputes_the_default_for_a_new_item_shape() -> None:
    """The default is recomputed when the operation changed the item shape."""

    example = Vector3(np.zeros((4, 3)))
    obj = Qube._new_from_parts(np.ones((4,)), False, nrank=0, example=example)

    assert obj._default is not example._default
    assert obj._default == 1.


def test_qube_new_from_parts_recomputes_the_default_for_a_new_dtype() -> None:
    """The default is recomputed when the operation changed the dtype."""

    example = Scalar(np.zeros(4, dtype='int'))
    obj = Qube._new_from_parts(np.ones(4, dtype='float'), False, nrank=0,
                               example=example)

    assert type(example._default) is int
    assert type(obj._default) is float


def test_qube_new_from_parts_marks_a_read_only_array() -> None:
    """A read-only values array yields a read-only object with a read-only mask."""

    values = np.zeros((4, 3))
    values.flags['WRITEABLE'] = False
    mask = np.zeros(4, dtype='bool')

    obj = Qube._new_from_parts(values, mask, nrank=1)
    assert obj.readonly
    assert not obj.mask.flags['WRITEABLE']


def test_qube_default_for_uses_the_class_default() -> None:
    """_default_for() prefers the class default when there is no denominator."""

    assert Scalar._default_for((), 0, 'float') == 1.
    assert type(Scalar._default_for((), 0, 'int')) is int
    assert np.all(Matrix._default_for((3, 3), 0, 'float') == np.ones((3, 3)))
    assert np.all(Vector._default_for((3, 2), 1, 'float') == np.ones((3, 2)))


##########################################################################################
