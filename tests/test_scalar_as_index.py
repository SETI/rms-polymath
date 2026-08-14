##########################################################################################
# tests/test_scalar_as_index.py
##########################################################################################

import numpy as np

from polymath import Scalar


def test_scalar_as_index() -> None:
    """Exercise scalar as index."""

    a = Scalar(np.arange(12).reshape(3,4))
    assert np.all(a.as_index() == a.values)
    mask = a.values % 2 == 0
    a = Scalar(np.arange(12).reshape(3,4), mask)
    assert np.all(a.as_index() == np.arange(1,12,2))
    test = a.as_index(masked=-7)
    assert test.shape == (3,4)
    for i in range(3):
        for j in range(4):
            if mask[i,j]:
                assert test[i,j] == -7
            else:
                assert test[i,j] == a.values[i,j]


def test_scalar_as_index_and_mask_without_purge_or_replacement() -> None:
    """Masked items keep their values when purge is False and masked is None."""

    a = Scalar([1, 2, 3], [False, True, False])
    (index, mask) = a.as_index_and_mask(purge=False, masked=None)
    assert np.all(index == [1, 2, 3])
    assert index.dtype.kind == 'i'
    assert np.all(mask == [False, True, False])


##########################################################################################
