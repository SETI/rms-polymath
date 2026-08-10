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


##########################################################################################
