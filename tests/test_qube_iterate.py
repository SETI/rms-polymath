##########################################################################################
# tests/test_qube_iterate.py
##########################################################################################

import numpy as np

from polymath import Pair, Scalar


def test_qube_iterate_shape() -> None:
    """shape ()."""

    array = Scalar(np.arange(10))
    count = 0
    for a in array:
        assert a == count
        assert isinstance(a, Scalar)
        count += 1
    array = Scalar(np.arange(10))
    count = 0
    for a in array.__iter__():
        assert a == count
        assert isinstance(a, Scalar)
        count += 1
    array = Scalar(np.arange(10), mask=[1,1,1,1,1,0,0,0,0,0])
    count = 0
    for a in array:
        assert a.vals == count
        assert a.mask == (count < 5)
        assert isinstance(a, Scalar)
        count += 1
    array = Pair(list(zip(np.arange(10), -3 * np.arange(10), strict=False)))
    count = 0
    for a in array:
        assert a == (count, -3 * count)
        assert isinstance(a, Pair)
        count += 1
    count = 0
    for k,a in enumerate(array):
        assert a == (k, -3 * k)
        assert k == count
        assert isinstance(a, Pair)
        count += 1
    count = 0
    for k,a in array.ndenumerate():
        assert a == (k[0], -3 * k[0])
        assert a == array[k]
        assert k[0] == count
        assert isinstance(a, Pair)
        count += 1
    array = Scalar(np.arange(10).reshape(5,2))
    for k,a in enumerate(array):
        assert a == (2*k, 2*k+1)
        assert a == array[k]
    for k,a in array.ndenumerate():
        assert a == array[k]

    array = Scalar(7)
    count = 0
    for a in array:
        assert a == array
        count += 1
    assert count == 1
    count = 0
    for k,a in array.ndenumerate():
        assert k[0] == 0
        assert a == array
        count += 1
    assert count == 1


##########################################################################################
