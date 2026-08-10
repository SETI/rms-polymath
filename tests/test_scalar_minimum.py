##########################################################################################
# tests/test_scalar_minimum.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar


def test_scalar_minimum() -> None:
    """Exercise scalar minimum."""

    np.random.seed(2251)
    with pytest.raises(ValueError):
        Scalar.minimum()
    a = Scalar(np.random.randn(10,1))
    assert Scalar.minimum(a) == a
    assert Scalar.minimum(a,100) == a
    assert Scalar.minimum(a,100,Scalar.MASKED) == a
    b = Scalar(np.random.randn(4,1,10))
    assert Scalar.minimum(a,b).shape == (4,10,10)
    ab = Scalar.minimum(a,b,100,Scalar.MASKED)
    ab2 = Scalar(np.minimum(a.values,b.values))
    assert ab == ab2
    a = Scalar(np.random.randn(10,1), np.random.randn(10,1) < -0.5)
    b = Scalar(np.random.randn(4,1,10), np.random.randn(4,1,10) < -0.5)
    ab = Scalar.minimum(a,b)
    for i in range(4):
        for j in range(10):
            for k in range(10):
                if a.mask[j,0] and b.mask[i,0,k]:
                    assert ab[i,j,k].mask
                elif a.mask[j,0]:
                    assert ab[i,j,k].vals == b[i,0,k].vals
                    assert not ab[i,j,k].mask
                elif b.mask[i,0,k]:
                    assert ab[i,j,k].vals == a[j,0].vals
                    assert not ab[i,j,k].mask
                else:
                    assert ab[i,j,k] == min(a[j,0],b[i,0,k])
                    assert not ab[i,j,k].mask


##########################################################################################
