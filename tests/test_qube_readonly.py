##########################################################################################
# tests/test_qube_readonly.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Vector


def test_qube_readonly() -> None:
    """Exercise qube readonly."""

    np.random.seed(6687)
    a = Vector(np.random.randn(4,5,6,3,2), drank=1)
    assert a.readonly == False
    a.values[0,0,0,0,0] = 1.
    a = a.as_readonly()
    assert a.readonly == True
    with pytest.raises((ValueError,RuntimeError)):
        a.values.__setitem__((0,0,0,0,0), 1.)
    a = Scalar(np.arange(10)).as_readonly()
    b = a.copy()
    c = a.clone()
    assert a.readonly == True
    assert b.readonly == False
    assert c.readonly == True
    b[0] = 10
    assert a[0] == 0
    assert b[0] == 10
    assert c[0] == 0
    with pytest.raises(ValueError):
        a.__setitem__(0, 10)
    with pytest.raises(ValueError):
        c.__setitem__(0, 10)
    a = Scalar(np.arange(10)).as_readonly()
    b = a.copy(readonly=True)
    assert a.readonly == True
    assert b.readonly == True
    with pytest.raises(ValueError):
        b.__setitem__(0, 10)
    with pytest.raises(ValueError):
        b[0].__iadd__(10)
    a = Vector(np.random.randn(5,3))
    da_dm = Vector(np.random.randn(5,3,2,3), drank=2)
    a.insert_deriv('m', da_dm)
    assert a.readonly == False
    assert a.d_dm.readonly == False
    b = a.copy(readonly=True, recursive=False)
    assert b.readonly == True
    assert not hasattr(b, 'd_dm')
    b = a.copy(readonly=False, recursive=True)
    assert b.readonly == False
    assert b.d_dm.readonly == False
    b = a.copy(readonly=True, recursive=True)
    assert b.readonly == True
    assert b.d_dm.readonly == True
    a = Vector(np.random.randn(5,3))
    da_dm = Vector(np.random.randn(5,3,2,3), drank=2)
    a.insert_deriv('m', da_dm)
    assert a.readonly == False
    assert a.d_dm.readonly == False
    b = a.copy()
    assert np.all(a.values == b.values)
    b.values[0,0] = 42
    assert (a.values[0,0] != 42)
    b.d_dm.values[0,0,0,0] = 42
    assert (a.d_dm.values[0,0,0,0] != 42)


##########################################################################################
