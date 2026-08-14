##########################################################################################
# tests/test_quaternion_parts.py
##########################################################################################

import numpy as np

from polymath import Quaternion, Scalar


def test_quaternion_parts() -> None:
    """Exercise quaternion parts."""

    np.random.seed(3219)
    a = Quaternion.from_parts(1., [(1,0,0),(0,1,0),(0,0,1)])
    assert a.shape == (3,)
    assert a[0] == (1,1,0,0)
    assert a[1] == (1,0,1,0)
    assert a[2] == (1,0,0,1)
    assert not a.readonly
    a = Quaternion.from_parts(1., [(1,0,0),(0,1,0),(0,0,1)])
    a.insert_deriv('t', Quaternion((1.,2.,3.,4.)))
    assert a.d_dt.shape == (3,)
    assert a.d_dt[0] == (1,2,3,4)
    assert a.d_dt[1] == (1,2,3,4)
    assert a.d_dt[2] == (1,2,3,4)
    angle = Scalar(0., derivs={'t': Scalar(1.)})
    a = Quaternion.from_rotation(angle, [(1,0,0),(0,1,0),(0,0,1)])
    (s,v) = a.to_parts()
    assert s == 1.
    assert s.d_dt == 0.
    assert v == (0,0,0)
    assert v[0].d_dt == (0.5,0,0)
    assert v[1].d_dt == (0,0.5,0)
    assert v[2].d_dt == (0,0,0.5)
    assert not s.readonly
    assert not v.readonly
    ####
    N = 100
    q = Quaternion(np.random.randn(N,4), mask=(np.random.rand(N) < 0.2))
    dq_dt = Quaternion(np.random.randn(N,4,2), mask=(np.random.rand(N) < 0.2),
                       drank=1)
    q.insert_deriv('t', dq_dt)
    (s,v) = q.to_parts(recursive=False)
    assert hasattr(q, 'd_dt') == True
    assert hasattr(s, 'd_dt') == False
    assert hasattr(v, 'd_dt') == False
    assert q.readonly == False
    assert s.readonly == False
    assert v.readonly == False
    assert np.all(s.values == q.values[...,0])
    assert np.all(v.values == q.values[...,1:4])
    s.values[0] = 42.
    assert q.values[0,0] == 42.    # demonstrates shared memory
    v.values[0,0] = 42.
    assert q.values[0,1] == 42.
    (s,v) = q.to_parts(recursive=True)
    assert hasattr(q, 'd_dt') == True
    assert hasattr(s, 'd_dt') == True
    assert hasattr(v, 'd_dt') == True
    assert q.readonly == False
    assert s.readonly == False
    assert v.readonly == False
    assert q.d_dt.readonly == False
    assert s.d_dt.readonly == False
    assert v.d_dt.readonly == False
    assert np.all(s.d_dt.values == q.d_dt.values[...,0,:])
    assert np.all(v.d_dt.values == q.d_dt.values[...,1:4,:])
    q = q.as_readonly()
    (s,v) = q.to_parts(recursive=True)
    assert q.readonly == True
    assert s.readonly == True
    assert v.readonly == True
    assert q.d_dt.readonly == True
    assert s.d_dt.readonly == True
    assert v.d_dt.readonly == True


##########################################################################################
