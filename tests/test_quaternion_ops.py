##########################################################################################
# tests/test_quaternion_ops.py
##########################################################################################

import numpy as np

from polymath import Quaternion


def test_quaternion_ops_multiply() -> None:
    """Multiply."""

    np.random.seed(8291)
    N = 3
    M = 2
    a = Quaternion(np.random.randn(N,1,4))
    a.insert_deriv('t', Quaternion(np.random.randn(N,1,4,2), drank=1))
    b = Quaternion(np.random.randn(M,4))
    b.insert_deriv('t', Quaternion(np.random.randn(M,4,2), drank=1))
    assert a == a * Quaternion.IDENTITY
    assert a == a / Quaternion.IDENTITY
    assert a == a + Quaternion.ZERO
    assert a == a - Quaternion.ZERO

    (sa,va) = a.to_parts()
    (sb,vb) = b.to_parts()

    sab = sa * sb - va.dot(vb)
    vab = sa * vb + sb * va + va.cross(vb)
    ab = Quaternion.from_parts(sab, vab)
    DEL = 1.e-14
    assert ((ab - a*b).rms().max() < DEL)
    dab_dt = a.wod * b.d_dt + a.d_dt * b.wod
    assert ((dab_dt - (a*b).d_dt).rms().max() < DEL)

    test = ab / b
    assert ((test - a).rms().max() < DEL)
    b_inv = b.reciprocal()
    test = ab * b_inv
    assert ((test - a).rms().max() < DEL)
    dtest_dt = ab.d_dt * b_inv.wod + ab.wod * b_inv.d_dt
    assert ((dtest_dt - a.d_dt).rms().max() < DEL)


##########################################################################################
