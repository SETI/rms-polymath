##########################################################################################
# tests/test_vector_element_div.py
##########################################################################################

import numpy as np
import pytest

from polymath import Pair, Unit, Vector, Vector3


def test_vector_element_div_single_values() -> None:
    """Single values."""

    np.random.seed(1472)

    assert Vector((2,21,0)).element_div((1,3,1)) == (2,7,0)
    assert Vector((20,30,40)).element_div((10,10,-20)) == (2,3,-2)
    assert Vector((2,3,0),True).element_div((10,10,-20)).mask
    assert Vector((2,3,0),False).element_div((10,10,0)).mask
    vec = Vector3((2,3,0)).element_div(Vector((10,10,0)))
    assert type(vec) is Vector3
    vec = Vector((2,3,0)).element_div(Vector3((10,10,0)))
    assert type(vec) is Vector
    vec = Pair((2,3)).element_div(Vector((10,0)))
    assert type(vec) is Pair
    vec = Vector((2,3)).element_div(Pair((10,0)))
    assert type(vec) is Vector

    N = 100
    x = Vector(np.random.randn(N,5))
    y = Vector(np.random.randn(N,5))
    z = y.element_div(x)
    DEL = 3.e-12
    for i in range(N):
        for _k in range(5):
            assert z[i] == y.values[i]/x.values[i] or abs(z[i] - y.values[i]/x.values[i]) <= DEL
    N = 100
    x = Vector(np.random.randn(N,4), np.random.randn(N) < -0.5)
    y = Vector(np.random.randn(N,4), np.random.randn(N) < -0.5)
    z = y.element_div(x)
    assert np.all(z.mask == (x.mask | y.mask))

    zz = z[~z.mask]
    xx = x[~z.mask]
    yy = y[~z.mask]
    for i in range(len(zz)):
        for _k in range(4):
            assert zz[i] == yy.values[i]/xx.values[i] or abs(zz[i] - yy.values[i]/xx.values[i]) <= DEL
    N = 100
    x = Vector(np.random.randn(N,4), np.random.randn(N) < -0.5)
    y = Vector(np.random.randn(N,4), np.random.randn(N) < -0.5)
    zero_mask = (np.random.randn(N,4) < -1.)
    x.values[zero_mask] = 0.
    zero_mask = np.any(zero_mask, axis=-1)
    z = y.element_div(x)
    assert np.all(z.mask[x.mask])
    assert np.all(z.mask[y.mask])
    assert np.all(z.mask[zero_mask])
    assert np.all(z.mask == (x.mask | y.mask | zero_mask))
    for i in range(N):
        for _k in range(4):
            if not z[i].mask:
                assert z[i] == y.values[i]/x.values[i] or abs(z[i] - y.values[i]/x.values[i]) <= DEL

    N = 100
    x = Vector(np.random.randn(N,3), unit=Unit.S)
    y = Vector(np.random.randn(N,3), unit=Unit.KM)
    z = y.element_div(x)
    assert z.unit_ == Unit.KM/Unit.SECONDS

    N = 100
    x = Vector(np.random.randn(N*3).reshape((N,3)))
    y = Vector(np.random.randn(N*3).reshape((N,3)))
    x.insert_deriv('f', Vector(np.random.randn(N,3)))
    x.insert_deriv('h', Vector(np.random.randn(N,3)))
    y.insert_deriv('g', Vector(np.random.randn(N,3)))
    y.insert_deriv('h', Vector(np.random.randn(N,3)))
    z = y.element_div(x)
    assert 'f' in x.derivs
    assert hasattr(x, 'd_df')
    assert 'g' not in x.derivs
    assert not hasattr(x, 'd_dg')
    assert 'h' in x.derivs
    assert hasattr(x, 'd_dh')
    assert 'f' not in y.derivs
    assert not hasattr(y, 'd_df')
    assert 'g' in y.derivs
    assert hasattr(y, 'd_dg')
    assert 'h' in y.derivs
    assert hasattr(y, 'd_dh')
    assert 'f' in z.derivs
    assert hasattr(z, 'd_df')
    assert 'g' in z.derivs
    assert hasattr(z, 'd_dg')
    assert 'h' in z.derivs
    assert hasattr(z, 'd_dh')

    EPS = 1.e-6
    z1 = y.element_div(x + (EPS,0,0))
    z0 = y.element_div(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS
    z1 = y.element_div(x + (0,EPS,0))
    z0 = y.element_div(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS
    z1 = y.element_div(x + (0,0,EPS))
    z0 = y.element_div(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS
    new_values = np.empty((N,3,3))
    new_values[...,0] = dz_dx0.values
    new_values[...,1] = dz_dx1.values
    new_values[...,2] = dz_dx2.values
    dz_dx = Vector(new_values, drank=1)

    z1 = (y + (EPS,0,0)).element_div(x)
    z0 = (y - (EPS,0,0)).element_div(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,EPS,0)).element_div(x)
    z0 = (y - (0,EPS,0)).element_div(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,0,EPS)).element_div(x)
    z0 = (y - (0,0,EPS)).element_div(x)
    dz_dy2 = 0.5 * (z1 - z0) / EPS
    new_values = np.empty((N,3,3))
    new_values[...,0] = dz_dy0.values
    new_values[...,1] = dz_dy1.values
    new_values[...,2] = dz_dy2.values
    dz_dy = Vector(new_values, drank=1)
    dz_df = dz_dx.chain(x.d_df)
    dz_dg = dz_dy.chain(y.d_dg)
    dz_dh = dz_dx.chain(x.d_dh) + dz_dy.chain(y.d_dh)
    DEL = 1.e-3
    for i in range(N):
        for k in range(3):
            assert z.d_df.values[i,k] == dz_df.values[i,k] or abs(z.d_df.values[i,k] - dz_df.values[i,k]) <= max(1., abs(dz_df.values[i,k])) * DEL
            assert z.d_dg.values[i,k] == dz_dg.values[i,k] or abs(z.d_dg.values[i,k] - dz_dg.values[i,k]) <= max(1., abs(dz_dg.values[i,k])) * DEL
            assert z.d_dh.values[i,k] == dz_dh.values[i,k] or abs(z.d_dh.values[i,k] - dz_dh.values[i,k]) <= max(1., abs(dz_dh.values[i,k])) * DEL

    N = 300
    x = Vector(np.random.randn(N,3))
    y = Vector(np.random.randn(N,3))
    x.insert_deriv('f', Vector(np.random.randn(N,3,2), drank=1))
    x.insert_deriv('h', Vector(np.random.randn(N,3,2), drank=1))
    y.insert_deriv('g', Vector(np.random.randn(N,3,2), drank=1))
    y.insert_deriv('h', Vector(np.random.randn(N,3,2), drank=1))
    z = y.element_div(x)
    assert 'f' in x.derivs
    assert hasattr(x, 'd_df')
    assert 'g' not in x.derivs
    assert not hasattr(x, 'd_dg')
    assert 'h' in x.derivs
    assert hasattr(x, 'd_dh')
    assert 'f' not in y.derivs
    assert not hasattr(y, 'd_df')
    assert 'g' in y.derivs
    assert hasattr(y, 'd_dg')
    assert 'h' in y.derivs
    assert hasattr(y, 'd_dh')
    assert 'f' in z.derivs
    assert hasattr(z, 'd_df')
    assert 'g' in z.derivs
    assert hasattr(z, 'd_dg')
    assert 'h' in z.derivs
    assert hasattr(z, 'd_dh')
    EPS = 1.e-6
    z1 = y.element_div(x + (EPS,0,0))
    z0 = y.element_div(x - (EPS,0,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS
    z1 = y.element_div(x + (0,EPS,0))
    z0 = y.element_div(x - (0,EPS,0))
    dz_dx1 = 0.5 * (z1 - z0) / EPS
    z1 = y.element_div(x + (0,0,EPS))
    z0 = y.element_div(x - (0,0,EPS))
    dz_dx2 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (EPS,0,0)).element_div(x)
    z0 = (y - (EPS,0,0)).element_div(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,EPS,0)).element_div(x)
    z0 = (y - (0,EPS,0)).element_div(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,0,EPS)).element_div(x)
    z0 = (y - (0,0,EPS)).element_div(x)
    dz_dy2 = 0.5 * (z1 - z0) / EPS
    dz_df0 = (dz_dx0 * x.d_df.values[:,0,0] +
              dz_dx1 * x.d_df.values[:,1,0] +
              dz_dx2 * x.d_df.values[:,2,0])
    dz_df1 = (dz_dx0 * x.d_df.values[:,0,1] +
              dz_dx1 * x.d_df.values[:,1,1] +
              dz_dx2 * x.d_df.values[:,2,1])
    dz_dg0 = (dz_dy0 * y.d_dg.values[:,0,0] +
              dz_dy1 * y.d_dg.values[:,1,0] +
              dz_dy2 * y.d_dg.values[:,2,0])
    dz_dg1 = (dz_dy0 * y.d_dg.values[:,0,1] +
              dz_dy1 * y.d_dg.values[:,1,1] +
              dz_dy2 * y.d_dg.values[:,2,1])
    dz_dh0 = (dz_dx0 * x.d_dh.values[:,0,0] + dz_dy0 * y.d_dh.values[:,0,0] +
              dz_dx1 * x.d_dh.values[:,1,0] + dz_dy1 * y.d_dh.values[:,1,0] +
              dz_dx2 * x.d_dh.values[:,2,0] + dz_dy2 * y.d_dh.values[:,2,0])
    dz_dh1 = (dz_dx0 * x.d_dh.values[:,0,1] + dz_dy0 * y.d_dh.values[:,0,1] +
              dz_dx1 * x.d_dh.values[:,1,1] + dz_dy1 * y.d_dh.values[:,1,1] +
              dz_dx2 * x.d_dh.values[:,2,1] + dz_dy2 * y.d_dh.values[:,2,1])
    DEL = 1.e-3
    for i in range(N):
        for k in range(3):
            assert z.d_df.values[i,k,0] == dz_df0.values[i,k] or abs(z.d_df.values[i,k,0] - dz_df0.values[i,k]) <= max(1., abs(dz_df0.values[i,k])) * DEL
            assert z.d_dg.values[i,k,0] == dz_dg0.values[i,k] or abs(z.d_dg.values[i,k,0] - dz_dg0.values[i,k]) <= max(1., abs(dz_dg0.values[i,k])) * DEL
            assert z.d_dh.values[i,k,0] == dz_dh0.values[i,k] or abs(z.d_dh.values[i,k,0] - dz_dh0.values[i,k]) <= max(1., abs(dz_dh0.values[i,k])) * DEL

            assert z.d_df.values[i,k,1] == dz_df1.values[i,k] or abs(z.d_df.values[i,k,1] - dz_df1.values[i,k]) <= max(1., abs(dz_df1.values[i,k])) * DEL
            assert z.d_dg.values[i,k,1] == dz_dg1.values[i,k] or abs(z.d_dg.values[i,k,1] - dz_dg1.values[i,k]) <= max(1., abs(dz_dg1.values[i,k])) * DEL
            assert z.d_dh.values[i,k,1] == dz_dh1.values[i,k] or abs(z.d_dh.values[i,k,1] - dz_dh1.values[i,k]) <= max(1., abs(dz_dh1.values[i,k])) * DEL

    assert y.element_div(x, recursive=False).derivs == {}
    assert hasattr(x, 'd_df')
    assert hasattr(x, 'd_dh')
    assert hasattr(y, 'd_dg')
    assert hasattr(y, 'd_dh')
    assert not hasattr(y.element_div(x, recursive=False), 'd_df')
    assert not hasattr(y.element_div(x, recursive=False), 'd_dg')
    assert not hasattr(y.element_div(x, recursive=False), 'd_dh')

    N = 10
    y = Vector(np.random.randn(N*3).reshape(N,3))
    x = Vector(np.random.randn(N*3).reshape(N,3))
    assert not x.readonly
    assert not y.readonly
    assert not y.element_div(x).readonly
    assert x.as_readonly().readonly
    assert y.as_readonly().readonly
    assert not y.as_readonly().element_div(x.as_readonly()).readonly
    assert not y.as_readonly().element_div(x).readonly
    assert not y.element_div(x.as_readonly()).readonly

    x = Vector(np.arange(9).reshape(3,3))
    y = Vector(np.arange(4))
    with pytest.raises(ValueError) as cm:
        x.element_div(y)
    assert str(cm.value) == ('incompatible numerator shapes for '
                                        'Vector.element_div(): (3,), (4,)')
    x = Vector3(np.arange(18).reshape(3,3,2), drank=1)
    y = Vector3(np.arange(1,19).reshape(3,3,2), drank=1)
    with pytest.raises(ValueError) as cm:
        x.element_div(y)
    assert str(cm.value) == ('Vector3.element_div() operand cannot have a '
                                        'denominator')


def test_vector_element_div_vector_with_derivs_vector_without_derivs() -> None:
    """Vector with derivs / Vector without derivs."""

    np.random.seed(1472)

    x = Vector3(np.arange(18).reshape(3,3,2), drank=1)
    y = Vector((1,1,1))
    ratio = x.element_div(y)
    assert ratio == x
    assert type(ratio) is Vector3
    x = Vector3(np.arange(18).reshape(3,3,2), drank=1)
    y = Vector((0,0,0))
    ratio = x.element_div(y)
    assert np.all(ratio.mask)
    assert type(ratio) is Vector3


def test_vector_element_div_derivative_unit_is_the_inverse_square() -> None:
    """The derivative of a quotient carries the divisor's unit to the inverse square."""

    a = Vector(np.ones(3), unit=Unit.KM)
    b = Vector(np.full(3, 2.), unit=Unit.S)
    b.insert_deriv('t', Vector(np.ones(3)))

    result = a.element_div(b)
    assert str(result.unit_) == 'km/s'
    assert str(result.derivs['t'].unit_) == 'km/s**2'


##########################################################################################
