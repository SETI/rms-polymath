##########################################################################################
# tests/test_vector_cross_2x2.py
##########################################################################################

import numpy as np

from polymath import Vector, Scalar, Unit


def test_vector_cross_2x2_this_calculation_has_a_small_probability_of_a_sizable_error() -> None:
    """This calculation has a small probability of a sizable error."""

    np.random.seed(8752)
    omega = Vector(np.random.randn(30,2))
    vec = Vector(np.random.randn(20,30,2))
    cross1 = omega.cross(vec)
    assert cross1.shape == (20,30)
    assert type(cross1) == Scalar
    assert cross1.numer == ()
    assert cross1.denom == ()
    cross1 = omega.unit().cross(vec.unit())
    cross2 = omega.unit().dot(vec.unit()).arccos().sin()

    diff = abs(abs(cross1) - cross2)
    assert np.all(diff.values < 1.e-10)


def test_vector_cross_2x2_test_units() -> None:
    """Test units."""

    np.random.seed(8752)
    omega = Vector(np.random.randn(30,2))
    vec = Vector(np.random.randn(20,30,2))
    cross1 = omega.cross(vec)
    assert cross1.shape == (20,30)
    assert type(cross1) == Scalar
    assert cross1.numer == ()
    assert cross1.denom == ()

    omega = Vector(np.random.randn(2), unit=Unit.KM)
    vec = Vector(np.random.randn(2), unit=Unit.SECONDS**(-1))
    cross = omega.cross(vec)
    assert cross.unit_ == Unit.KM/Unit.SECONDS


def test_vector_cross_2x2_derivatives_denom() -> None:
    """Derivatives, denom = ()."""

    np.random.seed(8752)
    omega = Vector(np.random.randn(30,2))
    vec = Vector(np.random.randn(20,30,2))
    cross1 = omega.cross(vec)
    assert cross1.shape == (20,30)
    assert type(cross1) == Scalar
    assert cross1.numer == ()
    assert cross1.denom == ()

    N = 10
    x = Vector(np.random.randn(N,2))
    y = Vector(np.random.randn(N,2))
    x.insert_deriv('f', Vector(np.random.randn(N,2)))
    x.insert_deriv('h', Vector(np.random.randn(N,2)))
    y.insert_deriv('g', Vector(np.random.randn(N,2)))
    y.insert_deriv('h', Vector(np.random.randn(N,2)))
    z = y.cross(x)
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
    z1 = y.cross(x + (EPS,0))
    z0 = y.cross(x - (EPS,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS
    z1 = y.cross(x + (0,EPS))
    z0 = y.cross(x - (0,EPS))
    dz_dx1 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (EPS,0)).cross(x)
    z0 = (y - (EPS,0)).cross(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,EPS)).cross(x)
    z0 = (y - (0,EPS)).cross(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS
    dz_df = (dz_dx0 * x.d_df.values[:,0] +
             dz_dx1 * x.d_df.values[:,1])
    dz_dg = (dz_dy0 * y.d_dg.values[:,0] +
             dz_dy1 * y.d_dg.values[:,1])
    dz_dh = (dz_dx0 * x.d_dh.values[:,0] + dz_dy0 * y.d_dh.values[:,0] +
             dz_dx1 * x.d_dh.values[:,1] + dz_dy1 * y.d_dh.values[:,1])
    for i in range(N):
        assert z.d_df.values[i] == dz_df.values[i] or abs(z.d_df.values[i] - dz_df.values[i]) <= EPS
        assert z.d_dg.values[i] == dz_dg.values[i] or abs(z.d_dg.values[i] - dz_dg.values[i]) <= EPS
        assert z.d_dh.values[i] == dz_dh.values[i] or abs(z.d_dh.values[i] - dz_dh.values[i]) <= EPS


def test_vector_cross_2x2_derivatives_denom_2() -> None:
    """Derivatives, denom = (2,)."""

    np.random.seed(8752)
    omega = Vector(np.random.randn(30,2))
    vec = Vector(np.random.randn(20,30,2))
    cross1 = omega.cross(vec)
    assert cross1.shape == (20,30)
    assert type(cross1) == Scalar
    assert cross1.numer == ()
    assert cross1.denom == ()

    N = 100
    x = Vector(np.random.randn(N,2))
    y = Vector(np.random.randn(N,2))
    x.insert_deriv('f', Vector(np.random.randn(N,2,2), drank=1))
    x.insert_deriv('h', Vector(np.random.randn(N,2,2), drank=1))
    y.insert_deriv('g', Vector(np.random.randn(N,2,2), drank=1))
    y.insert_deriv('h', Vector(np.random.randn(N,2,2), drank=1))
    z = y.cross(x)
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
    z1 = y.cross(x + (EPS,0))
    z0 = y.cross(x - (EPS,0))
    dz_dx0 = 0.5 * (z1 - z0) / EPS
    z1 = y.cross(x + (0,EPS))
    z0 = y.cross(x - (0,EPS))
    dz_dx1 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (EPS,0)).cross(x)
    z0 = (y - (EPS,0)).cross(x)
    dz_dy0 = 0.5 * (z1 - z0) / EPS
    z1 = (y + (0,EPS)).cross(x)
    z0 = (y - (0,EPS)).cross(x)
    dz_dy1 = 0.5 * (z1 - z0) / EPS
    dz_df0 = (dz_dx0 * x.d_df.values[:,0,0] +
              dz_dx1 * x.d_df.values[:,1,0])
    dz_df1 = (dz_dx0 * x.d_df.values[:,0,1] +
              dz_dx1 * x.d_df.values[:,1,1])
    dz_dg0 = (dz_dy0 * y.d_dg.values[:,0,0] +
              dz_dy1 * y.d_dg.values[:,1,0])
    dz_dg1 = (dz_dy0 * y.d_dg.values[:,0,1] +
              dz_dy1 * y.d_dg.values[:,1,1])
    dz_dh0 = (dz_dx0 * x.d_dh.values[:,0,0] + dz_dy0 * y.d_dh.values[:,0,0] +
              dz_dx1 * x.d_dh.values[:,1,0] + dz_dy1 * y.d_dh.values[:,1,0])
    dz_dh1 = (dz_dx0 * x.d_dh.values[:,0,1] + dz_dy0 * y.d_dh.values[:,0,1] +
              dz_dx1 * x.d_dh.values[:,1,1] + dz_dy1 * y.d_dh.values[:,1,1])
    for i in range(N):
        assert z.d_df.values[i,0] == dz_df0.values[i] or abs(z.d_df.values[i,0] - dz_df0.values[i]) <= EPS
        assert z.d_dg.values[i,0] == dz_dg0.values[i] or abs(z.d_dg.values[i,0] - dz_dg0.values[i]) <= EPS
        assert z.d_dh.values[i,0] == dz_dh0.values[i] or abs(z.d_dh.values[i,0] - dz_dh0.values[i]) <= EPS

        assert z.d_df.values[i,1] == dz_df1.values[i] or abs(z.d_df.values[i,1] - dz_df1.values[i]) <= EPS
        assert z.d_dg.values[i,1] == dz_dg1.values[i] or abs(z.d_dg.values[i,1] - dz_dg1.values[i]) <= EPS
        assert z.d_dh.values[i,1] == dz_dh1.values[i] or abs(z.d_dh.values[i,1] - dz_dh1.values[i]) <= EPS

    assert y.cross(x, recursive=False).derivs == {}
    assert hasattr(x, 'd_df')
    assert hasattr(x, 'd_dh')
    assert hasattr(y, 'd_dg')
    assert hasattr(y, 'd_dh')
    assert not hasattr(y.cross(x, recursive=False), 'd_df')
    assert not hasattr(y.cross(x, recursive=False), 'd_dg')
    assert not hasattr(y.cross(x, recursive=False), 'd_dh')


def test_vector_cross_2x2_read_only_status_should_be_preserved() -> None:
    """Read-only status should be preserved."""

    np.random.seed(8752)
    omega = Vector(np.random.randn(30,2))
    vec = Vector(np.random.randn(20,30,2))
    cross1 = omega.cross(vec)
    assert cross1.shape == (20,30)
    assert type(cross1) == Scalar
    assert cross1.numer == ()
    assert cross1.denom == ()

    N = 10
    y = Vector(np.random.randn(N,2))
    x = Vector(np.random.randn(N,2))
    assert not x.readonly
    assert not y.readonly
    assert not y.cross(x).readonly
    assert x.as_readonly().readonly
    assert y.as_readonly().readonly
    assert not y.as_readonly().cross(x.as_readonly()).readonly
    assert not y.as_readonly().cross(x).readonly
    assert not y.cross(x.as_readonly()).readonly


##########################################################################################
