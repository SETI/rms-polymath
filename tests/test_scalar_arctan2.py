##########################################################################################
# tests/test_scalar_arctan2.py
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar, Unit


def test_scalar_arctan2_individual_values() -> None:
    """Individual values."""

    np.random.seed(3622)

    assert Scalar(1.).arctan2(1.) == np.arctan2(1,1)
    assert type(Scalar(1.).arctan2(1.)) == Scalar
    assert Scalar(0.).arctan2(0.) == np.arctan2(0,0)
    assert Scalar(0.).arctan2(1.) == 0.
    assert Scalar( 1.).arctan2( 1.) == 0.25 * np.pi or abs(Scalar( 1.).arctan2( 1.) - 0.25 * np.pi) <= 1.e-15
    assert Scalar( 1.).arctan2( 0.) == 0.5  * np.pi or abs(Scalar( 1.).arctan2( 0.) - 0.5  * np.pi) <= 1.e-15
    assert Scalar( 1.).arctan2(-1.) == 0.75 * np.pi or abs(Scalar( 1.).arctan2(-1.) - 0.75 * np.pi) <= 1.e-15
    assert Scalar( 0.).arctan2(-1.) == np.pi or abs(Scalar( 0.).arctan2(-1.) - np.pi) <= 1.e-15
    assert Scalar(-1.).arctan2(-1.) == -0.75 * np.pi or abs(Scalar(-1.).arctan2(-1.) - -0.75 * np.pi) <= 1.e-15
    assert Scalar(-1.).arctan2( 0.) == -0.5  * np.pi or abs(Scalar(-1.).arctan2( 0.) - -0.5  * np.pi) <= 1.e-15
    assert Scalar(-1.).arctan2( 1.) == -0.25 * np.pi or abs(Scalar(-1.).arctan2( 1.) - -0.25 * np.pi) <= 1.e-15

    assert (abs(4/np.pi * Scalar(1.).arctan2((1,0,-1)) -
                     (1,2,3)).max() < 1.e-15)
    assert (abs(4/np.pi * Scalar(-1.).arctan2((1,0,-1)) -
                     (-1,-2,-3)).max() < 1.e-15)
    assert (abs(4/np.pi * Scalar((1,0,-1)).arctan2((1,0,-1)) -
                     (1,0,-3)).max() < 1.e-15)
    assert (abs(4/np.pi * Scalar((1,0,-1)).arctan2((1.,)) -
                     (1,0,-1)).max() < 1.e-15)

    N = 1000
    y = Scalar(np.random.randn(N))
    x = Scalar(np.random.randn(N))
    angle = y.arctan2(x)
    for i in range(N):
        assert angle[i] == np.arctan2(y.values[i], x.values[i])
    for i in range(N-1):
        assert angle[i:i+2] == (np.arctan2(y.values[i:i+2],
                                                  x.values[i:i+2]))

    values = np.random.randn(10)
    y = Scalar(values, unit=Unit.KM)
    x = Scalar(values, unit=Unit.CM)
    assert not np.any(y.arctan2(x).mask)
    values = np.random.randn(10)
    y = Scalar(values, unit=Unit.KM)
    x = Scalar(values, unit=None)
    assert not np.any(y.arctan2(x).mask)
    values = np.random.randn(10)
    y = Scalar(values, unit=Unit.KM)
    x = Scalar(values, unit=Unit.SECONDS)
    with pytest.raises(ValueError):
        y.arctan2(x)
    values = np.random.randn(10)
    y = Scalar(values, unit=Unit.KM)
    x = Scalar(values, unit=Unit.UNITLESS)
    with pytest.raises(ValueError):
        y.arctan2(x)


def test_scalar_arctan2_units_should_be_removed() -> None:
    """Units should be removed."""

    np.random.seed(3622)

    values = np.random.randn(10)
    y = Scalar(values, unit=Unit.KM)
    x = Scalar(values, unit=Unit.CM)
    assert (y.arctan2(x).units is None)


def test_scalar_arctan2_units_should_be_removed_2() -> None:
    """Units should be removed."""

    np.random.seed(3622)

    N = 100
    y = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))
    z = y.arctan2(x)
    assert np.all(z.mask[x.mask])
    assert np.all(z.mask[y.mask])
    assert not np.any(z.mask[~x.mask & ~y.mask])


def test_scalar_arctan2_derivatives() -> None:
    """Derivatives."""

    np.random.seed(3622)

    N = 20
    y = Scalar(np.random.randn(N))
    x = Scalar(np.random.randn(N))
    x.insert_deriv('f', Scalar(np.random.randn(N)))
    x.insert_deriv('h', Scalar(np.random.randn(N)))
    y.insert_deriv('g', Scalar(np.random.randn(N)))
    y.insert_deriv('h', Scalar(np.random.randn(N)))
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
    assert 'f' in y.arctan2(x).derivs
    assert hasattr(y.arctan2(x), 'd_df')
    assert 'g' in y.arctan2(x).derivs
    assert hasattr(y.arctan2(x), 'd_dg')
    assert 'h' in y.arctan2(x).derivs
    assert hasattr(y.arctan2(x), 'd_dh')
    EPS = 1.e-6
    z1 = y.arctan2(x + EPS)
    z0 = y.arctan2(x - EPS)
    dz_dx = 0.5 * (z1 - z0) / EPS
    z1 = (y + EPS).arctan2(x)
    z0 = (y - EPS).arctan2(x)
    dz_dy = 0.5 * (z1 - z0) / EPS
    z = y.arctan2(x)
    for i in range(N):
        assert dz_dx[i]*x.d_df[i] == z.d_df[i] or abs(dz_dx[i]*x.d_df[i] - z.d_df[i]) <= EPS
        assert dz_dy[i]*y.d_dg[i] == z.d_dg[i] or abs(dz_dy[i]*y.d_dg[i] - z.d_dg[i]) <= EPS
        assert dz_dx[i]*x.d_dh[i] + dz_dy[i]*y.d_dh[i] == z.d_dh[i] or abs(dz_dx[i]*x.d_dh[i] + dz_dy[i]*y.d_dh[i] - z.d_dh[i]) <= EPS

    assert y.arctan2(x, recursive=False).derivs == {}
    assert hasattr(x, 'd_df')
    assert hasattr(x, 'd_dh')
    assert hasattr(y, 'd_dg')
    assert hasattr(y, 'd_dh')
    assert not hasattr(y.arctan2(x, recursive=False), 'd_df')
    assert not hasattr(y.arctan2(x, recursive=False), 'd_dg')
    assert not hasattr(y.arctan2(x, recursive=False), 'd_dh')


def test_scalar_arctan2_read_only_status_should_be_preserved() -> None:
    """Read-only status should be preserved."""

    np.random.seed(3622)

    N = 10
    y = Scalar(np.random.randn(N))
    x = Scalar(np.random.randn(N))
    assert not x.readonly
    assert not y.readonly
    assert not y.arctan2(x).readonly
    assert x.as_readonly().readonly
    assert y.as_readonly().readonly
    assert not y.as_readonly().arctan2(x.as_readonly()).readonly
    assert not y.as_readonly().arctan2(x).readonly
    assert not y.arctan2(x.as_readonly()).readonly


##########################################################################################
