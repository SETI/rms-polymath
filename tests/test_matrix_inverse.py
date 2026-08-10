##########################################################################################
# tests/test_matrix_inverse.py
##########################################################################################

import numpy as np
import pytest

from polymath import Matrix, Unit


def test_matrix_inverse_make_sure_3x3_matrix_inversion_is_successful() -> None:
    """Make sure 3x3 matrix inversion is successful."""

    np.random.seed(9893)
    DEL = 2.e-11

    a = Matrix(np.random.randn(3,3))
    b =a.inverse()
    axb = a * b
    bxa = b * a
    for j in range(3):
        for k in range(3):
            assert axb.values[j,k] == int(j==k) or abs(axb.values[j,k] - int(j==k)) <= DEL
            assert bxa.values[j,k] == int(j==k) or abs(bxa.values[j,k] - int(j==k)) <= DEL
    N = 30
    a = Matrix(np.random.randn(N,3,3))
    b = a.inverse()
    assert not np.any(b.mask)
    axb = a * b
    bxa = b * a
    for i in range(N):
        for j in range(3):
            for k in range(3):
                assert axb.values[i,j,k] == int(j==k) or abs(axb.values[i,j,k] - int(j==k)) <= DEL

    a = Matrix(np.random.randn(2,2))
    b = a.inverse()
    axb = a * b
    bxa = b * a
    for j in range(2):
        for k in range(2):
            assert axb.values[j,k] == int(j==k) or abs(axb.values[j,k] - int(j==k)) <= DEL
            assert bxa.values[j,k] == int(j==k) or abs(bxa.values[j,k] - int(j==k)) <= DEL
    N = 30
    a = Matrix(np.random.randn(N,2,2))
    b = a.inverse()
    assert not np.any(b.mask)
    axb = a * b
    bxa = b * a
    for i in range(N):
        for j in range(2):
            for k in range(2):
                assert axb.values[i,j,k] == int(j==k) or abs(axb.values[i,j,k] - int(j==k)) <= DEL
                assert bxa.values[i,j,k] == int(j==k) or abs(bxa.values[i,j,k] - int(j==k)) <= DEL

    a = Matrix(np.random.randn(N,N,5,5))
    b = a.inverse()
    axb = a * b
    bxa = b * a
    for i0 in range(N):
        for i1 in range(N):
            for j in range(5):
                for k in range(5):
                    assert axb.values[i0,i1,j,k] == int(j==k) or abs(axb.values[i0,i1,j,k] - int(j==k)) <= DEL
                    assert bxa.values[i0,i1,j,k] == int(j==k) or abs(bxa.values[i0,i1,j,k] - int(j==k)) <= DEL
    N = 30
    size = 5
    mats = np.random.randn(N,size,size)
    for i in range(N):
        for j in range(size):
            for k in range(size):
                if j != k:
                    mats[i,j,k] = 0.
    a = Matrix(mats)
    b = a.inverse()
    assert not np.any(b.mask)
    axb = a * b
    bxa = b * a
    for i in range(N):
        for j in range(size):
            for k in range(size):
                assert axb.values[i,j,k] == int(j==k) or abs(axb.values[i,j,k] - int(j==k)) <= DEL
                assert bxa.values[i,j,k] == int(j==k) or abs(bxa.values[i,j,k] - int(j==k)) <= DEL

    N = 30
    values = np.random.randn(N,3,3)
    values[0,0,0] = 0.
    values[0,0,1] = 0.
    values[0,0,2] = 0.
    a = Matrix(values)
    b = a.inverse()
    axb = a * b
    bxa = b * a
    assert b.mask[0]
    assert not np.any(b.mask[1:])
    for i in range(1,N):
        for j in range(3):
            for k in range(3):
                assert axb.values[i,j,k] == int(j==k) or abs(axb.values[i,j,k] - int(j==k)) <= DEL
                assert bxa.values[i,j,k] == int(j==k) or abs(bxa.values[i,j,k] - int(j==k)) <= DEL

    N = 30
    size = 5
    values = np.random.randn(N,size,size)
    for j in range(size):
        for k in range(size):
            if j != k:
                values[0,j,k] = 0.
    values[0,0,0] = 0.
    a = Matrix(values)
    b = a.inverse()
    axb = a * b
    bxa = b * a
    assert b.mask[0]
    assert not np.any(b.mask[1:])
    for i in range(1,N):
        for j in range(5):
            for k in range(5):
                assert axb.values[i,j,k] == int(j==k) or abs(axb.values[i,j,k] - int(j==k)) <= DEL
                assert bxa.values[i,j,k] == int(j==k) or abs(bxa.values[i,j,k] - int(j==k)) <= DEL

    a = Matrix(np.random.randn(N,3,4))
    with pytest.raises(ValueError):
        a.inverse()
    a = Matrix(np.random.randn(N,3,3,2,4), drank=2)
    with pytest.raises(ValueError):
        a.inverse()


def test_matrix_inverse_test_unit() -> None:
    """Test unit."""

    np.random.seed(9893)

    N = 5
    a = Matrix(np.random.randn(N,3,3), unit=Unit.CM**2/Unit.S)
    b = a.inverse()
    assert b.units == Unit.S/Unit.CM**2


def test_matrix_inverse_derivatives_3x3() -> None:
    """Derivatives, 3x3."""

    np.random.seed(9893)
    DEL = 2.e-11

    N = 30
    a = Matrix(np.random.randn(N,3,3))
    a.insert_deriv('t', Matrix(np.random.randn(N,3,3)))
    a.insert_deriv('v', Matrix(np.random.randn(N,3,3,2), drank=1))
    assert 't' in a.derivs
    assert hasattr(a, 'd_dt')
    assert 'v' in a.derivs
    assert hasattr(a, 'd_dv')
    b = a.inverse(recursive=False)
    assert 't' not in b.derivs
    assert not hasattr(b, 'd_dt')
    assert 'v' not in b.derivs
    assert not hasattr(b, 'd_dv')
    b = a.inverse(recursive=True)
    assert 't' in b.derivs
    assert hasattr(b, 'd_dt')
    assert 'v' in b.derivs
    assert hasattr(b, 'd_dv')
    EPS = 1.e-6
    db_da_values = np.empty((N,3,3,3,3))
    for i in range(3):
        for j in range(3):
            da = np.zeros((3,3))
            da[i,j] = EPS
            b1 = (a + da).inverse()
            b0 = (a - da).inverse()
            db_da_values[...,i,j] = (0.5/EPS) * (b1 - b0).values
    db_da = Matrix(db_da_values, drank=2)
    db_dt = db_da.chain(a.d_dt)
    db_dv = db_da.chain(a.d_dv)
    tscale = np.sqrt(np.mean(np.mean(db_dt.values**2, axis=-1), axis=-1))
    vscale = np.sqrt(np.mean(np.mean(db_dv.values**2, axis=-2), axis=-2))
    DEL = 2.e-4
    for i in range(N):
        for j in range(3):
            for k in range(3):
                assert db_dt.values[i,j,k] == b.d_dt.values[i,j,k] or abs(db_dt.values[i,j,k] - b.d_dt.values[i,j,k]) <= DEL * max(1., tscale[i])
                assert db_dv.values[i,j,k,0] == b.d_dv.values[i,j,k,0] or abs(db_dv.values[i,j,k,0] - b.d_dv.values[i,j,k,0]) <= DEL * max(1., vscale[i,0])
                assert db_dv.values[i,j,k,1] == b.d_dv.values[i,j,k,1] or abs(db_dv.values[i,j,k,1] - b.d_dv.values[i,j,k,1]) <= DEL * max(1., vscale[i,1])


def test_matrix_inverse_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(9893)

    N = 10
    a = Matrix(np.random.randn(N,3,3))
    a.inverse()
    assert not a.readonly
    assert not a.inverse().readonly
    assert a.as_readonly().readonly
    assert not a.as_readonly().inverse().readonly


def test_matrix_inverse_leaves_a_singular_input_unmodified() -> None:
    """inverse() masks a singular matrix without altering the object it was called on."""

    a = Matrix([[[1., 0.], [0., 1.]], [[0., 0.], [0., 0.]]])
    saved = a.values.copy()
    result = a.inverse()
    assert np.all(a.values == saved)
    assert result.mask[1]
    assert not result.mask[0]


##########################################################################################
