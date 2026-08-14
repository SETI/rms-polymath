##########################################################################################
# tests/test_vector_as_diagonal.py
##########################################################################################

import numpy as np

from polymath import Matrix, Vector, Scalar, Unit


def test_vector_as_diagonal_check_one_matrix() -> None:
    """Check one matrix."""

    np.random.seed(7098)

    a = Vector(np.arange(6))
    b = a.as_diagonal()
    for i in range(6):
        for j in range(6):
            if i == j:
                assert b.values[i,i] == a.values[i]
            else:
                assert b.values[i,j] == 0.


def test_vector_as_diagonal_check_an_array_of_matrices_some_masked() -> None:
    """Check an array of matrices, some masked."""

    np.random.seed(7098)

    a = Vector(np.random.randn(100,4), mask= np.random.rand(100) < -0.05)
    b = a.as_diagonal()
    for i in range(4):
        for j in range(4):
            aa = a.extract_numer(0, i, Scalar)
            bb = b.extract_numer(0, i, Vector).extract_numer(0, j, Scalar)

            if i == j:
                assert bb == aa
            else:
                assert bb == 0.
    assert np.all(a.mask == b.mask)


def test_vector_as_diagonal_test_unit() -> None:
    """Test unit."""

    np.random.seed(7098)

    a = Vector(np.random.randn(4), unit=Unit.KM)
    assert a.as_diagonal().unit_ == Unit.KM


def test_vector_as_diagonal_derivatives() -> None:
    """Derivatives."""

    np.random.seed(7098)

    N = 100
    x = Vector(np.random.randn(N,3))
    x.insert_deriv('t', Vector(np.random.randn(N,3)))
    x.insert_deriv('v', Vector(np.random.randn(N,3,2), drank=1))
    y = x.as_diagonal()
    assert 't' in x.derivs
    assert hasattr(x, 'd_dt')
    assert 'v' in x.derivs
    assert hasattr(x, 'd_dv')
    assert 't' in y.derivs
    assert hasattr(y, 'd_dt')
    assert 'v' in y.derivs
    assert hasattr(y, 'd_dv')
    EPS = 1.e-6
    y1 = (x + (EPS,0,0)).as_diagonal()
    y0 = (x - (EPS,0,0)).as_diagonal()
    dy_dx0 = 0.5 * (y1 - y0) / EPS
    y1 = (x + (0,EPS,0)).as_diagonal()
    y0 = (x - (0,EPS,0)).as_diagonal()
    dy_dx1 = 0.5 * (y1 - y0) / EPS
    y1 = (x + (0,0,EPS)).as_diagonal()
    y0 = (x - (0,0,EPS)).as_diagonal()
    dy_dx2 = 0.5 * (y1 - y0) / EPS
    new_values = np.empty((N,3,3,3))
    new_values[...,0] = dy_dx0.values
    new_values[...,1] = dy_dx1.values
    new_values[...,2] = dy_dx2.values
    dy_dx = Matrix(new_values, drank=1)
    dy_dt = dy_dx.chain(x.d_dt)
    dy_dv = dy_dx.chain(x.d_dv)
    DEL = 1.e-5
    for i in range(N):
        for j in range(3):
            for k in range(3):
                assert dy_dt.values[i,j,k] == y.d_dt.values[i,j,k] or abs(dy_dt.values[i,j,k] - y.d_dt.values[i,j,k]) <= DEL
                assert dy_dv.values[i,j,k,0] == y.d_dv.values[i,j,k,0] or abs(dy_dv.values[i,j,k,0] - y.d_dv.values[i,j,k,0]) <= DEL
                assert dy_dv.values[i,j,k,1] == y.d_dv.values[i,j,k,1] or abs(dy_dv.values[i,j,k,1] - y.d_dv.values[i,j,k,1]) <= DEL

    assert x.as_diagonal(recursive=False).derivs == {}
    assert hasattr(x, 'd_dt')
    assert hasattr(x, 'd_dv')
    assert not hasattr(x.as_diagonal(recursive=False), 'd_dt')
    assert not hasattr(x.as_diagonal(recursive=False), 'd_dv')


def test_vector_as_diagonal_read_only_status_should_not_be_preserved() -> None:
    """Read-only status should NOT be preserved."""

    np.random.seed(7098)

    N = 10
    x = Vector(np.random.randn(N,7))
    assert not x.readonly
    assert not x.as_diagonal().readonly
    assert not x.as_readonly().as_diagonal().readonly


##########################################################################################
