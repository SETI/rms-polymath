##########################################################################################
# Matrix.column_vector() and Matrix.column_vectors()
##########################################################################################

import numpy as np

from polymath import Matrix, Vector, Vector3, Unit


def test_matrix_column_vectors_check_unit_and_masks() -> None:
    """check unit and masks."""

    np.random.seed(2897)
    N = 100
    a = Matrix(np.random.randn(N,7,1))
    b = a.column_vector(0)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert a.shape == b.shape
    assert a.values.shape == (N,7,1)
    assert b.values.shape == (N,7)
    assert type(b) == Vector
    c = a.column_vectors()
    assert np.all(a.values.ravel() == c[0].values.ravel())
    assert a.shape == c[0].shape
    assert b == c[0]
    assert type(c[0]) == Vector
    N = 100
    a = Matrix(np.random.randn(N,3,2))
    b = a.column_vector(0)
    assert a.shape == b.shape
    assert a.values.shape == (N,3,2)
    assert b.values.shape == (N,3)
    assert type(b) == Vector3
    assert type(a.column_vector(0, classes=Vector)) == Vector
    c = a.column_vectors()
    assert a.shape == c[0].shape
    assert b == c[0]
    assert type(c[0]) == Vector3

    N = 100
    a = Matrix(np.random.randn(N,4,4), mask=(np.random.randn(N) < -0.5),
               unit=Unit.RAD)
    c = a.column_vectors()
    assert a.unit_ == c[0].unit_
    b = a.column_vector(1)
    assert b == c[1]
    assert a.unit_ == b.unit_
    assert np.all(b.values == a.values[...,1])
    assert np.all(b.mask == a.mask)
    b[0].values[0] = 22.
    assert a[0].values[0,1] == 22.


def test_matrix_column_vectors_check_derivatives() -> None:
    """check derivatives."""

    np.random.seed(2897)
    N = 100
    a = Matrix(np.random.randn(N,7,1))
    b = a.column_vector(0)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert a.shape == b.shape
    assert a.values.shape == (N,7,1)
    assert b.values.shape == (N,7)
    assert type(b) == Vector
    c = a.column_vectors()
    assert np.all(a.values.ravel() == c[0].values.ravel())
    assert a.shape == c[0].shape
    assert b == c[0]
    assert type(c[0]) == Vector
    N = 100
    a = Matrix(np.random.randn(N,3,2))
    b = a.column_vector(0)
    assert a.shape == b.shape
    assert a.values.shape == (N,3,2)
    assert b.values.shape == (N,3)
    assert type(b) == Vector3
    assert type(a.column_vector(0, classes=Vector)) == Vector
    c = a.column_vectors()
    assert a.shape == c[0].shape
    assert b == c[0]
    assert type(c[0]) == Vector3

    N = 100
    a = Matrix(np.random.randn(N,3,4), mask=(np.random.randn(N) < -0.5))
    da_dt = Matrix(np.random.randn(N,3,4))
    da_dv = Matrix(np.random.randn(N,3,4,2), drank=1)
    a.insert_deriv('t', da_dt)
    a.insert_deriv('v', da_dv)
    assert hasattr(a, 'd_dt')
    assert hasattr(a, 'd_dv')
    b = a.column_vector(3, recursive=False)
    assert not hasattr(b, 'd_dt')
    assert not hasattr(b, 'd_dv')
    b = a.column_vector(3, recursive=True)
    assert hasattr(b, 'd_dt')
    assert hasattr(b, 'd_dv')
    assert b.d_dt.shape == a.shape
    assert b.d_dt.numer == (3,)
    assert b.d_dt.denom == ()
    assert b.d_dv.shape == a.shape
    assert b.d_dv.numer == (3,)
    assert b.d_dv.denom == (2,)
    assert np.all(a.values[...,3] == b.values)
    assert np.all(a.mask == b.mask)
    assert np.all(a.d_dt.values[...,3] == b.d_dt.values)
    assert np.all(a.d_dv.values[...,3,:] == b.d_dv.values)
    c = a.column_vectors(recursive=False)[3]
    assert not hasattr(c, 'd_dt')
    assert not hasattr(c, 'd_dv')
    c = a.column_vectors(recursive=True)[3]
    assert hasattr(c, 'd_dt')
    assert hasattr(c, 'd_dv')
    assert c.d_dt.shape == a.shape
    assert c.d_dt.numer == (3,)
    assert c.d_dt.denom == ()
    assert c.d_dv.shape == a.shape
    assert c.d_dv.numer == (3,)
    assert c.d_dv.denom == (2,)
    assert np.all(a.values[...,3] == c.values)
    assert np.all(a.mask == c.mask)
    assert np.all(a.d_dt.values[...,3] == c.d_dt.values)
    assert np.all(a.d_dv.values[...,3,:] == c.d_dv.values)


def test_matrix_column_vectors_read_only_status() -> None:
    """read-only status."""

    np.random.seed(2897)
    N = 100
    a = Matrix(np.random.randn(N,7,1))
    b = a.column_vector(0)
    assert np.all(a.values.ravel() == b.values.ravel())
    assert a.shape == b.shape
    assert a.values.shape == (N,7,1)
    assert b.values.shape == (N,7)
    assert type(b) == Vector
    c = a.column_vectors()
    assert np.all(a.values.ravel() == c[0].values.ravel())
    assert a.shape == c[0].shape
    assert b == c[0]
    assert type(c[0]) == Vector
    N = 100
    a = Matrix(np.random.randn(N,3,2))
    b = a.column_vector(0)
    assert a.shape == b.shape
    assert a.values.shape == (N,3,2)
    assert b.values.shape == (N,3)
    assert type(b) == Vector3
    assert type(a.column_vector(0, classes=Vector)) == Vector
    c = a.column_vectors()
    assert a.shape == c[0].shape
    assert b == c[0]
    assert type(c[0]) == Vector3

    N = 10
    a = Matrix(np.random.randn(N,4,4), mask=(np.random.randn(N) < -0.5))
    assert not a.readonly
    b = a.column_vector(3)
    assert not b.readonly
    c = a.column_vectors()[3]
    assert not c.readonly
    a = Matrix(np.random.randn(N,4,4), mask=(np.random.randn(N) < -0.5))
    a = a.as_readonly()
    assert a.readonly
    b = a.column_vector(3)
    assert b.readonly # preserved because of overlapping memory
    c = a.column_vectors()[3]
    assert c.readonly # preserved because of overlapping memory


##########################################################################################
