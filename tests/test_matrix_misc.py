##########################################################################################
# tests/test_matrix_misc.py
# Old Matrix tests, updated by MRS 2/19/14
##########################################################################################

import numpy as np
import pytest

from polymath import Matrix, Scalar, Vector


def test_matrix_misc_outer_multiply() -> None:
    """Outer multiply."""

    np.random.seed(6921)
    a = Vector((1,2))
    b = Vector((0,1,-1))

    ab = a.outer(b)
    assert ab == (Matrix([(0.,1.,-1.),
                                  (0.,2.,-2.)]))
    assert ab * Vector((3,2,1)) == Vector([1.,2.])
    assert (ab * Vector([(3,2,1),
                                  (1,2,0)])) == (Vector(([1.,2.],
                                                     [2.,4.])))
    v = Vector([(3,2,1),(1,2,0)])
    assert v.shape == (2,)
    assert v.item == (3,)
    assert v*2 == Vector([(6,4,2),(2,4,0)])
    assert v/2 == Vector([(1.5,1.,0.5),(0.5,1.,0.)])
    assert 2*v == 2.*v
    m = Matrix([(3,2,1),(1,2,0)])
    assert m.shape == ()
    assert m.item == (2,3)
    assert m*2 == Matrix([(6,4,2),(2,4,0)])
    assert m/2 == Matrix([(1.5,1.,0.5),(0.5,1.,0.)])
    assert 2*m == 2.*m
    i = Matrix([(-1,0,0),(0,2,0),(0,0,0)])
    assert m*i == Matrix([(-3,4,0),(-1,4,0)])
    assert i*v == Vector([(-3,4,0),(-1,4,0)])
    j = Matrix([(-1,0),(0,2),(1,1)])
    assert j*m == Matrix([(-3,-2,-1),(2,4,0),(4,4,1)])

    test = Matrix(np.random.rand(200,3,3))
    inverse = test.inverse()
    product = test * inverse
    DEL = 1.e-11
    assert np.all(abs(product.vals[...,0,0] - 1) < DEL)
    assert np.all(abs(product.vals[...,1,1] - 1) < DEL)
    assert np.all(abs(product.vals[...,2,2] - 1) < DEL)
    assert np.all(abs(product.vals[...,0,1]) < DEL)
    assert np.all(abs(product.vals[...,1,0]) < DEL)
    assert np.all(abs(product.vals[...,2,0]) < DEL)
    assert np.all(abs(product.vals[...,0,2]) < DEL)
    assert np.all(abs(product.vals[...,2,1]) < DEL)
    assert np.all(abs(product.vals[...,1,2]) < DEL)

    ##################################################################################
    # Additional coverage tests
    ##################################################################################

    v = Vector(np.random.randn(3, 2), drank=1)
    m = Matrix.as_matrix(v)
    assert type(m) == Matrix
    assert m.numer == (3, 2)

    v = Vector(np.random.randn(3, 2), drank=1)
    v.insert_deriv('t', Vector(np.random.randn(3, 2), drank=1))
    m = Matrix.as_matrix(v, recursive=False)
    assert not hasattr(m, 'd_dt')

    with pytest.raises(ValueError) as cm:
        Matrix.from_scalars(*[Scalar(float(i)) for i in range(5)])
    assert 'incorrect number of Scalars' in str(cm.value)

    original_debug = Matrix._DEBUG
    try:
        Matrix._DEBUG = True
        # Use array of matrices to ensure rms._values is an array
        m = Matrix(np.random.randn(2, 3, 3))
        m_unitary = m.unitary()
        assert type(m_unitary).__name__ == 'Matrix3'
    finally:
        Matrix._DEBUG = original_debug

    m = Matrix(np.random.randn(3, 3))
    m_unitary = m.unitary()
    assert type(m_unitary).__name__ == 'Matrix3'

    m = Matrix(np.random.randn(3, 3, 3))
    m = Matrix(m._values, mask=np.array([False, True, False]))
    m_unitary = m.unitary()
    assert type(m_unitary).__name__ == 'Matrix3'

    m = Matrix([[1., 2.], [3., 4.]])

    with pytest.raises((TypeError, AttributeError)):
        _ = m.__rfloordiv__(5)

    with pytest.raises((TypeError, AttributeError)):
        _ = m.__rmod__(5)


##########################################################################################
