##########################################################################################
# tests/test_matrix_misc.py
# Old Matrix tests, updated by MRS 2/19/14
##########################################################################################

import numpy as np
import unittest

from polymath import Matrix, Scalar, Vector


class Test_Matrix_misc(unittest.TestCase):

    def runTest(self):

        np.random.seed(6921)

        a = Vector((1,2))
        b = Vector((0,1,-1))

        # Outer multiply
        ab = a.outer(b)

        self.assertEqual(ab, Matrix([(0.,1.,-1.),
                                      (0.,2.,-2.)]))

        self.assertEqual(ab * Vector((3,2,1)), Vector([1.,2.]))
        self.assertEqual(ab * Vector([(3,2,1),
                                      (1,2,0)]), Vector(([1.,2.],
                                                         [2.,4.])))

        v = Vector([(3,2,1),(1,2,0)])
        self.assertEqual(v.shape, (2,))
        self.assertEqual(v.item, (3,))
        self.assertEqual(v*2, Vector([(6,4,2),(2,4,0)]))
        self.assertEqual(v/2, Vector([(1.5,1.,0.5),(0.5,1.,0.)]))
        self.assertEqual(2*v, 2.*v)

        m = Matrix([(3,2,1),(1,2,0)])
        self.assertEqual(m.shape, ())
        self.assertEqual(m.item, (2,3))
        self.assertEqual(m*2, Matrix([(6,4,2),(2,4,0)]))
        self.assertEqual(m/2, Matrix([(1.5,1.,0.5),(0.5,1.,0.)]))
        self.assertEqual(2*m, 2.*m)

        i = Matrix([(-1,0,0),(0,2,0),(0,0,0)])
        self.assertEqual(m*i, Matrix([(-3,4,0),(-1,4,0)]))
        self.assertEqual(i*v, Vector([(-3,4,0),(-1,4,0)]))

        j = Matrix([(-1,0),(0,2),(1,1)])
        self.assertEqual(j*m, Matrix([(-3,-2,-1),(2,4,0),(4,4,1)]))

        # 3x3 Matrix inverse
        test = Matrix(np.random.rand(200,3,3))
        inverse = test.inverse()
        product = test * inverse

        DEL = 1.e-11
        self.assertTrue(np.all(abs(product.vals[...,0,0] - 1) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,1,1] - 1) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,2,2] - 1) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,0,1]) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,1,0]) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,2,0]) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,0,2]) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,2,1]) < DEL))
        self.assertTrue(np.all(abs(product.vals[...,1,2]) < DEL))

        ##################################################################################
        # Additional coverage tests
        ##################################################################################

        # Test as_matrix with Vector having drank=1
        v = Vector(np.random.randn(3, 2), drank=1)
        m = Matrix.as_matrix(v)
        self.assertEqual(type(m), Matrix)
        self.assertEqual(m.numer, (3, 2))

        # Test as_matrix with recursive=False
        v = Vector(np.random.randn(3, 2), drank=1)
        v.insert_deriv('t', Vector(np.random.randn(3, 2), drank=1))
        m = Matrix.as_matrix(v, recursive=False)
        self.assertFalse(hasattr(m, 'd_dt'))

        # Test from_scalars with non-square number of args
        with self.assertRaises(ValueError) as cm:
            Matrix.from_scalars(*[Scalar(float(i)) for i in range(5)])
        self.assertIn('incorrect number of Scalars', str(cm.exception))

        # Test unitary with _DEBUG=True
        original_debug = Matrix._DEBUG
        try:
            Matrix._DEBUG = True
            # Use array of matrices to ensure rms._values is an array
            m = Matrix(np.random.randn(2, 3, 3))
            m_unitary = m.unitary()
            self.assertEqual(type(m_unitary).__name__, 'Matrix3')
        finally:
            Matrix._DEBUG = original_debug

        # Test unitary with new_mask not any
        m = Matrix(np.random.randn(3, 3))
        m_unitary = m.unitary()
        self.assertEqual(type(m_unitary).__name__, 'Matrix3')

        # Test unitary with new_mask having some True and self._mask not False
        # Use array of matrices to have compatible mask shape
        m = Matrix(np.random.randn(3, 3, 3))
        m = Matrix(m._values, mask=np.array([False, True, False]))
        m_unitary = m.unitary()
        self.assertEqual(type(m_unitary).__name__, 'Matrix3')

        # Test __rfloordiv__ - this is called when int // Matrix
        m = Matrix([[1., 2.], [3., 4.]])
        # The error occurs inside _raise_unsupported_op, so we test the method directly
        with self.assertRaises((TypeError, AttributeError)):
            _ = m.__rfloordiv__(5)

        # Test __rmod__ - this is called when int % Matrix
        with self.assertRaises((TypeError, AttributeError)):
            _ = m.__rmod__(5)

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
##########################################################################################
