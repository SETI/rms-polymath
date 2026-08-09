##########################################################################################
# tests/test_vector_int.py
##########################################################################################

import numpy as np
import unittest

from polymath import Pair, Scalar, Unit, Vector, Vector3


class Test_Vector_int(unittest.TestCase):

    def runTest(self):

        np.random.seed(5394)

        # int input
        a = Vector(np.arange(30).reshape(10,3))
        b = a.int()
        self.assertIs(a, b)

        a = Vector3(np.arange(30).reshape(10,3), unit=Unit.KM)
        with self.assertRaises(ValueError) as cm:
            b = a.int()
        self.assertEqual(str(cm.exception), 'Vector3.int() unit is not permitted: km')

        a = Pair(np.arange(60).reshape(10,2,3), drank=1)
        with self.assertRaises(ValueError) as cm:
            b = a.int()
        self.assertEqual(str(cm.exception), 'Pair.int() does not support denominators')

        a = Pair(np.arange(-40.,40.).reshape(-1,2)/10.)
        b = a.int()
        self.assertTrue(np.all(b.vals == np.floor(a.vals)))
        self.assertTrue(b.is_int())
        self.assertFalse(b.mask)

        a = Pair(np.arange(-40.,40.).reshape(-1,2)/10.)
        b = a.int(remask=True)
        self.assertTrue(np.all(b.vals == np.floor(a.vals)))
        self.assertTrue(b.is_int())
        self.assertTrue(np.all(b.vals[b.mask] < 0))
        self.assertTrue(np.all(b.vals[~b.mask] >= 0))

        # top = 2
        a = Pair(np.arange(-40.,40.).reshape(-1,2)/10.)
        b = a.int(top=(2,3))

        # TBD!

        ##################################################################################
        # Additional coverage tests
        ##################################################################################

        # Test int() with top=None and negative values, clip=True
        a = Vector([-1., 2., 3.])
        b = a.int(top=None, clip=True)
        self.assertEqual(b.values[0], 0)
        self.assertEqual(b.values[1], 2)
        self.assertEqual(b.values[2], 3)

        # Test vector_scale with recursive=False
        v = Vector([1., 0., 0.])
        factor = Vector([2., 0., 0.])
        result = v.vector_scale(factor, recursive=False)
        self.assertEqual(type(result), Vector)

        # Test combos with all int scalars
        s1 = Scalar([1, 2])
        s2 = Scalar([3, 4])
        v = Vector.combos(s1, s2)
        self.assertEqual(v.shape, (2, 2))
        self.assertEqual(v.numer, (2,))
        self.assertTrue(v.is_int())

##########################################################################################
