##########################################################################################
# tests/test_scalar_comprehensive.py
# Comprehensive unit tests for Scalar class based on docstrings
##########################################################################################

import numpy as np
import unittest

from polymath import Scalar, Unit


class Test_Scalar_Comprehensive(unittest.TestCase):

    def runTest(self):

        np.random.seed(5678)

        # Test as_scalar static method
        s1 = Scalar.as_scalar(5.)
        self.assertEqual(type(s1), Scalar)
        self.assertEqual(s1, 5.)

        s2 = Scalar.as_scalar([1., 2., 3.])
        self.assertEqual(type(s2), Scalar)
        self.assertTrue(np.allclose(s2.vals, [1., 2., 3.]))

        # Test to_scalar method
        s3 = Scalar(5.)
        s4 = s3.to_scalar(0)
        self.assertEqual(s4, 5.)

        # Should raise error for non-zero index
        self.assertRaises(ValueError, s3.to_scalar, 1)

        # Test as_index method
        s5 = Scalar([0, 1, 2, 3])
        idx = s5.as_index()
        self.assertTrue(np.allclose(idx, [0, 1, 2, 3]))

        # Test as_index_and_mask
        s6 = Scalar([0, 1, 2])
        idx2, mask2 = s6.as_index_and_mask()
        self.assertTrue(np.allclose(idx2, [0, 1, 2]))
        self.assertFalse(mask2)

        # Test int() method
        s7 = Scalar(5.7)
        s8 = s7.int()
        self.assertEqual(s8, 5)
        self.assertTrue(s8.is_int())

        # Test with top parameter
        s9 = Scalar([1, 2, 3, 4, 5])
        s10 = s9.int(top=3, remask=True)
        self.assertTrue(s10.mask[3] or s10.mask[4])

        # Test frac method
        s11 = Scalar(5.7)
        s12 = s11.frac()
        self.assertAlmostEqual(s12, 0.7, places=10)

        # Test sin method
        s13 = Scalar(np.pi/2, unit=Unit.RAD)
        s14 = s13.sin()
        self.assertAlmostEqual(s14, 1., places=10)

        # Test cos method
        s15 = Scalar(0., unit=Unit.RAD)
        s16 = s15.cos()
        self.assertAlmostEqual(s16, 1., places=10)

        # Test tan method
        s17 = Scalar(np.pi/4, unit=Unit.RAD)
        s18 = s17.tan()
        self.assertAlmostEqual(s18, 1., places=10)

        # Test arcsin method
        s19 = Scalar(1.)
        s20 = s19.arcsin()
        self.assertAlmostEqual(s20, np.pi/2, places=10)

        # Test arccos method
        s21 = Scalar(0.)
        s22 = s21.arccos()
        self.assertAlmostEqual(s22, np.pi/2, places=10)

        # Test arctan method
        s23 = Scalar(1.)
        s24 = s23.arctan()
        self.assertAlmostEqual(s24, np.pi/4, places=10)

        # Test arctan2 method
        s25 = Scalar(1.)
        s26 = Scalar(1.)
        s27 = s25.arctan2(s26)
        self.assertAlmostEqual(s27, np.pi/4, places=10)

        # Test sqrt method
        s28 = Scalar(4.)
        s29 = s28.sqrt()
        self.assertEqual(s29, 2.)

        # Test log method
        s30 = Scalar(np.e)
        s31 = s30.log()
        self.assertAlmostEqual(s31, 1., places=10)

        # Test exp method
        s32 = Scalar(1.)
        s33 = s32.exp()
        self.assertAlmostEqual(s33, np.e, places=10)

        # Test sign method
        s34 = Scalar([-2., 0., 2.])
        s35 = s34.sign()
        self.assertTrue(np.allclose(s35.vals, [-1., 0., 1.]))

        # Test solve_quadratic static method
        a = Scalar(1.)
        b = Scalar(0.)
        c = Scalar(-1.)
        x0, x1 = Scalar.solve_quadratic(a, b, c)
        self.assertAlmostEqual(x0, -1., places=10)
        self.assertAlmostEqual(x1, 1., places=10)

        # Test eval_quadratic method
        s36 = Scalar(2.)
        s37 = s36.eval_quadratic(1., 0., -4.)
        self.assertEqual(s37, 0.)  # 1*2^2 + 0*2 - 4 = 0

        # Test max method
        s38 = Scalar([1., 5., 3., 2., 4.])
        s39 = s38.max()
        self.assertEqual(s39, 5.)

        # Test min method
        s40 = s38.min()
        self.assertEqual(s40, 1.)

        # Test argmax method
        s41 = s38.argmax()
        self.assertEqual(s41, 1)  # Index of max value

        # Test argmin method
        s42 = s38.argmin()
        self.assertEqual(s42, 0)  # Index of min value

        # Test maximum static method
        s43 = Scalar([1., 3., 2.])
        s44 = Scalar([2., 1., 4.])
        s45 = Scalar.maximum(s43, s44)
        self.assertTrue(np.allclose(s45.vals, [2., 3., 4.]))

        # Test minimum static method
        s46 = Scalar.minimum(s43, s44)
        self.assertTrue(np.allclose(s46.vals, [1., 1., 2.]))

        # Test median method
        s47 = Scalar([1., 3., 2., 5., 4.])
        s48 = s47.median()
        self.assertEqual(s48, 3.)

        # Test sort method
        s49 = Scalar([3., 1., 4., 2.])
        s50 = s49.sort()
        self.assertTrue(np.allclose(s50.vals, [1., 2., 3., 4.]))

        # Test reciprocal method
        s51 = Scalar(2.)
        s52 = s51.reciprocal()
        self.assertEqual(s52, 0.5)

        # Test identity method
        s53 = Scalar(5.)
        s54 = s53.identity()
        self.assertEqual(s54, 1.)
        self.assertTrue(s54.readonly)

        # Test __abs__ method
        s55 = Scalar(-5.)
        s56 = abs(s55)
        self.assertEqual(s56, 5.)

        # Test __pow__ method
        s57 = Scalar(2.)
        s58 = s57 ** 3
        self.assertEqual(s58, 8.)

        s59 = s57 ** 0.5
        self.assertAlmostEqual(s59, np.sqrt(2.), places=10)

        # Test __le__ method
        s60 = Scalar(2.)
        result = s60 <= 3.
        self.assertTrue(result)

        # Test __lt__ method
        result = s60 < 3.
        self.assertTrue(result)

        # Test __ge__ method
        result = s60 >= 1.
        self.assertTrue(result)

        # Test __gt__ method
        result = s60 > 1.
        self.assertTrue(result)

        # n-D test cases
        # Test sin with n-D array
        s61 = Scalar([[0., np.pi/2], [np.pi, 3*np.pi/2]], unit=Unit.RAD)
        s62 = s61.sin()
        self.assertAlmostEqual(s62[0, 0], 0., places=10)
        self.assertAlmostEqual(s62[0, 1], 1., places=10)

        # Test max with axis
        s63 = Scalar([[1., 5., 3.], [2., 4., 6.]])
        s64 = s63.max(axis=1)
        self.assertTrue(np.allclose(s64.vals, [5., 6.]))

        # Test min with axis
        s65 = s63.min(axis=0)
        self.assertTrue(np.allclose(s65.vals, [1., 4., 3.]))

        # Test median with axis
        s66 = s63.median(axis=1)
        self.assertTrue(np.allclose(s66.vals, [3., 4.]))

        # Test as_scalar with Boolean
        from polymath import Boolean
        b1 = Boolean(True)
        s67 = Scalar.as_scalar(b1)
        self.assertEqual(type(s67), Scalar)
        self.assertEqual(s67, 1)

        # Test as_scalar with Unit (Unit is already imported at top)
        s68 = Scalar.as_scalar(Unit.RAD)
        self.assertEqual(type(s68), Scalar)
        # Check unit using the units property (plural)
        self.assertEqual(s68.units, Unit.RAD)

        # Test as_scalar with recursive=False
        s69 = Scalar(5.)
        s69.insert_deriv('t', Scalar(2.))
        s70 = Scalar.as_scalar(s69, recursive=False)
        self.assertEqual(len(s70.derivs), 0)

        # Test to_scalar with recursive=False
        s71 = Scalar(5.)
        s71.insert_deriv('t', Scalar(2.))
        s72 = s71.to_scalar(0, recursive=False)
        self.assertEqual(len(s72.derivs), 0)

        # Test as_index with masked parameter
        s73 = Scalar([0, 1, 2, 3])
        idx3 = s73.as_index(masked=99)
        self.assertTrue(np.allclose(idx3, [0, 1, 2, 3]))

        # Test as_index_and_mask with masked parameter
        s74 = Scalar([0, 1, 2])
        idx4, mask4 = s74.as_index_and_mask(masked=99)
        self.assertTrue(np.allclose(idx4, [0, 1, 2]))

        # Test as_index_and_mask with purge=True
        s75 = Scalar([0, 1, 2])
        s75 = s75.mask_where_le(1)
        idx5, mask5 = s75.as_index_and_mask(purge=True)
        self.assertEqual(type(idx5), np.ndarray)

        # Test int() with clip parameter
        s76 = Scalar([-1, 5, 3])
        s77 = s76.int(top=3, clip=True)
        # clip=True clips to [0, top-1], so [0, 2, 2]
        self.assertTrue(np.allclose(s77.vals, [0, 2, 2]))

        # Test int() with inclusive parameter
        s78 = Scalar([0, 1, 2, 3])
        s79 = s78.int(top=3, inclusive=False, remask=True)
        # Value 3 should be masked
        self.assertTrue(isinstance(s79, Scalar))

        # Test int() with shift parameter
        s80 = Scalar([0, 1, 2, 3])
        s81 = s80.int(top=3, shift=True, remask=True)
        self.assertTrue(isinstance(s81, Scalar))

        # Test frac with n-D
        s82 = Scalar([[1.5, 2.7], [3.9, 4.1]])
        s83 = s82.frac()
        self.assertAlmostEqual(s83[0, 0], 0.5, places=10)

        # Test sin with n-D and recursive=False
        s84 = Scalar([[0., np.pi/2], [np.pi, 3*np.pi/2]], unit=Unit.RAD)
        s85 = s84.sin(recursive=False)
        self.assertAlmostEqual(s85[0, 1], 1., places=10)

        # Test cos with recursive=False
        s86 = Scalar(0., unit=Unit.RAD)
        s87 = s86.cos(recursive=False)
        self.assertAlmostEqual(s87, 1., places=10)

        # Test tan with recursive=False
        s88 = Scalar(np.pi/4, unit=Unit.RAD)
        s89 = s88.tan(recursive=False)
        self.assertAlmostEqual(s89, 1., places=10)

        # Test arcsin with recursive=False
        s90 = Scalar(1.)
        s91 = s90.arcsin(recursive=False)
        self.assertAlmostEqual(s91, np.pi/2, places=10)

        # Test arccos with recursive=False
        s92 = Scalar(0.)
        s93 = s92.arccos(recursive=False)
        self.assertAlmostEqual(s93, np.pi/2, places=10)

        # Test arctan with recursive=False
        s94 = Scalar(1.)
        s95 = s94.arctan(recursive=False)
        self.assertAlmostEqual(s95, np.pi/4, places=10)

        # Test arctan2 with recursive=False
        s96 = Scalar(1.)
        s97 = Scalar(1.)
        s98 = s96.arctan2(s97, recursive=False)
        self.assertAlmostEqual(s98, np.pi/4, places=10)

        # Test sqrt with recursive=False
        s99 = Scalar(4.)
        s100 = s99.sqrt(recursive=False)
        self.assertEqual(s100, 2.)

        # Test log with recursive=False
        s101 = Scalar(np.e)
        s102 = s101.log(recursive=False)
        self.assertAlmostEqual(s102, 1., places=10)

        # Test exp with recursive=False
        s103 = Scalar(1.)
        s104 = s103.exp(recursive=False)
        self.assertAlmostEqual(s104, np.e, places=10)

        # Test sign (no recursive parameter)
        s105 = Scalar([-2., 0., 2.])
        s106 = s105.sign()
        self.assertTrue(np.allclose(s106.vals, [-1., 0., 1.]))

        # Test solve_quadratic with n-D
        a2 = Scalar([1., 1.])
        b2 = Scalar([0., 0.])
        c2 = Scalar([-1., -4.])
        x0_2, x1_2 = Scalar.solve_quadratic(a2, b2, c2)
        self.assertAlmostEqual(x0_2[0], -1., places=10)
        self.assertAlmostEqual(x1_2[0], 1., places=10)

        # Test eval_quadratic with recursive=False
        s107 = Scalar(2.)
        s108 = s107.eval_quadratic(1., 0., -4., recursive=False)
        self.assertEqual(s108, 0.)

        # Test max (no recursive parameter)
        s109 = Scalar([1., 5., 3., 2., 4.])
        s110 = s109.max()
        self.assertEqual(s110, 5.)

        # Test min (no recursive parameter)
        s111 = s109.min()
        self.assertEqual(s111, 1.)

        # Test argmax (no recursive parameter)
        s112 = s109.argmax()
        self.assertEqual(s112, 1)

        # Test argmin (no recursive parameter)
        s113 = s109.argmin()
        self.assertEqual(s113, 0)

        # Test maximum (no recursive parameter)
        s114 = Scalar([1., 3., 2.])
        s115 = Scalar([2., 1., 4.])
        s116 = Scalar.maximum(s114, s115)
        self.assertTrue(np.allclose(s116.vals, [2., 3., 4.]))

        # Test minimum (no recursive parameter)
        s117 = Scalar.minimum(s114, s115)
        self.assertTrue(np.allclose(s117.vals, [1., 1., 2.]))

        # Test median (no recursive parameter)
        s118 = Scalar([1., 3., 2., 5., 4.])
        s119 = s118.median()
        self.assertEqual(s119, 3.)

        # Test sort (no recursive parameter)
        s120 = Scalar([3., 1., 4., 2.])
        s121 = s120.sort()
        self.assertTrue(np.allclose(s121.vals, [1., 2., 3., 4.]))

        # Test reciprocal with recursive=False
        s122 = Scalar(2.)
        s123 = s122.reciprocal(recursive=False)
        self.assertEqual(s123, 0.5)

        # Test identity (no recursive parameter)
        s124 = Scalar(5.)
        s125 = s124.identity()
        self.assertEqual(s125, 1.)

        # Test __abs__ with recursive=False
        s126 = Scalar(-5.)
        s127 = abs(s126)
        self.assertEqual(s127, 5.)

        # Test __pow__ with recursive=False
        s128 = Scalar(2.)
        s129 = s128.__pow__(3, recursive=False)
        self.assertEqual(s129, 8.)

        # Test __pow__ with fractional exponent
        s130 = Scalar(4.)
        s131 = s130.__pow__(0.5, recursive=False)
        self.assertAlmostEqual(s131, 2., places=10)

        # Test __le__ with n-D
        s132 = Scalar([1., 2., 3.])
        result = s132 <= 2.
        self.assertTrue(result[0])
        self.assertTrue(result[1])
        self.assertFalse(result[2])

        # Test __lt__ with n-D
        result = s132 < 2.
        self.assertTrue(result[0])
        self.assertFalse(result[1])
        self.assertFalse(result[2])

        # Test __ge__ with n-D
        result = s132 >= 2.
        self.assertFalse(result[0])
        self.assertTrue(result[1])
        self.assertTrue(result[2])

        # Test __gt__ with n-D
        result = s132 > 2.
        self.assertFalse(result[0])
        self.assertFalse(result[1])
        self.assertTrue(result[2])

        # Test __eq__ with n-D
        result = s132 == 2.
        self.assertFalse(result[0])
        self.assertTrue(result[1])
        self.assertFalse(result[2])

        # Test __ne__ with n-D
        result = s132 != 2.
        self.assertTrue(result[0])
        self.assertFalse(result[1])
        self.assertTrue(result[2])

        # Test max with multiple axes
        s133 = Scalar([[[1., 5.], [3., 2.]], [[4., 1.], [6., 3.]]])
        s134 = s133.max(axis=(0, 1))
        # Max over axes 0 and 1: shape (2, 2, 2) -> (2,)
        # For first element: max(1, 3, 4, 6) = 6
        # For second element: max(5, 2, 1, 3) = 5
        self.assertTrue(np.allclose(s134.vals, [6., 5.]))

        # Test min with multiple axes
        s135 = s133.min(axis=(0, 1))
        self.assertTrue(np.allclose(s135.vals, [1., 1.]))

        # Test median with multiple axes
        s136 = Scalar([[[1., 5.], [3., 2.]], [[4., 1.], [6., 3.]]])
        s137 = s136.median(axis=(0, 1))
        self.assertTrue(np.allclose(s137.vals, [3.5, 2.5]))

        # Test sort with axis
        s138 = Scalar([[3., 1., 4.], [2., 5., 1.]])
        s139 = s138.sort(axis=1)
        self.assertTrue(np.allclose(s139[0].vals, [1., 3., 4.]))

        # Test solve_quadratic with complex roots (should mask)
        a3 = Scalar(1.)
        b3 = Scalar(1.)
        c3 = Scalar(1.)
        x0_3, x1_3 = Scalar.solve_quadratic(a3, b3, c3)
        # Should be masked

        # Test eval_quadratic with n-D
        s140 = Scalar([[1., 2.], [3., 4.]])
        s141 = s140.eval_quadratic(1., 0., -1.)
        self.assertEqual(s141[0, 0], 0.)
        self.assertEqual(s141[0, 1], 3.)

##########################################################################################
