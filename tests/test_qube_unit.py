##########################################################################################
# tests/test_qube_unit.py
##########################################################################################

import numpy as np
import unittest

from polymath import Boolean, Matrix3, Quaternion, Scalar, Unit


class Test_Qube_unit(unittest.TestCase):

    def runTest(self):

        ##################################################################################
        # set_unit(self, unit, override=False)
        ##################################################################################

        a = Scalar((1.,2.,3.))
        self.assertEqual(a.units, None)
        self.assertTrue(np.all(a.values == (1,2,3)))

        a.set_unit(Unit.KM)
        self.assertEqual(a.units, Unit.KM)
        self.assertTrue(np.all(a.values == (1,2,3)))

        a.set_unit(Unit.CM)
        self.assertEqual(a.units, Unit.CM)
        self.assertTrue(np.all(a.values == (1,2,3)))

        self.assertRaises(ValueError, a.set_unit, Unit.DEG)   # incompatible

        a.set_unit(Unit.M)
        self.assertEqual(a.units, Unit.M)
        self.assertTrue(np.all(a.values == (1,2,3)))

        a = a.as_readonly()
        self.assertTrue(a.readonly)
        self.assertRaises(ValueError, a.set_unit, Unit.KM)

        a.set_unit(Unit.KM, override=True)
        self.assertTrue(a.readonly)
        self.assertEqual(a.units, Unit.KM)
        self.assertTrue(np.all(a.values == (1,2,3)))

        # Classes for which units are not allowed
        a = Matrix3([(1,0,0),(0,1,0),(0,0,1)])
        self.assertRaises(TypeError, a.set_unit, Unit.KM)

        a = Quaternion((1,0,0,0))
        self.assertRaises(TypeError, a.set_unit, Unit.KM)

        a = Boolean([True, False])
        self.assertRaises(TypeError, a.set_unit, Unit.KM)

        ##################################################################################
        # without_unit(self, recursive=True)
        ##################################################################################

        a = Scalar((1.,2.,3.), unit=Unit.KM)
        b = a.without_unit()
        self.assertEqual(a.units, Unit.KM)

        self.assertEqual(b.units, None)
        self.assertTrue(np.all(a.values == b.values))

        self.assertEqual(a.readonly, False)
        self.assertEqual(b.readonly, False)

        a = a.as_readonly()
        self.assertEqual(a.readonly, True)

        b = a.without_unit()
        self.assertEqual(b.readonly, True)
        self.assertEqual(b.units, None)
        self.assertTrue(np.all(b.values == (1,2,3)))

        ##################################################################################
        # into_unit(self, recursive=True)
        ##################################################################################

        a = Scalar((1.,2.,3.))
        self.assertEqual(a.units, None)
        self.assertTrue(np.all(a.values == (1,2,3)))

        a.set_unit(Unit.M)
        self.assertEqual(a.units, Unit.M)
        self.assertTrue(np.all(a.values == (1,2,3)))

        vals = a.into_unit()
        self.assertTrue(np.all(vals == (1000, 2000, 3000)))

        vals = a.into_unit(recursive=True)
        self.assertTrue(np.all(vals[0] == (1000, 2000, 3000)))
        self.assertTrue(vals[1] == {})

        a = Scalar((1.,2.,3.), unit=Unit.M)
        da_dt = Scalar((4., 5., 6.), unit=Unit.CM/Unit.S)
        a.insert_deriv('t', da_dt)

        vals = a.into_unit(recursive=False)
        self.assertTrue(np.all(vals == (1000, 2000, 3000)))

        vals = a.into_unit(recursive=True)
        self.assertTrue(np.all(vals[0] == (1000, 2000, 3000)))
        self.assertEqual(set(vals[1].keys()), {'t'})
        self.assertTrue(np.all(vals[1]['t'] == (400000, 500000, 600000)))

        # Test with n-D arrays
        a_nd = Scalar(np.random.rand(2, 3, 4), unit=Unit.M)
        vals = a_nd.into_unit()
        self.assertEqual(vals.shape, (2, 3, 4))
        expected = a_nd.values * 1000  # M to mm conversion
        self.assertTrue(np.allclose(vals, expected))

        # Test with unitless object
        a_unitless = Scalar((1., 2., 3.))
        vals = a_unitless.into_unit()
        self.assertTrue(np.all(vals == (1., 2., 3.)))

        # Test with unit that has factor == 1
        a_km = Scalar((1., 2., 3.), unit=Unit.KM)
        vals = a_km.into_unit()
        self.assertTrue(np.all(vals == (1., 2., 3.)))

        ##################################################################################
        # confirm_unit(self, unit)
        ##################################################################################

        # Test: Compatible units should not raise
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        result = a.confirm_unit(Unit.M)
        self.assertEqual(result, a)

        # Test: Same unit should not raise
        result = a.confirm_unit(Unit.KM)
        self.assertEqual(result, a)

        # Test: Unitless should be compatible with unitless
        a_unitless = Scalar((1., 2., 3.))
        result = a_unitless.confirm_unit(None)
        self.assertEqual(result, a_unitless)

        # Test: Incompatible units should raise ValueError
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        self.assertRaises(ValueError, a.confirm_unit, Unit.DEG)

        # Test: Unit with incompatible dimensions should raise
        a = Scalar((1., 2., 3.), unit=Unit.S)
        self.assertRaises(ValueError, a.confirm_unit, Unit.KM)

        # Test with n-D arrays
        a_nd = Scalar(np.random.rand(2, 3, 4), unit=Unit.M)
        result = a_nd.confirm_unit(Unit.CM)
        self.assertEqual(result, a_nd)

        # Test: None unit with unitless object
        a_unitless = Scalar((1., 2., 3.))
        result = a_unitless.confirm_unit(None)
        self.assertEqual(result, a_unitless)

        ##################################################################################
        # is_unitless(self)
        ##################################################################################

        # Test: Unitless object
        a = Scalar((1., 2., 3.))
        self.assertTrue(a.is_unitless())

        # Test: Object with unit
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        self.assertFalse(a.is_unitless())

        # Test: Object with angle unit
        a = Scalar((1., 2., 3.), unit=Unit.DEG)
        self.assertFalse(a.is_unitless())

        # Test: Object with time unit
        a = Scalar((1., 2., 3.), unit=Unit.S)
        self.assertFalse(a.is_unitless())

        # Test with n-D arrays
        a_nd = Scalar(np.random.rand(2, 3, 4))
        self.assertTrue(a_nd.is_unitless())

        a_nd = Scalar(np.random.rand(2, 3, 4), unit=Unit.M)
        self.assertFalse(a_nd.is_unitless())

        # Test: Setting unit to None makes it unitless
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        self.assertFalse(a.is_unitless())
        a.set_unit(None)
        self.assertTrue(a.is_unitless())

        # Test: without_unit makes it unitless
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        b = a.without_unit()
        self.assertTrue(b.is_unitless())

        ##################################################################################
        # Additional comprehensive tests for set_unit
        ##################################################################################

        # Test with n-D arrays
        a_nd = Scalar(np.random.rand(2, 3, 4))
        a_nd.set_unit(Unit.KM)
        self.assertEqual(a_nd.units, Unit.KM)
        self.assertEqual(a_nd.shape, (2, 3, 4))

        # Test setting unit to None
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        a.set_unit(None)
        self.assertEqual(a.units, None)
        self.assertTrue(a.is_unitless())

        # Test with compatible unit conversion
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        a.set_unit(Unit.M)
        self.assertEqual(a.units, Unit.M)
        # Values should remain the same (in standard units)
        self.assertTrue(np.all(a.values == (1., 2., 3.)))

        # Test with read-only object and override=False
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        a = a.as_readonly()
        self.assertRaises(ValueError, a.set_unit, Unit.M)

        # Test with read-only object and override=True
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        a = a.as_readonly()
        a.set_unit(Unit.M, override=True)
        self.assertEqual(a.units, Unit.M)

        ##################################################################################
        # Additional comprehensive tests for without_unit
        ##################################################################################

        # Test with n-D arrays
        a_nd = Scalar(np.random.rand(2, 3, 4), unit=Unit.KM)
        b_nd = a_nd.without_unit()
        self.assertEqual(b_nd.units, None)
        self.assertEqual(b_nd.shape, (2, 3, 4))
        self.assertTrue(np.all(a_nd.values == b_nd.values))

        # Test with recursive=False (should strip derivatives)
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        da_dt = Scalar((4., 5., 6.), unit=Unit.M/Unit.S)
        a.insert_deriv('t', da_dt)
        b = a.without_unit(recursive=False)
        self.assertEqual(b.units, None)
        self.assertEqual(len(b.derivs), 0)

        # Test with recursive=True (should keep derivatives and strip their units)
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        da_dt = Scalar((4., 5., 6.), unit=Unit.M/Unit.S)
        a.insert_deriv('t', da_dt)
        b = a.without_unit(recursive=True)
        self.assertEqual(b.units, None)
        self.assertEqual(len(b.derivs), 1)
        self.assertIn('t', b.derivs)
        # Derivatives should have their units stripped
        self.assertEqual(b.derivs['t'].units, None)

        # Test that original object is unchanged
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        b = a.without_unit()
        self.assertEqual(a.units, Unit.KM)
        self.assertEqual(b.units, None)

        # Test with read-only object
        a = Scalar((1., 2., 3.), unit=Unit.KM)
        a = a.as_readonly()
        b = a.without_unit()
        self.assertTrue(b.readonly)
        self.assertEqual(b.units, None)

        ##################################################################################
        # Additional comprehensive tests for into_unit
        ##################################################################################

        # Test with angle units
        # Values are in standard units (radians), into_unit converts to degrees
        a = Scalar(np.array([np.pi/2, np.pi, 3*np.pi/2]), unit=Unit.DEG)
        vals = a.into_unit()
        expected = np.array([90., 180., 270.])
        self.assertTrue(np.allclose(vals, expected))

        # Test with time units
        # Values are in standard units (seconds), into_unit converts to minutes
        a = Scalar(np.array([3600., 7200., 10800.]), unit=Unit.MIN)
        vals = a.into_unit()
        expected = np.array([60., 120., 180.])
        self.assertTrue(np.allclose(vals, expected))

        # Test with recursive=True and multiple derivatives
        a = Scalar((1., 2., 3.), unit=Unit.M)
        da_dt = Scalar((4., 5., 6.), unit=Unit.CM/Unit.S)
        da_dx = Scalar((7., 8., 9.), unit=Unit.M/Unit.KM)
        a.insert_deriv('t', da_dt)
        a.insert_deriv('x', da_dx)
        vals = a.into_unit(recursive=True)
        self.assertTrue(np.all(vals[0] == (1000, 2000, 3000)))
        self.assertEqual(set(vals[1].keys()), {'t', 'x'})
        # da_dt: CM/S to mm/s = 400000, 500000, 600000
        self.assertTrue(np.allclose(vals[1]['t'], (400000, 500000, 600000)))
        # da_dx: M/KM to mm/km = 7000, 8000, 9000
        self.assertTrue(np.allclose(vals[1]['x'], (7000, 8000, 9000)))

        # Test with n-D arrays and recursive=True
        a_nd = Scalar(np.random.rand(2, 3, 4), unit=Unit.M)
        da_dt_nd = Scalar(np.random.rand(2, 3, 4), unit=Unit.CM/Unit.S)
        a_nd.insert_deriv('t', da_dt_nd)
        vals = a_nd.into_unit(recursive=True)
        self.assertEqual(vals[0].shape, (2, 3, 4))
        self.assertEqual(vals[1]['t'].shape, (2, 3, 4))

##########################################################################################
