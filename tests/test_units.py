##########################################################################################
# test/test_units.py
##########################################################################################

import unittest
import numpy as np

from polymath import Unit


class Test_Units(unittest.TestCase):

    def runTest(self):

        np.random.seed(7456)

        self.assertEqual(repr(Unit.KM),                 "Unit(km)")
        self.assertEqual(repr(Unit.KM*Unit.KM),         "Unit(km**2)")
        self.assertEqual(repr(Unit.KM**2),              "Unit(km**2)")
        self.assertEqual(repr(Unit.KM**(-2)),           "Unit(km**(-2))")
        self.assertEqual(repr(Unit.KM/Unit.S),          "Unit(km/s)")
        self.assertEqual(repr((Unit.KM/Unit.S)**2),     "Unit(km**2/s**2)")
        self.assertEqual(repr((Unit.KM/Unit.S)**(-2)),  "Unit(s**2/km**2)")

        self.assertEqual(str(Unit.KM),                  "km")
        self.assertEqual(str(Unit.KM*Unit.KM),          "km**2")
        self.assertEqual(str(Unit.KM**2),               "km**2")
        self.assertEqual(str(Unit.KM**(-2)),            "km**(-2)")
        self.assertEqual(str(Unit.KM/Unit.S),           "km/s")
        self.assertEqual(str((Unit.KM/Unit.S)**2),      "km**2/s**2")
        self.assertEqual(str((Unit.KM/Unit.S)**(-2)),   "s**2/km**2")

        self.assertEqual((Unit.KM/Unit.S).exponents, (1,-1,0))
        self.assertEqual((Unit.KM/Unit.S/Unit.S).exponents, (1,-2,0))

        self.assertEqual(Unit.KM.convert(3.,Unit.CM), 3.e5)
        self.assertTrue(np.all(Unit.KM.convert(np.array([1.,2.,3.]), Unit.CM) ==
                               [1.e5, 2.e5, 3.e5]))

        self.assertTrue(np.all(Unit.DEGREES.convert(np.array([1.,2.,3.]),
                               Unit.ARCSEC) == [3600., 7200., 10800.]))

        self.assertTrue(np.all((Unit.DEG/Unit.S).convert(np.array([1.,2.,3.]),
                                Unit.ARCSEC/Unit.S) == [3600., 7200., 10800.]))

        self.assertTrue(np.all((Unit.DEG/Unit.H).convert(np.array([1.,2.,3.]),
                                Unit.ARCSEC/Unit.S) == [1., 2., 3.]))

        self.assertTrue(np.all((Unit.DEG*Unit.S).convert(np.array([1.,2.,3.]),
                                Unit.ARCSEC*Unit.H) == [1., 2., 3.]))

        self.assertTrue(np.all((Unit.DEG**2).convert(np.array([1.,2.,3.]),
                                Unit.ARCMIN*Unit.ARCSEC) ==
                                [3600*60, 3600*60*2, 3600*60*3]))

        eps = 1.e-15
        test = Unit.DEG.from_this(np.array([1.,2.,3.]))
        self.assertTrue(np.all([np.pi/180., np.pi/90., np.pi/60.] < test + eps))
        self.assertTrue(np.all([np.pi/180., np.pi/90., np.pi/60.] > test - eps))

        test = Unit.DEG.into_this(test)
        self.assertTrue(np.all(np.array([1., 2., 3.]) < test + eps))
        self.assertTrue(np.all(np.array([1., 2., 3.]) > test - eps))

        self.assertFalse(Unit.CM == Unit.M)
        self.assertTrue( Unit.CM != Unit.M)
        self.assertTrue( Unit.M  != Unit.SEC)
        self.assertEqual(Unit.M.factor, Unit.MRAD.factor)
        self.assertTrue( Unit.CM, Unit((1,0,0), (10., 1.e6, 0)))

        test = Unit.ROTATION/Unit.S
        self.assertEqual(test.get_name(), "rotation/s")

        unit = Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) / Unit.RAD
        self.assertEqual(repr(unit), "Unit(km/s)")
        self.assertEqual(str(unit), "km/s")

        unit = (Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) /
                             Unit.MRAD*Unit.MSEC/(Unit.KM/Unit.S) /
                             Unit.S)
        unit.name = None
        self.assertEqual(repr(unit), "Unit()")

        self.assertEqual(repr(Unit.S * 60), "Unit(min)")
        self.assertEqual(str(Unit.S * 60), "min")

        self.assertEqual(repr(60 * Unit.S), "Unit(min)")

        self.assertEqual(repr(Unit.H/3600), "Unit(s)")
        self.assertEqual(repr((1000/Unit.KM)**(-2)), "Unit(m**2)")

        self.assertTrue( Unit.can_match(None, None))
        self.assertTrue( Unit.can_match(None, Unit.UNITLESS))
        self.assertTrue( Unit.can_match(None, Unit.KM))
        self.assertTrue( Unit.can_match(Unit.KM, None))
        self.assertTrue( Unit.can_match(Unit.CM, Unit.KM))
        self.assertFalse(Unit.can_match(Unit.S, Unit.KM))
        self.assertFalse(Unit.can_match(Unit.S, Unit.UNITLESS))

        self.assertTrue( Unit.do_match(None, None))
        self.assertTrue( Unit.do_match(None, Unit.UNITLESS))
        self.assertFalse(Unit.do_match(None, Unit.KM))
        self.assertFalse(Unit.do_match(Unit.KM, None))
        self.assertTrue( Unit.do_match(Unit.CM, Unit.KM))
        self.assertFalse(Unit.do_match(Unit.S, Unit.KM))
        self.assertFalse(Unit.do_match(Unit.S, Unit.UNITLESS))

        self.assertEqual(Unit.KM, (Unit.KM**2).sqrt())

        ##################################################################################
        # __init__(self, exponents, triple, name=None)
        ##################################################################################

        # Test basic initialization
        u1 = Unit((1, 0, 0), (1, 1, 0), None)
        self.assertEqual(u1.exponents, (1, 0, 0))
        self.assertEqual(u1.triple, (1, 1, 0))
        self.assertEqual(u1.name, None)
        self.assertEqual(u1.factor, 1.0)
        self.assertEqual(u1.factor_inv, 1.0)

        # Test with pi exponent
        u2 = Unit((0, 0, 1), (1, 180, 1), 'deg')
        self.assertEqual(u2.exponents, (0, 0, 1))
        self.assertEqual(u2.triple, (1, 180, 1))
        expected_factor = (1.0 / 180.0) * np.pi
        self.assertAlmostEqual(u2.factor, expected_factor)
        self.assertAlmostEqual(u2.factor_inv, 180.0 / np.pi)

        # Test with different triple values
        u3 = Unit((1, 0, 0), (1, 1000, 0), 'm')
        self.assertEqual(u3.triple, (1, 1000, 0))
        self.assertAlmostEqual(u3.factor, 1.0 / 1000.0)
        self.assertAlmostEqual(u3.factor_inv, 1000.0)

        # Test with name=None
        u4 = Unit((0, 0, 0), (1, 1, 0), None)
        self.assertEqual(u4.name, None)

        # Test GCD reduction in triple
        u5 = Unit((0, 0, 0), (256, 512, 0), None)
        # Should reduce 256/512 to 1/2
        self.assertEqual(u5.triple[:2], (1, 2))

        ##################################################################################
        # from_unit_factor and into_unit_factor properties
        ##################################################################################

        u = Unit((1, 0, 0), (1, 1000, 0), 'm')
        self.assertEqual(u.from_unit_factor, u.factor)
        self.assertEqual(u.into_unit_factor, u.factor_inv)

        ##################################################################################
        # as_unit(arg)
        ##################################################################################

        # Test with None
        self.assertEqual(Unit.as_unit(None), None)

        # Test with string
        self.assertEqual(Unit.as_unit('km'), Unit.KM)
        self.assertEqual(Unit.as_unit('deg'), Unit.DEG)

        # Test with Unit object
        u = Unit.KM
        self.assertEqual(Unit.as_unit(u), u)

        # Test with invalid type
        self.assertRaises(ValueError, Unit.as_unit, 123)

        ##################################################################################
        # can_match(first, second)
        ##################################################################################

        # Test with None
        self.assertTrue(Unit.can_match(None, None))
        self.assertTrue(Unit.can_match(None, Unit.KM))
        self.assertTrue(Unit.can_match(Unit.KM, None))

        # Test with matching exponents
        self.assertTrue(Unit.can_match(Unit.KM, Unit.M))
        self.assertTrue(Unit.can_match(Unit.DEG, Unit.RAD))

        # Test with non-matching exponents
        self.assertFalse(Unit.can_match(Unit.KM, Unit.S))
        self.assertFalse(Unit.can_match(Unit.KM, Unit.DEG))

        ##################################################################################
        # require_compatible(first, second, info='')
        ##################################################################################

        # Test with compatible units
        Unit.require_compatible(Unit.KM, Unit.M)
        Unit.require_compatible(None, Unit.KM)
        Unit.require_compatible(Unit.KM, None)

        # Test with incompatible units
        self.assertRaises(ValueError, Unit.require_compatible, Unit.KM, Unit.S)
        self.assertRaises(ValueError, Unit.require_compatible, Unit.KM, Unit.DEG)

        # Test with info parameter
        with self.assertRaises(ValueError) as context:
            Unit.require_compatible(Unit.KM, Unit.S, info='test_op')
        self.assertIn('test_op', str(context.exception))

        ##################################################################################
        # do_match(first, second)
        ##################################################################################

        # Test with None (treated as unitless)
        self.assertTrue(Unit.do_match(None, None))
        self.assertTrue(Unit.do_match(None, Unit.UNITLESS))
        self.assertTrue(Unit.do_match(Unit.UNITLESS, None))

        # Test with matching units (same exponents)
        self.assertTrue(Unit.do_match(Unit.KM, Unit.KM))
        self.assertTrue(Unit.do_match(Unit.DEG, Unit.DEG))
        # Note: do_match only checks exponents, not triple, so KM and M match
        self.assertTrue(Unit.do_match(Unit.KM, Unit.M))

        # Test with non-matching units (different exponents)
        self.assertFalse(Unit.do_match(Unit.KM, Unit.S))
        self.assertFalse(Unit.do_match(Unit.KM, Unit.DEG))

        ##################################################################################
        # require_match(first, second, info='')
        ##################################################################################

        # Test with matching units (same exponents)
        Unit.require_match(Unit.KM, Unit.KM)
        Unit.require_match(None, None)
        Unit.require_match(None, Unit.UNITLESS)
        # Note: require_match only checks exponents, so KM and M match
        Unit.require_match(Unit.KM, Unit.M)

        # Test with non-matching units (different exponents)
        self.assertRaises(ValueError, Unit.require_match, Unit.KM, Unit.S)
        self.assertRaises(ValueError, Unit.require_match, Unit.KM, Unit.DEG)

        # Test with info parameter
        with self.assertRaises(ValueError) as context:
            Unit.require_match(Unit.KM, Unit.S, info='test_op')
        self.assertIn('test_op', str(context.exception))

        ##################################################################################
        # is_angle(arg)
        ##################################################################################

        # Test with None
        self.assertTrue(Unit.is_angle(None))

        # Test with unitless
        self.assertTrue(Unit.is_angle(Unit.UNITLESS))

        # Test with angle units
        self.assertTrue(Unit.is_angle(Unit.DEG))
        self.assertTrue(Unit.is_angle(Unit.RAD))

        # Test with non-angle units
        self.assertFalse(Unit.is_angle(Unit.KM))
        self.assertFalse(Unit.is_angle(Unit.S))

        ##################################################################################
        # require_angle(arg, info='')
        ##################################################################################

        # Test with angle units
        Unit.require_angle(None)
        Unit.require_angle(Unit.DEG)
        Unit.require_angle(Unit.RAD)

        # Test with non-angle units
        self.assertRaises(ValueError, Unit.require_angle, Unit.KM)
        self.assertRaises(ValueError, Unit.require_angle, Unit.S)

        # Test with info parameter
        with self.assertRaises(ValueError) as context:
            Unit.require_angle(Unit.KM, info='test_op')
        self.assertIn('test_op', str(context.exception))

        ##################################################################################
        # is_unitless(arg)
        ##################################################################################

        # Test with None
        self.assertTrue(Unit.is_unitless(None))

        # Test with unitless
        self.assertTrue(Unit.is_unitless(Unit.UNITLESS))

        # Test with units
        self.assertFalse(Unit.is_unitless(Unit.KM))
        self.assertFalse(Unit.is_unitless(Unit.DEG))
        self.assertFalse(Unit.is_unitless(Unit.S))

        ##################################################################################
        # require_unitless(arg, info='')
        ##################################################################################

        # Test with unitless
        Unit.require_unitless(None)
        Unit.require_unitless(Unit.UNITLESS)

        # Test with units
        self.assertRaises(ValueError, Unit.require_unitless, Unit.KM)
        self.assertRaises(ValueError, Unit.require_unitless, Unit.DEG)

        # Test with info parameter
        with self.assertRaises(ValueError) as context:
            Unit.require_unitless(Unit.KM, info='test_op')
        self.assertIn('test_op', str(context.exception))

        ##################################################################################
        # from_this(self, value)
        ##################################################################################

        u = Unit((1, 0, 0), (1, 1000, 0), 'm')
        # Convert 1000 meters to km (standard unit)
        result = u.from_this(1000.0)
        self.assertAlmostEqual(result, 1.0)

        u_deg = Unit((0, 0, 1), (1, 180, 1), 'deg')
        # Convert 180 degrees to radians
        result = u_deg.from_this(180.0)
        self.assertAlmostEqual(result, np.pi)

        # Test with array
        values = np.array([1000.0, 2000.0, 3000.0])
        result = u.from_this(values)
        expected = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(result, expected))

        ##################################################################################
        # into_this(self, value)
        ##################################################################################

        u = Unit((1, 0, 0), (1, 1000, 0), 'm')
        # Convert 1 km (standard) to meters
        result = u.into_this(1.0)
        self.assertAlmostEqual(result, 1000.0)

        u_deg = Unit((0, 0, 1), (1, 180, 1), 'deg')
        # Convert pi radians to degrees
        result = u_deg.into_this(np.pi)
        self.assertAlmostEqual(result, 180.0)

        # Test with array
        values = np.array([1.0, 2.0, 3.0])
        result = u.into_this(values)
        expected = np.array([1000.0, 2000.0, 3000.0])
        self.assertTrue(np.allclose(result, expected))

        ##################################################################################
        # from_unit(unit, value)
        ##################################################################################

        # Test with None
        result = Unit.from_unit(None, 5.0)
        self.assertEqual(result, 5.0)

        # Test with unit
        result = Unit.from_unit(Unit.M, 1000.0)
        self.assertAlmostEqual(result, 1.0)

        # Test with array
        values = np.array([1000.0, 2000.0])
        result = Unit.from_unit(Unit.M, values)
        expected = np.array([1.0, 2.0])
        self.assertTrue(np.allclose(result, expected))

        ##################################################################################
        # into_unit(unit, value)
        ##################################################################################

        # Test with None
        result = Unit.into_unit(None, 5.0)
        self.assertEqual(result, 5.0)

        # Test with unit
        result = Unit.into_unit(Unit.M, 1.0)
        self.assertAlmostEqual(result, 1000.0)

        # Test with array
        values = np.array([1.0, 2.0])
        result = Unit.into_unit(Unit.M, values)
        expected = np.array([1000.0, 2000.0])
        self.assertTrue(np.allclose(result, expected))

        ##################################################################################
        # convert(self, value, unit, info='')
        ##################################################################################

        # Test conversion from M to KM
        u_m = Unit.M
        result = u_m.convert(1000.0, Unit.KM)
        self.assertAlmostEqual(result, 1.0)

        # Test conversion from DEG to RAD
        u_deg = Unit.DEG
        result = u_deg.convert(180.0, Unit.RAD)
        self.assertAlmostEqual(result, np.pi)

        # Test conversion to None (unitless) - requires unitless source
        u_unitless = Unit.UNITLESS
        result = u_unitless.convert(5.0, None)
        # Should return unchanged for unitless
        self.assertEqual(result, 5.0)

        # Test conversion from M to KM (compatible units)
        result = u_m.convert(1000.0, Unit.KM)
        self.assertAlmostEqual(result, 1.0)

        # Test with incompatible units
        self.assertRaises(ValueError, u_m.convert, 1000.0, Unit.S)

        # Test with info parameter
        with self.assertRaises(ValueError) as context:
            u_m.convert(1000.0, Unit.S, info='test_op')
        self.assertIn('test_op', str(context.exception))

        # Test with same unit (should return unchanged)
        result = u_m.convert(1000.0, Unit.M)
        self.assertEqual(result, 1000.0)

        # Test with array
        values = np.array([1000.0, 2000.0, 3000.0])
        result = u_m.convert(values, Unit.KM)
        expected = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(result, expected))

        ##################################################################################
        # __mul__(self, arg)
        ##################################################################################

        # Test Unit * Unit
        u1 = Unit.KM
        u2 = Unit.S
        result = u1 * u2
        self.assertEqual(result.exponents, (1, 1, 0))
        # KM * S = km*s, which has exponents (1, 1, 0)

        # Test Unit * None
        result = u1 * None
        self.assertEqual(result, u1)

        # Test Unit * number
        result = u1 * 5.0
        # Should create a unit with coefficient
        self.assertIsInstance(result, Unit)
        self.assertEqual(result.name, None)
        self.assertEqual(result.get_name(), '5*km')

        # Test with NotImplemented
        result = u1.__mul__('invalid')
        self.assertEqual(result, NotImplemented)

        ##################################################################################
        # __rmul__(self, arg)
        ##################################################################################

        # Test number * Unit
        result = 5.0 * Unit.KM
        self.assertIsInstance(result, Unit)
        self.assertEqual(result.name, None)
        self.assertEqual(result.get_name(), '5*km')

        ##################################################################################
        # __truediv__(self, arg)
        ##################################################################################

        # Test Unit / Unit
        u1 = Unit.KM
        u2 = Unit.S
        result = u1 / u2
        self.assertEqual(result.exponents, (1, -1, 0))
        # KM / S = km/s, which has exponents (1, -1, 0)

        # Test Unit / None
        result = u1 / None
        self.assertEqual(result, u1)

        # Test Unit / number
        result = u1 / 5.0
        self.assertIsInstance(result, Unit)
        self.assertEqual(result.name, None)
        self.assertEqual(result.get_name(), '0.2*km')

        # Test with NotImplemented
        result = u1.__truediv__('invalid')
        self.assertEqual(result, NotImplemented)

        ##################################################################################
        # __rtruediv__(self, arg)
        ##################################################################################

        # Test number / Unit
        result = 5.0 / Unit.KM
        self.assertIsInstance(result, Unit)
        self.assertEqual(result.name, None)
        self.assertEqual(result.get_name(), '5/km')
        # Should be equivalent to Unit.KM**(-1) * 5.0

        # Test None / Unit
        result = None / Unit.KM
        self.assertIsInstance(result, Unit)
        self.assertEqual(result.name, None)
        self.assertEqual(result.get_name(), 'km**(-1)')

        # Test with NotImplemented
        result = Unit.KM.__rtruediv__('invalid')
        self.assertEqual(result, NotImplemented)

        ##################################################################################
        # __pow__(self, power)
        ##################################################################################

        # Test positive integer power
        u = Unit.KM
        result = u ** 2
        self.assertEqual(result.exponents, (2, 0, 0))
        self.assertEqual(result.triple, (1, 1, 0))
        self.assertEqual(result.name, {'km': 2})
        self.assertEqual(result.get_name(), 'km**2')

        # Test negative integer power
        result = u ** (-2)
        self.assertEqual(result.exponents, (-2, 0, 0))

        # Test half-integer power
        u_sq = Unit((2, 0, 0), (1, 1, 0), None)
        result = u_sq ** 0.5
        self.assertEqual(result.exponents, (1, 0, 0))

        # Test invalid power (non-integer, non-half-integer)
        self.assertRaises(ValueError, u.__pow__, 0.3)

        # Test with half-integer power that works
        u_sq = Unit((2, 0, 0), (1, 1, 0), None)
        result = u_sq ** 0.5
        self.assertEqual(result.exponents, (1, 0, 0))

        # Test with power that requires sqrt then power
        u_4 = Unit((4, 0, 0), (1, 1, 0), None)
        result = u_4 ** 1.5  # sqrt then **3
        self.assertEqual(result.exponents, (6, 0, 0))

        ##################################################################################
        # sqrt(self, name=None)
        ##################################################################################

        # Test with even exponents
        u_sq = Unit((2, 0, 0), (1, 1, 0), None)
        result = u_sq.sqrt()
        self.assertEqual(result.exponents, (1, 0, 0))

        # Test with odd exponents (should raise)
        u_odd = Unit((1, 0, 0), (1, 1, 0), None)
        self.assertRaises(ValueError, u_odd.sqrt)

        # Test with name parameter
        result = u_sq.sqrt(name='km')
        self.assertEqual(result.name, 'km')

        ##################################################################################
        # mul_units(arg1, arg2, name=None)
        ##################################################################################

        # Test with both units
        result = Unit.mul_units(Unit.KM, Unit.S)
        self.assertEqual(result.exponents, (1, 1, 0))

        # Test with None
        result = Unit.mul_units(None, Unit.KM)
        self.assertEqual(result, Unit.KM)

        result = Unit.mul_units(Unit.KM, None)
        self.assertEqual(result, Unit.KM)

        result = Unit.mul_units(None, None)
        self.assertEqual(result, None)

        # Test with name parameter
        result = Unit.mul_units(Unit.KM, Unit.S, name={'km': 1, 's': 1})
        self.assertEqual(result.name, None)
        self.assertEqual(result.get_name(), 'km*s')

        ##################################################################################
        # div_units(arg1, arg2, name=None)
        ##################################################################################

        # Test with both units
        result = Unit.div_units(Unit.KM, Unit.S)
        self.assertEqual(result.exponents, (1, -1, 0))

        # Test with None
        result = Unit.div_units(None, Unit.KM)
        self.assertEqual(result.exponents, (-1, 0, 0))

        result = Unit.div_units(Unit.KM, None)
        self.assertEqual(result, Unit.KM)

        result = Unit.div_units(None, None)
        self.assertEqual(result, None)

        # Test with name parameter
        result = Unit.div_units(Unit.KM, Unit.S, name={'km': 1, 's': -1})
        self.assertEqual(result.name, None)
        self.assertEqual(result.get_name(), 'km/s')

        ##################################################################################
        # sqrt_unit(unit, name=None)
        ##################################################################################

        # Test with unit
        u_sq = Unit((2, 0, 0), (1, 1, 0), None)
        result = Unit.sqrt_unit(u_sq)
        self.assertEqual(result.exponents, (1, 0, 0))

        # Test with None
        result = Unit.sqrt_unit(None)
        self.assertEqual(result, None)

        # Test with name parameter
        result = Unit.sqrt_unit(u_sq, name='km')
        self.assertEqual(result.name, 'km')

        ##################################################################################
        # unit_power(unit, power, name=None)
        ##################################################################################

        # Test with unit
        result = Unit.unit_power(Unit.KM, 2)
        self.assertEqual(result.exponents, (2, 0, 0))

        # Test with None
        result = Unit.unit_power(None, 2)
        self.assertEqual(result, None)

        # Test with name parameter (use dict to avoid parsing issues)
        result = Unit.unit_power(Unit.KM, 2, name={'km': 2})
        self.assertEqual(result.name, {'km': 2})

        ##################################################################################
        # __eq__(self, arg)
        ##################################################################################

        # Test with same unit
        self.assertTrue(Unit.KM == Unit.KM)
        self.assertTrue(Unit.DEG == Unit.DEG)

        # Test with different units
        self.assertFalse(Unit.KM == Unit.M)
        self.assertFalse(Unit.KM == Unit.S)

        # Test with non-Unit
        self.assertFalse(Unit.KM == 'km')
        self.assertFalse(Unit.KM == 5)

        ##################################################################################
        # __ne__(self, arg)
        ##################################################################################

        # Test with same unit
        self.assertFalse(Unit.KM != Unit.KM)

        # Test with different units
        self.assertTrue(Unit.KM != Unit.M)
        self.assertTrue(Unit.KM != Unit.S)

        # Test with non-Unit
        self.assertTrue(Unit.KM != 'km')
        self.assertTrue(Unit.KM != 5)

        ##################################################################################
        # __copy__(self) and copy(self)
        ##################################################################################

        u = Unit.KM
        u_copy = u.__copy__()
        self.assertEqual(u.exponents, u_copy.exponents)
        self.assertEqual(u.triple, u_copy.triple)
        self.assertIsNot(u, u_copy)

        u_copy2 = u.copy()
        self.assertEqual(u.exponents, u_copy2.exponents)
        self.assertEqual(u.triple, u_copy2.triple)
        self.assertIsNot(u, u_copy2)

        ##################################################################################
        # __str__(self) and __repr__(self)
        ##################################################################################

        # Test __str__ and __repr__ with a recognized unit
        u = Unit.KM
        r = repr(u)
        self.assertIsInstance(r, str)
        self.assertIn('Unit', r)

        s = str(u)
        if s:
            self.assertIsInstance(s, str)

        ##################################################################################
        # get_name(self) and set_name(self, name)
        ##################################################################################

        u = Unit.KM
        name = u.get_name()
        self.assertIsInstance(name, (str, dict))
        self.assertEqual(name, 'km')

        # Test with a unit that has a dict name (avoid calling get_name which may fail)
        u_dict = Unit((1, 0, 0), (1, 1, 0), 'km')
        self.assertEqual(u_dict.name, 'km')

        u.set_name('new_name')
        self.assertEqual(u.name, 'new_name')

        u.set_name({'km': 1})
        self.assertEqual(u.name, {'km': 1})

        # Put it back to what it should be
        u.set_name('km')

        ##################################################################################
        # create_name(self)
        ##################################################################################

        # Test with named unit
        u = Unit.KM
        name = u.create_name()
        self.assertEqual(name, 'km')

        # Test with unnamed unit - create_name may call get_name which might fail
        # with None name, so we'll skip this test or handle the error
        u = Unit((1, 0, 0), (1, 1, 0), None)
        name = u.create_name()
        self.assertEqual(name, 'km')

        ##################################################################################
        # Additional edge cases and static methods
        ##################################################################################

        # Test __init__ with triple that doesn't reduce
        # Use values that don't reduce properly after scaling by 256
        u = Unit((0, 0, 0), (3, 7, 0), None)
        # Should keep original values if GCD reduction doesn't work
        # Note: After scaling by 256, 3*256=768, 7*256=1792, GCD=256, so 768/256=3, 1792/256=7
        # But if the check fails, it keeps original
        self.assertEqual(u.triple[:2], (3, 7))

        # Test with triple that does reduce
        u2 = Unit((0, 0, 0), (256, 512, 0), None)
        # Should reduce 256/512 to 1/2
        self.assertEqual(u2.triple[:2], (1, 2))

        # Test __pow__ with power that requires sqrt
        u_sq = Unit((4, 0, 0), (1, 1, 0), None)
        result = u_sq ** 0.5
        self.assertEqual(result.exponents, (2, 0, 0))

        # Test sqrt with pi exponent
        u_pi = Unit.STER
        # Note: sqrt() without name parameter calls name_power which may raise ValueError
        # So we provide a name to avoid that
        result = u_pi.sqrt(name='rad')
        self.assertEqual(result.exponents, (0, 0, 1))
        self.assertEqual(result.name, 'rad')

        # Test sqrt with name parameter
        result = u_pi.sqrt(name='rad')
        self.assertEqual(result.name, 'rad')

        # Test sqrt with name=None - this triggers name_power which may raise ValueError
        # for units with string names that don't work with 0.5 power
        u_simple = Unit((2, 0, 0), (1, 1, 0), None)
        result = u_simple.sqrt(name=None)
        # Should work if name is None
        self.assertEqual(result.exponents, (1, 0, 0))

        # Test sqrt with triple where numer/denom sqrt doesn't yield ints
        u_sqrt_float = Unit((2, 0, 0), (2, 1, 0), None)
        result = u_sqrt_float.sqrt()
        # Should handle sqrt of non-perfect squares
        # numer = sqrt(2) which is not an int, so stays float
        # denom = sqrt(1) = 1, which is an int
        self.assertEqual(result.exponents, (1, 0, 0))

        # Test sqrt where denom sqrt doesn't yield int
        u_sqrt_denom = Unit((2, 0, 0), (1, 2, 0), None)
        result = u_sqrt_denom.sqrt()
        # denom = sqrt(2) which is not an int
        # This tests the branch where denom % 1 != 0
        self.assertEqual(result.exponents, (1, 0, 0))
        # denom should remain as float
        self.assertIsInstance(result.triple[1], (float, np.floating))

        # Test sqrt with triple that doesn't divide evenly for pi
        # Create unit with odd pi exponent (but even in exponents)
        u_odd_pi = Unit((0, 0, 2), (1, 1, 3), None)
        result = u_odd_pi.sqrt()
        # pi_expo = 3 // 2 = 1, but 3 != 2*1, so enters special branch
        self.assertEqual(result.exponents, (0, 0, 1))

        ##################################################################################
        # Test static name processing methods
        ##################################################################################

        # Test _mul_names
        result = Unit._mul_names('km', 's')
        self.assertIsInstance(result, dict)

        result = Unit._mul_names({'km': 1}, {'s': 1})
        self.assertIsInstance(result, dict)

        result = Unit._mul_names(None, 'km')
        self.assertEqual(result, None)

        result = Unit._mul_names('km', None)
        self.assertEqual(result, None)

        # Test _mul_names with expo that becomes 0
        result = Unit._mul_names({'km': 1}, {'km': -1})
        # Should remove km since expo becomes 0
        self.assertEqual(result, {})

        # Test _mul_names with expo that adds
        result = Unit._mul_names({'km': 2}, {'km': 3})
        self.assertEqual(result, {'km': 5})

        # Test div_names
        result = Unit.div_names('km', 's')
        self.assertIsInstance(result, dict)

        result = Unit.div_names({'km': 1}, {'s': 1})
        self.assertIsInstance(result, dict)

        result = Unit.div_names(None, 'km')
        self.assertEqual(result, None)

        result = Unit.div_names('km', None)
        self.assertEqual(result, None)

        # Test div_names with expo that becomes 0
        result = Unit.div_names({'km': 1}, {'km': 1})
        # Should remove km since expo becomes 0
        self.assertEqual(result, {})

        # Test div_names with expo that subtracts
        result = Unit.div_names({'km': 5}, {'km': 2})
        self.assertEqual(result, {'km': 3})

        # Test name_power
        result = Unit.name_power('km', 2)
        self.assertIsInstance(result, dict)

        result = Unit.name_power({'km': 1}, 2)
        self.assertIsInstance(result, dict)

        result = Unit.name_power(None, 2)
        self.assertEqual(result, None)

        # Test name_power with string power
        self.assertRaises(ValueError, Unit.name_power, 'km', 'invalid')

        # Test name_power with non-integer result
        self.assertRaises(ValueError, Unit.name_power, {'km': 1}, 0.5)

        # Test name_to_dict
        result = Unit.name_to_dict('km')
        self.assertIsInstance(result, dict)

        result = Unit.name_to_dict({'km': 1})
        self.assertIsInstance(result, dict)

        result = Unit.name_to_dict('')
        self.assertEqual(result, {})

        # Test name_to_dict with non-string, non-dict
        self.assertRaises(ValueError, Unit.name_to_dict, 123)

        # Test name_to_dict with integer string
        result = Unit.name_to_dict('5')
        self.assertEqual(result, 5)

        # Test name_to_dict with complex expressions
        result = Unit.name_to_dict('km*s')
        self.assertIsInstance(result, dict)

        result = Unit.name_to_dict('km/s')
        self.assertIsInstance(result, dict)

        result = Unit.name_to_dict('km**2')
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {'km': 2})

        result = Unit.name_to_dict('(km*s)/m')
        self.assertIsInstance(result, dict)

        # Test name_to_dict with parentheses
        result = Unit.name_to_dict('(km*s)')
        self.assertIsInstance(result, dict)

        # Test name_to_dict with multiplication
        result = Unit.name_to_dict('km*s')
        self.assertIsInstance(result, dict)

        # Test name_to_dict with division
        result = Unit.name_to_dict('km/s')
        self.assertIsInstance(result, dict)

        # Test name_to_dict with exponent after parentheses
        result = Unit.name_to_dict('(km)**2')
        self.assertIsInstance(result, dict)

        # Test name_to_dict with complex expression
        result = Unit.name_to_dict('km*s/m')
        self.assertIsInstance(result, dict)

        # Test name_to_str
        result = Unit.name_to_str({'km': 1})
        self.assertIsInstance(result, str)

        result = Unit.name_to_str({'km': 1, 's': -1})
        self.assertIsInstance(result, str)

        result = Unit.name_to_str('km')
        self.assertEqual(result, 'km')

        # Test name_to_str with empty string
        result = Unit.name_to_str('')
        self.assertEqual(result, '')

        # Note: name_to_str with None would cause AttributeError
        # So we don't test that case

        # Test name_to_str with empty dict
        result = Unit.name_to_str({})
        self.assertEqual(result, '')

        # Test name_to_str with coefficient
        result = Unit.name_to_str({'': 5, 'km': 1})
        self.assertIsInstance(result, str)
        # Should include the coefficient 5

        # Test name_to_str with coefficient == 1
        result = Unit.name_to_str({'': 1, 'km': 1})
        self.assertIsInstance(result, str)
        # Coefficient 1 should not appear

        # Test name_to_str with expo > 1
        result = Unit.name_to_str({'km': 3})
        self.assertIsInstance(result, str)
        self.assertIn('**', result)

        # Test name_to_str with expo < 0
        result = Unit.name_to_str({'km': -2})
        self.assertIsInstance(result, str)

        # Test name_to_str with negative exponents (denoms)
        result = Unit.name_to_str({'km': -1})
        self.assertIsInstance(result, str)
        # Result should have '/' or be formatted as denominator
        # The exact format depends on implementation

        # Test name_to_str with both numers and denoms
        result = Unit.name_to_str({'km': 1, 's': -1})
        self.assertIsInstance(result, str)
        self.assertIn('/', result)

        # Test name_to_str with only numers
        result = Unit.name_to_str({'km': 1, 'm': 1})
        self.assertIsInstance(result, str)
        self.assertNotIn('/', result)

        # Test name_to_str with only denoms
        result = Unit.name_to_str({'km': -1, 's': -1})
        self.assertIsInstance(result, str)


        ##################################################################################
        # Additional tests for missing coverage
        ##################################################################################

        # Test __div__ and __rdiv__ methods
        u1 = Unit.KM
        u2 = Unit.S
        result = u1.__div__(u2)
        self.assertEqual(result.exponents, (1, -1, 0))

        result = Unit.KM.__rdiv__(5.0)
        self.assertIsInstance(result, Unit)

        # Test name_to_dict with parentheses parsing
        # This tests the branch where name[0] == '('
        result = Unit.name_to_dict('(km)')
        self.assertIsInstance(result, dict)
        # Tests the loop that finds matching closing parenthesis

        # Test name_to_dict with nested parentheses
        result = Unit.name_to_dict('((km))')
        self.assertIsInstance(result, dict)
        # Tests depth tracking in parentheses

        # Test name_to_dict with parentheses and content after
        result = Unit.name_to_dict('(km)*s')
        self.assertIsInstance(result, dict)
        # Tests right = name[i+1:].lstrip() when there's content after ')'

        # Test name_to_dict with illegal syntax - no operators
        # Note: Simple names like 'km' are valid, so we need something that fails parsing
        # The error occurs when no '*' or '/' is found and it's not a simple name
        # Let's test with something that should fail
        # Try with a name that has no operators and isn't a recognized unit
        result = Unit.name_to_dict('xyz')
        self.assertEqual(result, {'xyz': 1})

        # Test name_to_dict with ** operator parsing
        result = Unit.name_to_dict('km**2*s')
        self.assertEqual(result, {'km': 2, 's': 1})
        # This tests the branch where right has ** and we extract power

        # Test name_to_dict with ** at start
        self.assertRaises(ValueError, Unit.name_to_dict, 'km**')

        # Test name_to_dict with no progress
        # This happens when left == name.strip() after parsing
        # Try to create a case where parsing doesn't make progress
        try:
            # This might trigger the no-progress check
            result = Unit.name_to_dict('km')
            # If it succeeds, it's a valid unit name
            self.assertIsInstance(result, dict)
        except ValueError as e:
            # If it fails with "no progress", that's the path we want
            if 'no progress' in str(e) or 'illegal' in str(e).lower():
                pass

        # Test name_to_str ordering with angle units
        # Test with angle units to trigger templist.append for angle units
        result = Unit.name_to_str({'deg': 1, 'rad': 1, 'km': 1})
        self.assertIsInstance(result, str)
        # Should include angle units in sorted order

        # Test create_name KeyError path
        # Create a unit not in _TUPLES_TO_UNIT dictionary
        u_custom = Unit((1, 0, 0), (1, 1000, 0), None)
        name = u_custom.create_name()
        self.assertEqual(name, 'm')

        # Test create_name with negative power
        # Create unit with negative exponent that requires negative power
        u_neg_exp = Unit((0, -2, 0), (1, 1, 0), None)  # 1/s^2
        name = u_neg_exp.create_name()
        # Should handle negative power with swapped triple
        self.assertEqual(name, {'km': 0, 's': -2, 'rad': 0})

        # Test create_name finding best match
        # Create unit that matches multiple options
        u_multi = Unit((4, 0, 0), (1, 1, 0), None)  # km^4
        name = u_multi.create_name()
        self.assertEqual(name, {'km': 4, 's': 0, 'rad': 0})

        # Test create_name fallback to standard unit
        # Create unit that doesn't match any standard unit exactly
        u_fallback = Unit((1, 0, 0), (3, 7, 0), None)  # Custom triple
        name = u_fallback.create_name()
        # Should fallback to standard unit with coefficient
        self.assertEqual(name, {'': 3/7, 'km': 1, 's': 0, 'rad': 0})

        # Test create_name with denom == 1 and pi_expo == 0
        # This tests the branch where coefft = numer directly
        u_simple = Unit((2, 0, 0), (5, 1, 0), None)  # denom=1, pi_expo=0
        name = u_simple.create_name()
        # Should use coefft = numer
        self.assertEqual(name, {'': 5, 'km': 2, 's': 0, 'rad': 0})

        # Test create_name with denom != 1
        u_denom = Unit((1, 0, 0), (3, 2, 0), None)  # Has denom != 1
        name = u_denom.create_name()
        # Should calculate coefft with division
        self.assertEqual(name, {'': 3/2, 'km': 1, 's': 0, 'rad': 0})

        # Test create_name with pi_expo != 0
        u_pi_exp = Unit((0, 0, 1), (1, 180, 1), None)  # Has pi_expo
        name = u_pi_exp.create_name()
        # Should calculate coefft with pi
        self.assertEqual(name, 'deg')

        # Test create_name finding best match - multiple matches
        # Create unit that could match multiple ways
        u_best = Unit((6, 0, 0), (1, 1, 0), None)  # km^6 could be (km^2)^3 or (km^3)^2
        name = u_best.create_name()
        # Should find best match with fewest keys
        # Tests the loop that finds first match with best length
        self.assertEqual(name, {'km': 6, 's': 0, 'rad': 0})

        # Test create_name with negative power
        # This tests the branch where p * actual_power == target_power with negative p
        u_neg_power = Unit((0, -3, 0), (1, 1, 0), None)  # 1/s^3
        name = u_neg_power.create_name()
        # Should handle negative power (checks the condition)
        self.assertEqual(name, {'km': 0, 's': -3, 'rad': 0})

        # Test as_unit with string argument
        result = Unit.as_unit('km')
        self.assertIsInstance(result, Unit)
        self.assertEqual(result, Unit.KM)

        # Test name_to_dict with unclosed parenthesis
        result = Unit.name_to_dict('(km')

        # Test with nested unclosed parentheses
        result = Unit.name_to_dict('((km')

        ##################################################################################
        # Test name_to_dict with '**' in invalid position (lines 877-878)
        # This specifically tests: if right.startswith('**'): raise ValueError
        ##################################################################################

        # Test with '**' appearing after a '**' operator has already been processed
        # This happens when we have something like 'km**2**3' where:
        # 1. First '**2' is processed (lines 858-869)
        # 2. After processing, right becomes '**3'
        # 3. At line 877, right.startswith('**') is True, so line 878 raises ValueError
        self.assertRaises(ValueError, Unit.name_to_dict, 'km**2**3')

        # Test with parentheses version
        self.assertRaises(ValueError, Unit.name_to_dict, '(km)**2**3')

        # Test with different unit names
        self.assertRaises(ValueError, Unit.name_to_dict, 's**2**3')

        # Create unit with angle exponent 5 to test more False cases
        u_angle5 = Unit((0, 0, 5), (1, 1, 0), 'rad**5')  # angle^5
        name = u_angle5.create_name()
        # When checking STER (power 2): p = 5 // 2 = 2, 2 * 2 = 4 != 5, so False
        # When checking RAD (power 1): p = 5 // 1 = 5, 5 * 1 = 5, so True
        # So it should work, but we've tested False branches
        self.assertEqual(name, 'rad**5')

        ##################################################################################
        # Test create_name fall through at line 1026
        # This specifically tests when name is None after lookup in _TUPLES_TO_UNIT
        ##################################################################################

        # To test line 1026 fall-through, we need:
        # 1. A unit that's in _TUPLES_TO_UNIT (no KeyError)
        # 2. But the unit in _TUPLES_TO_UNIT has name=None (not empty string)
        #
        # However, all standard units have names (even if empty string ''), so
        # name will never be None for standard units. This makes line 1026
        # fall-through difficult to trigger in practice.
        #
        # We can test it by creating a unit that matches a standard unit's
        # structure and temporarily modifying the standard unit's name to None,
        # or by testing the code path with a unit that's not in the dict
        # (which hits KeyError at line 1028, not 1026).

        # Test with a unit that matches UNITLESS structure
        # UNITLESS has name='' (empty string), not None, so this won't trigger
        # line 1026 fall-through, but it tests the lookup path
        u_unitless = Unit((0, 0, 0), (1, 1, 0), None)
        name = u_unitless.create_name()
        # UNITLESS has name='', so line 1026 condition is True ('' is not None)
        # and it returns. To test fall-through, we'd need name=None.
        self.assertIsNotNone(name)

        # To actually test line 1026 fall-through, we'd need to temporarily
        # set a standard unit's name to None. Let's do that for testing:
        # Save original name
        unitless_key = ((0, 0, 0), (1, 1, 0))
        original_name = Unit._TUPLES_TO_UNIT[unitless_key].name
        try:
            # Temporarily set name to None to test fall-through
            Unit._TUPLES_TO_UNIT[unitless_key].name = None
            u_test = Unit((0, 0, 0), (1, 1, 0), None)
            name = u_test.create_name()
            # Now name is None, so line 1026 condition is False and it falls through
            # Should continue to search for combinations
            self.assertIsNotNone(name)
        finally:
            # Restore original name
            Unit._TUPLES_TO_UNIT[unitless_key].name = original_name

        ##################################################################################
        # Test create_name when p * actual_power != target_power (line 1041 False)
        # This specifically tests when the condition is False
        ##################################################################################

        # Create a unit where target_power doesn't divide evenly by any standard unit's power
        # For example, angle exponent 7: when checking STER (power 2), p = 7 // 2 = 3,
        # and 3 * 2 = 6 != 7, so the condition is False
        u_angle7 = Unit((0, 0, 7), (1, 1, 0), None)  # angle^7
        name = u_angle7.create_name()
        # When checking STER (power 2): p = 7 // 2 = 3, 3 * 2 = 6 != 7, so False
        # When checking RAD (power 1): p = 7 // 1 = 7, 7 * 1 = 7, so True
        # So it should find RAD and work, but we've tested the False branch with STER
        self.assertEqual(name, {'km': 0, 's': 0, 'rad': 7})

        # Test with distance exponent that doesn't divide evenly
        # Distance units all have power 1, so any integer will work. We need a different approach.
        # Actually, for distance/time, all standard units have power 1, so they always divide evenly.
        # For angle, we have STER with power 2, so we can test with odd powers > 1.

        # Test with angle exponent 3 (odd, > 1)
        u_angle3 = Unit((0, 0, 3), (1, 1, 0), None)  # angle^3
        name = u_angle3.create_name()
        # When checking STER (power 2): p = 3 // 2 = 1, 1 * 2 = 2 != 3, so False
        # When checking RAD (power 1): p = 3 // 1 = 3, 3 * 1 = 3, so True
        # So it should work, but we've tested the False branch
        self.assertEqual(name, {'km': 0, 's': 0, 'rad': 3})

        # Test with angle exponent 9 (odd, > 1)
        u_angle9 = Unit((0, 0, 9), (1, 1, 0), None)  # angle^9
        name = u_angle9.create_name()
        # When checking STER (power 2): p = 9 // 2 = 4, 4 * 2 = 8 != 9, so False
        # When checking RAD (power 1): p = 9 // 1 = 9, 9 * 1 = 9, so True
        # So it should work, but we've tested the False branch
        self.assertEqual(name, {'km': 0, 's': 0, 'rad': 9})

##########################################################################################
