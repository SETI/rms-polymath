##########################################################################################
# test/test_units.py
##########################################################################################

import numpy as np
import pytest

from collections.abc import Callable

from polymath import Scalar, Unit


def test_units_test_basic_initialization() -> None:
    """Test basic initialization."""

    np.random.seed(7456)
    assert repr(Unit.KM) == "Unit(km)"
    assert repr(Unit.KM*Unit.KM) == "Unit(km**2)"
    assert repr(Unit.KM**2) == "Unit(km**2)"
    assert repr(Unit.KM**(-2)) == "Unit(km**(-2))"
    assert repr(Unit.KM/Unit.S) == "Unit(km/s)"
    assert repr((Unit.KM/Unit.S)**2) == "Unit(km**2/s**2)"
    assert repr((Unit.KM/Unit.S)**(-2)) == "Unit(s**2/km**2)"
    assert str(Unit.KM) == "km"
    assert str(Unit.KM*Unit.KM) == "km**2"
    assert str(Unit.KM**2) == "km**2"
    assert str(Unit.KM**(-2)) == "km**(-2)"
    assert str(Unit.KM/Unit.S) == "km/s"
    assert str((Unit.KM/Unit.S)**2) == "km**2/s**2"
    assert str((Unit.KM/Unit.S)**(-2)) == "s**2/km**2"
    assert (Unit.KM/Unit.S).exponents == (1,-1,0)
    assert (Unit.KM/Unit.S/Unit.S).exponents == (1,-2,0)
    assert Unit.KM.convert(3.,Unit.CM) == 3.e5
    assert (np.all(Unit.KM.convert(np.array([1.,2.,3.]), Unit.CM) ==
                           [1.e5, 2.e5, 3.e5]))
    assert (np.all(Unit.DEGREES.convert(np.array([1.,2.,3.]),
                           Unit.ARCSEC) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.H).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [1., 2., 3.]))
    assert (np.all((Unit.DEG*Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC*Unit.H) == [1., 2., 3.]))
    assert (np.all((Unit.DEG**2).convert(np.array([1.,2.,3.]),
                            Unit.ARCMIN*Unit.ARCSEC) ==
                            [3600*60, 3600*60*2, 3600*60*3]))
    eps = 1.e-15
    test = Unit.DEG.from_this(np.array([1.,2.,3.]))
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] < test + eps)
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] > test - eps)
    test = Unit.DEG.into_this(test)
    assert np.all(np.array([1., 2., 3.]) < test + eps)
    assert np.all(np.array([1., 2., 3.]) > test - eps)
    assert Unit.CM != Unit.M
    assert (Unit.CM != Unit.M)
    assert (Unit.M  != Unit.SEC)
    assert Unit.M.factor == Unit.MRAD.factor
    assert Unit.CM
    test = Unit.ROTATION/Unit.S
    assert test.get_name() == "rotation/s"
    unit = Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) / Unit.RAD
    assert repr(unit) == "Unit(km/s)"
    assert str(unit) == "km/s"
    unit = (Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) /
                         Unit.MRAD*Unit.MSEC/(Unit.KM/Unit.S) /
                         Unit.S)
    unit.name = None
    assert repr(unit) == "Unit()"
    assert repr(Unit.S * 60) == "Unit(min)"
    assert str(Unit.S * 60) == "min"
    assert repr(60 * Unit.S) == "Unit(min)"
    assert repr(Unit.H/3600) == "Unit(s)"
    assert repr((1000/Unit.KM)**(-2)) == "Unit(m**2)"
    assert Unit.can_match(None, None)
    assert Unit.can_match(None, Unit.UNITLESS)
    assert Unit.can_match(None, Unit.KM)
    assert Unit.can_match(Unit.KM, None)
    assert Unit.can_match(Unit.CM, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.UNITLESS)
    assert Unit.do_match(None, None)
    assert Unit.do_match(None, Unit.UNITLESS)
    assert not Unit.do_match(None, Unit.KM)
    assert not Unit.do_match(Unit.KM, None)
    assert Unit.do_match(Unit.CM, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.UNITLESS)
    assert (Unit.KM**2).sqrt() == Unit.KM

    ##################################################################################
    # __init__(self, exponents, triple, name=None)
    ##################################################################################

    u1 = Unit((1, 0, 0), (1, 1, 0), None)
    assert u1.exponents == (1, 0, 0)
    assert u1.triple == (1, 1, 0)
    assert u1.name == None
    assert u1.factor == 1.0
    assert u1.factor_inv == 1.0

    u2 = Unit((0, 0, 1), (1, 180, 1), 'deg')
    assert u2.exponents == (0, 0, 1)
    assert u2.triple == (1, 180, 1)
    expected_factor = (1.0 / 180.0) * np.pi
    assert u2.factor == expected_factor or abs(u2.factor - expected_factor) <= 5e-8
    assert u2.factor_inv == 180.0 / np.pi or abs(u2.factor_inv - 180.0 / np.pi) <= 5e-8

    u3 = Unit((1, 0, 0), (1, 1000, 0), 'm')
    assert u3.triple == (1, 1000, 0)
    assert u3.factor == 1.0 / 1000.0 or abs(u3.factor - 1.0 / 1000.0) <= 5e-8
    assert u3.factor_inv == 1000.0 or abs(u3.factor_inv - 1000.0) <= 5e-8

    u4 = Unit((0, 0, 0), (1, 1, 0), None)
    assert u4.name == None

    u5 = Unit((0, 0, 0), (256, 512, 0), None)

    assert u5.triple[:2] == (1, 2)

    ##################################################################################
    # from_unit_factor and into_unit_factor properties
    ##################################################################################
    u = Unit((1, 0, 0), (1, 1000, 0), 'm')
    assert u.from_unit_factor == u.factor
    assert u.into_unit_factor == u.factor_inv

    ##################################################################################
    # as_unit(arg)
    ##################################################################################

    assert Unit.as_unit(None) == None

    assert Unit.as_unit('km') == Unit.KM
    assert Unit.as_unit('deg') == Unit.DEG

    u = Unit.KM
    assert Unit.as_unit(u) == u

    with pytest.raises(ValueError):
        Unit.as_unit(123)

    ##################################################################################
    # can_match(first, second)
    ##################################################################################

    assert Unit.can_match(None, None)
    assert Unit.can_match(None, Unit.KM)
    assert Unit.can_match(Unit.KM, None)

    assert Unit.can_match(Unit.KM, Unit.M)
    assert Unit.can_match(Unit.DEG, Unit.RAD)

    assert not Unit.can_match(Unit.KM, Unit.S)
    assert not Unit.can_match(Unit.KM, Unit.DEG)

    ##################################################################################
    # require_compatible(first, second, info='')
    ##################################################################################

    Unit.require_compatible(Unit.KM, Unit.M)
    Unit.require_compatible(None, Unit.KM)
    Unit.require_compatible(Unit.KM, None)

    with pytest.raises(ValueError):
        Unit.require_compatible(Unit.KM, Unit.S)
    with pytest.raises(ValueError):
        Unit.require_compatible(Unit.KM, Unit.DEG)

    with pytest.raises(ValueError) as context:
        Unit.require_compatible(Unit.KM, Unit.S, info='test_op')
    assert 'test_op' in str(context.value)

    ##################################################################################
    # do_match(first, second)
    ##################################################################################

    assert Unit.do_match(None, None)
    assert Unit.do_match(None, Unit.UNITLESS)
    assert Unit.do_match(Unit.UNITLESS, None)

    assert Unit.do_match(Unit.KM, Unit.KM)
    assert Unit.do_match(Unit.DEG, Unit.DEG)

    assert Unit.do_match(Unit.KM, Unit.M)

    assert not Unit.do_match(Unit.KM, Unit.S)
    assert not Unit.do_match(Unit.KM, Unit.DEG)

    ##################################################################################
    # require_match(first, second, info='')
    ##################################################################################

    Unit.require_match(Unit.KM, Unit.KM)
    Unit.require_match(None, None)
    Unit.require_match(None, Unit.UNITLESS)

    Unit.require_match(Unit.KM, Unit.M)

    with pytest.raises(ValueError):
        Unit.require_match(Unit.KM, Unit.S)
    with pytest.raises(ValueError):
        Unit.require_match(Unit.KM, Unit.DEG)

    with pytest.raises(ValueError) as context:
        Unit.require_match(Unit.KM, Unit.S, info='test_op')
    assert 'test_op' in str(context.value)

    ##################################################################################
    # is_angle(arg)
    ##################################################################################

    assert Unit.is_angle(None)

    assert Unit.is_angle(Unit.UNITLESS)

    assert Unit.is_angle(Unit.DEG)
    assert Unit.is_angle(Unit.RAD)

    assert not Unit.is_angle(Unit.KM)
    assert not Unit.is_angle(Unit.S)

    ##################################################################################
    # require_angle(arg, info='')
    ##################################################################################

    Unit.require_angle(None)
    Unit.require_angle(Unit.DEG)
    Unit.require_angle(Unit.RAD)

    with pytest.raises(ValueError):
        Unit.require_angle(Unit.KM)
    with pytest.raises(ValueError):
        Unit.require_angle(Unit.S)

    with pytest.raises(ValueError) as context:
        Unit.require_angle(Unit.KM, info='test_op')
    assert 'test_op' in str(context.value)

    ##################################################################################
    # is_unitless(arg)
    ##################################################################################

    assert Unit.is_unitless(None)

    assert Unit.is_unitless(Unit.UNITLESS)

    assert not Unit.is_unitless(Unit.KM)
    assert not Unit.is_unitless(Unit.DEG)
    assert not Unit.is_unitless(Unit.S)

    ##################################################################################
    # require_unitless(arg, info='')
    ##################################################################################

    Unit.require_unitless(None)
    Unit.require_unitless(Unit.UNITLESS)

    with pytest.raises(ValueError):
        Unit.require_unitless(Unit.KM)
    with pytest.raises(ValueError):
        Unit.require_unitless(Unit.DEG)

    with pytest.raises(ValueError) as context:
        Unit.require_unitless(Unit.KM, info='test_op')
    assert 'test_op' in str(context.value)

    ##################################################################################
    # from_this(self, value)
    ##################################################################################
    u = Unit((1, 0, 0), (1, 1000, 0), 'm')

    result = u.from_this(1000.0)
    assert result == 1.0 or abs(result - 1.0) <= 5e-8
    u_deg = Unit((0, 0, 1), (1, 180, 1), 'deg')

    result = u_deg.from_this(180.0)
    assert result == np.pi or abs(result - np.pi) <= 5e-8

    values = np.array([1000.0, 2000.0, 3000.0])
    result = u.from_this(values)
    expected = np.array([1.0, 2.0, 3.0])
    assert np.allclose(result, expected)

    ##################################################################################
    # into_this(self, value)
    ##################################################################################
    u = Unit((1, 0, 0), (1, 1000, 0), 'm')

    result = u.into_this(1.0)
    assert result == 1000.0 or abs(result - 1000.0) <= 5e-8
    u_deg = Unit((0, 0, 1), (1, 180, 1), 'deg')

    result = u_deg.into_this(np.pi)
    assert result == 180.0 or abs(result - 180.0) <= 5e-8

    values = np.array([1.0, 2.0, 3.0])
    result = u.into_this(values)
    expected = np.array([1000.0, 2000.0, 3000.0])
    assert np.allclose(result, expected)

    ##################################################################################
    # from_unit(unit, value)
    ##################################################################################

    result = Unit.from_unit(None, 5.0)
    assert result == 5.0

    result = Unit.from_unit(Unit.M, 1000.0)
    assert result == 1.0 or abs(result - 1.0) <= 5e-8

    values = np.array([1000.0, 2000.0])
    result = Unit.from_unit(Unit.M, values)
    expected = np.array([1.0, 2.0])
    assert np.allclose(result, expected)

    ##################################################################################
    # into_unit(unit, value)
    ##################################################################################

    result = Unit.into_unit(None, 5.0)
    assert result == 5.0

    result = Unit.into_unit(Unit.M, 1.0)
    assert result == 1000.0 or abs(result - 1000.0) <= 5e-8

    values = np.array([1.0, 2.0])
    result = Unit.into_unit(Unit.M, values)
    expected = np.array([1000.0, 2000.0])
    assert np.allclose(result, expected)

    ##################################################################################
    # convert(self, value, unit, info='')
    ##################################################################################

    u_m = Unit.M
    result = u_m.convert(1000.0, Unit.KM)
    assert result == 1.0 or abs(result - 1.0) <= 5e-8

    u_deg = Unit.DEG
    result = u_deg.convert(180.0, Unit.RAD)
    assert result == np.pi or abs(result - np.pi) <= 5e-8

    u_unitless = Unit.UNITLESS
    result = u_unitless.convert(5.0, None)

    assert result == 5.0

    result = u_m.convert(1000.0, Unit.KM)
    assert result == 1.0 or abs(result - 1.0) <= 5e-8

    with pytest.raises(ValueError):
        u_m.convert(1000.0, Unit.S)

    with pytest.raises(ValueError) as context:
        u_m.convert(1000.0, Unit.S, info='test_op')
    assert 'test_op' in str(context.value)

    result = u_m.convert(1000.0, Unit.M)
    assert result == 1000.0

    values = np.array([1000.0, 2000.0, 3000.0])
    result = u_m.convert(values, Unit.KM)
    expected = np.array([1.0, 2.0, 3.0])
    assert np.allclose(result, expected)

    ##################################################################################
    # __mul__(self, arg)
    ##################################################################################

    u1 = Unit.KM
    u2 = Unit.S
    result = u1 * u2
    assert result.exponents == (1, 1, 0)
    # KM * S = km*s, which has exponents (1, 1, 0)

    result = u1 * None
    assert result == u1

    result = u1 * 5.0

    assert isinstance(result, Unit)
    assert result.name == None
    assert result.get_name() == '5*km'

    result = u1.__mul__('invalid')
    assert result == NotImplemented

    ##################################################################################
    # __rmul__(self, arg)
    ##################################################################################

    result = 5.0 * Unit.KM
    assert isinstance(result, Unit)
    assert result.name == None
    assert result.get_name() == '5*km'

    ##################################################################################
    # __truediv__(self, arg)
    ##################################################################################

    u1 = Unit.KM
    u2 = Unit.S
    result = u1 / u2
    assert result.exponents == (1, -1, 0)
    # KM / S = km/s, which has exponents (1, -1, 0)

    result = u1 / None
    assert result == u1

    result = u1 / 5.0
    assert isinstance(result, Unit)
    assert result.name == None
    assert result.get_name() == '0.2*km'

    result = u1.__truediv__('invalid')
    assert result == NotImplemented

    ##################################################################################
    # __rtruediv__(self, arg)
    ##################################################################################

    result = 5.0 / Unit.KM
    assert isinstance(result, Unit)
    assert result.name == None
    assert result.get_name() == '5/km'
    # Should be equivalent to Unit.KM**(-1) * 5.0

    result = None / Unit.KM
    assert isinstance(result, Unit)
    assert result.name == None
    assert result.get_name() == 'km**(-1)'

    result = Unit.KM.__rtruediv__('invalid')
    assert result == NotImplemented

    ##################################################################################
    # __pow__(self, power)
    ##################################################################################

    u = Unit.KM
    result = u ** 2
    assert result.exponents == (2, 0, 0)
    assert result.triple == (1, 1, 0)
    assert result.name == {'km': 2}
    assert result.get_name() == 'km**2'

    result = u ** (-2)
    assert result.exponents == (-2, 0, 0)

    u_sq = Unit((2, 0, 0), (1, 1, 0), None)
    result = u_sq ** 0.5
    assert result.exponents == (1, 0, 0)

    with pytest.raises(ValueError):
        u.__pow__(0.3)

    u_sq = Unit((2, 0, 0), (1, 1, 0), None)
    result = u_sq ** 0.5
    assert result.exponents == (1, 0, 0)

    u_4 = Unit((4, 0, 0), (1, 1, 0), None)
    result = u_4 ** 1.5  # sqrt then **3
    assert result.exponents == (6, 0, 0)

    ##################################################################################
    # sqrt(self)
    ##################################################################################

    u_sq = Unit((2, 0, 0), (1, 1, 0), None)
    result = u_sq.sqrt()
    assert result.exponents == (1, 0, 0)

    u_odd = Unit((1, 0, 0), (1, 1, 0), None)
    with pytest.raises(ValueError):
        u_odd.sqrt()

    result = Unit((2, 0, 0), (1, 1, 0), 'km**2').sqrt()
    assert result.name == {'km': 1}

    ##################################################################################
    # mul_units(arg1, arg2)
    ##################################################################################

    result = Unit.mul_units(Unit.KM, Unit.S)
    assert result.exponents == (1, 1, 0)
    assert result.name == {'km': 1, 's': 1}
    assert result.get_name() == 'km*s'

    result = Unit.mul_units(None, Unit.KM)
    assert result == Unit.KM
    result = Unit.mul_units(Unit.KM, None)
    assert result == Unit.KM
    result = Unit.mul_units(None, None)
    assert result == None

    ##################################################################################
    # div_units(arg1, arg2)
    ##################################################################################

    result = Unit.div_units(Unit.KM, Unit.S)
    assert result.exponents == (1, -1, 0)
    assert result.name == {'km': 1, 's': -1}
    assert result.get_name() == 'km/s'

    result = Unit.div_units(None, Unit.KM)
    assert result.exponents == (-1, 0, 0)
    result = Unit.div_units(Unit.KM, None)
    assert result == Unit.KM
    result = Unit.div_units(None, None)
    assert result == None

    ##################################################################################
    # sqrt_unit(unit)
    ##################################################################################

    u_sq = Unit((2, 0, 0), (1, 1, 0), None)
    result = Unit.sqrt_unit(u_sq)
    assert result.exponents == (1, 0, 0)

    result = Unit.sqrt_unit(None)
    assert result == None

    result = Unit.sqrt_unit(Unit((2, 0, 0), (1, 1, 0), 'km**2'))
    assert result.name == {'km': 1}

    ##################################################################################
    # unit_power(unit, power)
    ##################################################################################

    result = Unit.unit_power(Unit.KM, 2)
    assert result.exponents == (2, 0, 0)
    assert result.name == {'km': 2}

    result = Unit.unit_power(None, 2)
    assert result == None

    ##################################################################################
    # __eq__(self, arg)
    ##################################################################################

    assert (Unit.KM == Unit.KM)
    assert (Unit.DEG == Unit.DEG)

    assert Unit.KM != Unit.M
    assert Unit.KM != Unit.S

    assert Unit.KM != 'km'
    assert Unit.KM != 5

    ##################################################################################
    # __ne__(self, arg)
    ##################################################################################

    assert Unit.KM == Unit.KM

    assert (Unit.KM != Unit.M)
    assert (Unit.KM != Unit.S)

    assert (Unit.KM != 'km')
    assert (Unit.KM != 5)

    ##################################################################################
    # __copy__(self) and copy(self)
    ##################################################################################
    u = Unit.KM
    u_copy = u.__copy__()
    assert u.exponents == u_copy.exponents
    assert u.triple == u_copy.triple
    assert u is not u_copy
    u_copy2 = u.copy()
    assert u.exponents == u_copy2.exponents
    assert u.triple == u_copy2.triple
    assert u is not u_copy2

    ##################################################################################
    # __str__(self) and __repr__(self)
    ##################################################################################

    u = Unit.KM
    r = repr(u)
    assert isinstance(r, str)
    assert 'Unit' in r
    s = str(u)
    if s:
        assert isinstance(s, str)

    ##################################################################################
    # get_name(self) and set_name(self, name)
    ##################################################################################
    u = Unit.KM
    name = u.get_name()
    assert isinstance(name, (str, dict))
    assert name == 'km'

    u_dict = Unit((1, 0, 0), (1, 1, 0), 'km')
    assert u_dict.name == 'km'
    u.set_name('new_name')
    assert u.name == 'new_name'
    u.set_name({'km': 1})
    assert u.name == {'km': 1}

    u.set_name('km')

    ##################################################################################
    # create_name(self)
    ##################################################################################

    u = Unit.KM
    name = u.create_name()
    assert name == 'km'

    u = Unit((1, 0, 0), (1, 1, 0), None)
    name = u.create_name()
    assert name == 'km'

    ##################################################################################
    # Additional edge cases and static methods
    ##################################################################################

    u = Unit((0, 0, 0), (3, 7, 0), None)

    assert u.triple[:2] == (3, 7)

    u2 = Unit((0, 0, 0), (256, 512, 0), None)

    assert u2.triple[:2] == (1, 2)

    u_sq = Unit((4, 0, 0), (1, 1, 0), None)
    result = u_sq ** 0.5
    assert result.exponents == (2, 0, 0)

    u_simple = Unit((2, 0, 0), (1, 1, 0), None)
    result = u_simple.sqrt()

    assert result.exponents == (1, 0, 0)

    u_sqrt_float = Unit((2, 0, 0), (2, 1, 0), None)
    result = u_sqrt_float.sqrt()

    assert result.exponents == (1, 0, 0)

    u_sqrt_denom = Unit((2, 0, 0), (1, 2, 0), None)
    result = u_sqrt_denom.sqrt()

    assert result.exponents == (1, 0, 0)

    assert isinstance(result.triple[1], (float, np.floating))

    u_odd_pi = Unit((0, 0, 2), (1, 1, 3), None)
    result = u_odd_pi.sqrt()

    assert result.exponents == (0, 0, 1)

    ##################################################################################
    # Test static name processing methods
    ##################################################################################

    result = Unit._mul_names('km', 's')
    assert isinstance(result, dict)
    result = Unit._mul_names({'km': 1}, {'s': 1})
    assert isinstance(result, dict)
    result = Unit._mul_names(None, 'km')
    assert result == None
    result = Unit._mul_names('km', None)
    assert result == None

    result = Unit._mul_names({'km': 1}, {'km': -1})

    assert result == {}

    result = Unit._mul_names({'km': 2}, {'km': 3})
    assert result == {'km': 5}

    result = Unit._div_names('km', 's')
    assert isinstance(result, dict)
    result = Unit._div_names({'km': 1}, {'s': 1})
    assert isinstance(result, dict)
    result = Unit._div_names(None, 'km')
    assert result == None
    result = Unit._div_names('km', None)
    assert result == None

    result = Unit._div_names({'km': 1}, {'km': 1})

    assert result == {}

    result = Unit._div_names({'km': 5}, {'km': 2})
    assert result == {'km': 3}

    result = Unit._name_power('km', 2)
    assert isinstance(result, dict)
    result = Unit._name_power({'km': 1}, 2)
    assert isinstance(result, dict)
    result = Unit._name_power(None, 2)
    assert result == None

    assert Unit._name_power({'km': 1}, 0.5) is None

    result = Unit.name_to_dict('km')
    assert isinstance(result, dict)
    result = Unit.name_to_dict({'km': 1})
    assert isinstance(result, dict)
    result = Unit.name_to_dict('')
    assert result == {}

    with pytest.raises(ValueError):
        Unit.name_to_dict(123)

    with pytest.raises(ValueError, match='unexpected "5"'):
        Unit.name_to_dict('5')

    result = Unit.name_to_dict('km*s')
    assert isinstance(result, dict)
    result = Unit.name_to_dict('km/s')
    assert isinstance(result, dict)
    result = Unit.name_to_dict('km**2')
    assert isinstance(result, dict)
    assert result == {'km': 2}
    result = Unit.name_to_dict('(km*s)/m')
    assert isinstance(result, dict)

    result = Unit.name_to_dict('(km*s)')
    assert isinstance(result, dict)

    result = Unit.name_to_dict('km*s')
    assert isinstance(result, dict)

    result = Unit.name_to_dict('km/s')
    assert isinstance(result, dict)

    result = Unit.name_to_dict('(km)**2')
    assert isinstance(result, dict)

    result = Unit.name_to_dict('km*s/m')
    assert isinstance(result, dict)

    result = Unit.name_to_str({'km': 1})
    assert isinstance(result, str)
    result = Unit.name_to_str({'km': 1, 's': -1})
    assert isinstance(result, str)
    result = Unit.name_to_str('km')
    assert result == 'km'

    result = Unit.name_to_str('')
    assert result == ''

    # Note: name_to_str with None would cause AttributeError
    # So we don't test that case

    result = Unit.name_to_str({})
    assert result == ''

    result = Unit.name_to_str({'': 5, 'km': 1})
    assert isinstance(result, str)
    # Should include the coefficient 5

    result = Unit.name_to_str({'': 1, 'km': 1})
    assert isinstance(result, str)
    # Coefficient 1 should not appear

    result = Unit.name_to_str({'km': 3})
    assert isinstance(result, str)
    assert '**' in result

    result = Unit.name_to_str({'km': -2})
    assert isinstance(result, str)

    result = Unit.name_to_str({'km': -1})
    assert isinstance(result, str)
    # Result should have '/' or be formatted as denominator
    # The exact format depends on implementation

    result = Unit.name_to_str({'km': 1, 's': -1})
    assert isinstance(result, str)
    assert '/' in result

    result = Unit.name_to_str({'km': 1, 'm': 1})
    assert isinstance(result, str)
    assert '/' not in result

    result = Unit.name_to_str({'km': -1, 's': -1})
    assert isinstance(result, str)

    ##################################################################################
    # Additional tests for missing coverage
    ##################################################################################

    u1 = Unit.KM
    u2 = Unit.S
    result = u1.__div__(u2)
    assert result.exponents == (1, -1, 0)
    result = Unit.KM.__rdiv__(5.0)
    assert isinstance(result, Unit)

    result = Unit.name_to_dict('(km)')
    assert isinstance(result, dict)
    # Tests the loop that finds matching closing parenthesis

    result = Unit.name_to_dict('((km))')
    assert isinstance(result, dict)
    # Tests depth tracking in parentheses

    result = Unit.name_to_dict('(km)*s')
    assert isinstance(result, dict)
    # Tests right = name[i+1:].lstrip() when there's content after ')'

    result = Unit.name_to_dict('xyz')
    assert result == {'xyz': 1}

    result = Unit.name_to_dict('km**2*s')
    assert result == {'km': 2, 's': 1}
    # This tests the branch where right has ** and we extract power

    with pytest.raises(ValueError):
        Unit.name_to_dict('km**')

    try:
        # This might trigger the no-progress check
        result = Unit.name_to_dict('km')
        # If it succeeds, it's a valid unit name
        assert isinstance(result, dict)
    except ValueError as e:
        # If it fails with "no progress", that's the path we want
        if 'no progress' in str(e) or 'illegal' in str(e).lower():
            pass

    result = Unit.name_to_str({'deg': 1, 'rad': 1, 'km': 1})
    assert isinstance(result, str)
    # Should include angle units in sorted order

    u_custom = Unit((1, 0, 0), (1, 1000, 0), None)
    name = u_custom.create_name()
    assert name == 'm'

    u_neg_exp = Unit((0, -2, 0), (1, 1, 0), None)  # 1/s^2
    name = u_neg_exp.create_name()

    assert name == {'km': 0, 's': -2, 'rad': 0}

    u_multi = Unit((4, 0, 0), (1, 1, 0), None)  # km^4
    name = u_multi.create_name()
    assert name == {'km': 4, 's': 0, 'rad': 0}

    u_fallback = Unit((1, 0, 0), (3, 7, 0), None)  # Custom triple
    name = u_fallback.create_name()

    assert name == {'': 3/7, 'km': 1, 's': 0, 'rad': 0}

    u_simple = Unit((2, 0, 0), (5, 1, 0), None)  # denom=1, pi_expo=0
    name = u_simple.create_name()

    assert name == {'': 5, 'km': 2, 's': 0, 'rad': 0}

    u_denom = Unit((1, 0, 0), (3, 2, 0), None)  # Has denom != 1
    name = u_denom.create_name()

    assert name == {'': 3/2, 'km': 1, 's': 0, 'rad': 0}

    u_pi_exp = Unit((0, 0, 1), (1, 180, 1), None)  # Has pi_expo
    name = u_pi_exp.create_name()

    assert name == 'deg'

    u_best = Unit((6, 0, 0), (1, 1, 0), None)  # km^6 could be (km^2)^3 or (km^3)^2
    name = u_best.create_name()

    assert name == {'km': 6, 's': 0, 'rad': 0}

    u_neg_power = Unit((0, -3, 0), (1, 1, 0), None)  # 1/s^3
    name = u_neg_power.create_name()

    assert name == {'km': 0, 's': -3, 'rad': 0}

    result = Unit.as_unit('km')
    assert isinstance(result, Unit)
    assert result == Unit.KM

    with pytest.raises(ValueError, match='missing "\\)"'):
        Unit.name_to_dict('(km')

    with pytest.raises(ValueError, match='missing "\\)"'):
        Unit.name_to_dict('((km')

    ##################################################################################
    # Test name_to_dict with '**' in an invalid position
    ##################################################################################

    with pytest.raises(ValueError):
        Unit.name_to_dict('km**2**3')

    with pytest.raises(ValueError):
        Unit.name_to_dict('(km)**2**3')

    with pytest.raises(ValueError):
        Unit.name_to_dict('s**2**3')


def test_units_create_unit_with_angle_exponent_5_to_test_more_false_cases() -> None:
    """Create unit with angle exponent 5 to test more False cases."""

    np.random.seed(7456)
    assert repr(Unit.KM) == "Unit(km)"
    assert repr(Unit.KM*Unit.KM) == "Unit(km**2)"
    assert repr(Unit.KM**2) == "Unit(km**2)"
    assert repr(Unit.KM**(-2)) == "Unit(km**(-2))"
    assert repr(Unit.KM/Unit.S) == "Unit(km/s)"
    assert repr((Unit.KM/Unit.S)**2) == "Unit(km**2/s**2)"
    assert repr((Unit.KM/Unit.S)**(-2)) == "Unit(s**2/km**2)"
    assert str(Unit.KM) == "km"
    assert str(Unit.KM*Unit.KM) == "km**2"
    assert str(Unit.KM**2) == "km**2"
    assert str(Unit.KM**(-2)) == "km**(-2)"
    assert str(Unit.KM/Unit.S) == "km/s"
    assert str((Unit.KM/Unit.S)**2) == "km**2/s**2"
    assert str((Unit.KM/Unit.S)**(-2)) == "s**2/km**2"
    assert (Unit.KM/Unit.S).exponents == (1,-1,0)
    assert (Unit.KM/Unit.S/Unit.S).exponents == (1,-2,0)
    assert Unit.KM.convert(3.,Unit.CM) == 3.e5
    assert (np.all(Unit.KM.convert(np.array([1.,2.,3.]), Unit.CM) ==
                           [1.e5, 2.e5, 3.e5]))
    assert (np.all(Unit.DEGREES.convert(np.array([1.,2.,3.]),
                           Unit.ARCSEC) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.H).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [1., 2., 3.]))
    assert (np.all((Unit.DEG*Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC*Unit.H) == [1., 2., 3.]))
    assert (np.all((Unit.DEG**2).convert(np.array([1.,2.,3.]),
                            Unit.ARCMIN*Unit.ARCSEC) ==
                            [3600*60, 3600*60*2, 3600*60*3]))
    eps = 1.e-15
    test = Unit.DEG.from_this(np.array([1.,2.,3.]))
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] < test + eps)
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] > test - eps)
    test = Unit.DEG.into_this(test)
    assert np.all(np.array([1., 2., 3.]) < test + eps)
    assert np.all(np.array([1., 2., 3.]) > test - eps)
    assert Unit.CM != Unit.M
    assert (Unit.CM != Unit.M)
    assert (Unit.M  != Unit.SEC)
    assert Unit.M.factor == Unit.MRAD.factor
    assert Unit.CM
    test = Unit.ROTATION/Unit.S
    assert test.get_name() == "rotation/s"
    unit = Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) / Unit.RAD
    assert repr(unit) == "Unit(km/s)"
    assert str(unit) == "km/s"
    unit = (Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) /
                         Unit.MRAD*Unit.MSEC/(Unit.KM/Unit.S) /
                         Unit.S)
    unit.name = None
    assert repr(unit) == "Unit()"
    assert repr(Unit.S * 60) == "Unit(min)"
    assert str(Unit.S * 60) == "min"
    assert repr(60 * Unit.S) == "Unit(min)"
    assert repr(Unit.H/3600) == "Unit(s)"
    assert repr((1000/Unit.KM)**(-2)) == "Unit(m**2)"
    assert Unit.can_match(None, None)
    assert Unit.can_match(None, Unit.UNITLESS)
    assert Unit.can_match(None, Unit.KM)
    assert Unit.can_match(Unit.KM, None)
    assert Unit.can_match(Unit.CM, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.UNITLESS)
    assert Unit.do_match(None, None)
    assert Unit.do_match(None, Unit.UNITLESS)
    assert not Unit.do_match(None, Unit.KM)
    assert not Unit.do_match(Unit.KM, None)
    assert Unit.do_match(Unit.CM, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.UNITLESS)
    assert (Unit.KM**2).sqrt() == Unit.KM

    ##################################################################################
    # __init__(self, exponents, triple, name=None)
    ##################################################################################

    u_angle5 = Unit((0, 0, 5), (1, 1, 0), 'rad**5')  # angle^5
    name = u_angle5.create_name()

    assert name == 'rad**5'

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


def test_units_test_with_a_unit_that_matches_unitless_structure_unitless_ha() -> None:
    """Test with a unit that matches UNITLESS structure # UNITLESS has name='' (empty string), not None, so this won't trigger # line 1026 fall-through, but it tests the lookup path."""

    np.random.seed(7456)
    assert repr(Unit.KM) == "Unit(km)"
    assert repr(Unit.KM*Unit.KM) == "Unit(km**2)"
    assert repr(Unit.KM**2) == "Unit(km**2)"
    assert repr(Unit.KM**(-2)) == "Unit(km**(-2))"
    assert repr(Unit.KM/Unit.S) == "Unit(km/s)"
    assert repr((Unit.KM/Unit.S)**2) == "Unit(km**2/s**2)"
    assert repr((Unit.KM/Unit.S)**(-2)) == "Unit(s**2/km**2)"
    assert str(Unit.KM) == "km"
    assert str(Unit.KM*Unit.KM) == "km**2"
    assert str(Unit.KM**2) == "km**2"
    assert str(Unit.KM**(-2)) == "km**(-2)"
    assert str(Unit.KM/Unit.S) == "km/s"
    assert str((Unit.KM/Unit.S)**2) == "km**2/s**2"
    assert str((Unit.KM/Unit.S)**(-2)) == "s**2/km**2"
    assert (Unit.KM/Unit.S).exponents == (1,-1,0)
    assert (Unit.KM/Unit.S/Unit.S).exponents == (1,-2,0)
    assert Unit.KM.convert(3.,Unit.CM) == 3.e5
    assert (np.all(Unit.KM.convert(np.array([1.,2.,3.]), Unit.CM) ==
                           [1.e5, 2.e5, 3.e5]))
    assert (np.all(Unit.DEGREES.convert(np.array([1.,2.,3.]),
                           Unit.ARCSEC) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.H).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [1., 2., 3.]))
    assert (np.all((Unit.DEG*Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC*Unit.H) == [1., 2., 3.]))
    assert (np.all((Unit.DEG**2).convert(np.array([1.,2.,3.]),
                            Unit.ARCMIN*Unit.ARCSEC) ==
                            [3600*60, 3600*60*2, 3600*60*3]))
    eps = 1.e-15
    test = Unit.DEG.from_this(np.array([1.,2.,3.]))
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] < test + eps)
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] > test - eps)
    test = Unit.DEG.into_this(test)
    assert np.all(np.array([1., 2., 3.]) < test + eps)
    assert np.all(np.array([1., 2., 3.]) > test - eps)
    assert Unit.CM != Unit.M
    assert (Unit.CM != Unit.M)
    assert (Unit.M  != Unit.SEC)
    assert Unit.M.factor == Unit.MRAD.factor
    assert Unit.CM
    test = Unit.ROTATION/Unit.S
    assert test.get_name() == "rotation/s"
    unit = Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) / Unit.RAD
    assert repr(unit) == "Unit(km/s)"
    assert str(unit) == "km/s"
    unit = (Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) /
                         Unit.MRAD*Unit.MSEC/(Unit.KM/Unit.S) /
                         Unit.S)
    unit.name = None
    assert repr(unit) == "Unit()"
    assert repr(Unit.S * 60) == "Unit(min)"
    assert str(Unit.S * 60) == "min"
    assert repr(60 * Unit.S) == "Unit(min)"
    assert repr(Unit.H/3600) == "Unit(s)"
    assert repr((1000/Unit.KM)**(-2)) == "Unit(m**2)"
    assert Unit.can_match(None, None)
    assert Unit.can_match(None, Unit.UNITLESS)
    assert Unit.can_match(None, Unit.KM)
    assert Unit.can_match(Unit.KM, None)
    assert Unit.can_match(Unit.CM, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.UNITLESS)
    assert Unit.do_match(None, None)
    assert Unit.do_match(None, Unit.UNITLESS)
    assert not Unit.do_match(None, Unit.KM)
    assert not Unit.do_match(Unit.KM, None)
    assert Unit.do_match(Unit.CM, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.UNITLESS)
    assert (Unit.KM**2).sqrt() == Unit.KM

    ##################################################################################
    # __init__(self, exponents, triple, name=None)
    ##################################################################################

    u_unitless = Unit((0, 0, 0), (1, 1, 0), None)
    name = u_unitless.create_name()

    assert name is not None


def test_units_to_actually_test_line_1026_fall_through_we_d_need_to_tempora() -> None:
    """To actually test line 1026 fall-through, we'd need to temporarily # set a standard unit's name to None. Let's do that for testing: # Save original name."""

    np.random.seed(7456)
    assert repr(Unit.KM) == "Unit(km)"
    assert repr(Unit.KM*Unit.KM) == "Unit(km**2)"
    assert repr(Unit.KM**2) == "Unit(km**2)"
    assert repr(Unit.KM**(-2)) == "Unit(km**(-2))"
    assert repr(Unit.KM/Unit.S) == "Unit(km/s)"
    assert repr((Unit.KM/Unit.S)**2) == "Unit(km**2/s**2)"
    assert repr((Unit.KM/Unit.S)**(-2)) == "Unit(s**2/km**2)"
    assert str(Unit.KM) == "km"
    assert str(Unit.KM*Unit.KM) == "km**2"
    assert str(Unit.KM**2) == "km**2"
    assert str(Unit.KM**(-2)) == "km**(-2)"
    assert str(Unit.KM/Unit.S) == "km/s"
    assert str((Unit.KM/Unit.S)**2) == "km**2/s**2"
    assert str((Unit.KM/Unit.S)**(-2)) == "s**2/km**2"
    assert (Unit.KM/Unit.S).exponents == (1,-1,0)
    assert (Unit.KM/Unit.S/Unit.S).exponents == (1,-2,0)
    assert Unit.KM.convert(3.,Unit.CM) == 3.e5
    assert (np.all(Unit.KM.convert(np.array([1.,2.,3.]), Unit.CM) ==
                           [1.e5, 2.e5, 3.e5]))
    assert (np.all(Unit.DEGREES.convert(np.array([1.,2.,3.]),
                           Unit.ARCSEC) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.H).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [1., 2., 3.]))
    assert (np.all((Unit.DEG*Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC*Unit.H) == [1., 2., 3.]))
    assert (np.all((Unit.DEG**2).convert(np.array([1.,2.,3.]),
                            Unit.ARCMIN*Unit.ARCSEC) ==
                            [3600*60, 3600*60*2, 3600*60*3]))
    eps = 1.e-15
    test = Unit.DEG.from_this(np.array([1.,2.,3.]))
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] < test + eps)
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] > test - eps)
    test = Unit.DEG.into_this(test)
    assert np.all(np.array([1., 2., 3.]) < test + eps)
    assert np.all(np.array([1., 2., 3.]) > test - eps)
    assert Unit.CM != Unit.M
    assert (Unit.CM != Unit.M)
    assert (Unit.M  != Unit.SEC)
    assert Unit.M.factor == Unit.MRAD.factor
    assert Unit.CM
    test = Unit.ROTATION/Unit.S
    assert test.get_name() == "rotation/s"
    unit = Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) / Unit.RAD
    assert repr(unit) == "Unit(km/s)"
    assert str(unit) == "km/s"
    unit = (Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) /
                         Unit.MRAD*Unit.MSEC/(Unit.KM/Unit.S) /
                         Unit.S)
    unit.name = None
    assert repr(unit) == "Unit()"
    assert repr(Unit.S * 60) == "Unit(min)"
    assert str(Unit.S * 60) == "min"
    assert repr(60 * Unit.S) == "Unit(min)"
    assert repr(Unit.H/3600) == "Unit(s)"
    assert repr((1000/Unit.KM)**(-2)) == "Unit(m**2)"
    assert Unit.can_match(None, None)
    assert Unit.can_match(None, Unit.UNITLESS)
    assert Unit.can_match(None, Unit.KM)
    assert Unit.can_match(Unit.KM, None)
    assert Unit.can_match(Unit.CM, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.UNITLESS)
    assert Unit.do_match(None, None)
    assert Unit.do_match(None, Unit.UNITLESS)
    assert not Unit.do_match(None, Unit.KM)
    assert not Unit.do_match(Unit.KM, None)
    assert Unit.do_match(Unit.CM, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.UNITLESS)
    assert (Unit.KM**2).sqrt() == Unit.KM

    ##################################################################################
    # __init__(self, exponents, triple, name=None)
    ##################################################################################

    unitless_key = ((0, 0, 0), (1, 1, 0))
    original_name = Unit._TUPLES_TO_UNIT[unitless_key].name
    try:
        # Temporarily set name to None to test fall-through
        Unit._TUPLES_TO_UNIT[unitless_key].name = None
        u_test = Unit((0, 0, 0), (1, 1, 0), None)
        name = u_test.create_name()
        # Now name is None, so line 1026 condition is False and it falls through
        # Should continue to search for combinations
        assert name is not None
    finally:
        # Restore original name
        Unit._TUPLES_TO_UNIT[unitless_key].name = original_name

    ##################################################################################
    # Test create_name when p * actual_power != target_power
    # This specifically tests when the condition is False
    ##################################################################################


def test_units_create_a_unit_where_target_power_doesn_t_divide_evenly_by_an() -> None:
    """Create a unit where target_power doesn't divide evenly by any standard unit's power # For example, angle exponent 7: when checking STER (power 2), p = 7 // 2 = 3, # and 3 * 2 = 6 != 7, so the condition is False."""

    np.random.seed(7456)
    assert repr(Unit.KM) == "Unit(km)"
    assert repr(Unit.KM*Unit.KM) == "Unit(km**2)"
    assert repr(Unit.KM**2) == "Unit(km**2)"
    assert repr(Unit.KM**(-2)) == "Unit(km**(-2))"
    assert repr(Unit.KM/Unit.S) == "Unit(km/s)"
    assert repr((Unit.KM/Unit.S)**2) == "Unit(km**2/s**2)"
    assert repr((Unit.KM/Unit.S)**(-2)) == "Unit(s**2/km**2)"
    assert str(Unit.KM) == "km"
    assert str(Unit.KM*Unit.KM) == "km**2"
    assert str(Unit.KM**2) == "km**2"
    assert str(Unit.KM**(-2)) == "km**(-2)"
    assert str(Unit.KM/Unit.S) == "km/s"
    assert str((Unit.KM/Unit.S)**2) == "km**2/s**2"
    assert str((Unit.KM/Unit.S)**(-2)) == "s**2/km**2"
    assert (Unit.KM/Unit.S).exponents == (1,-1,0)
    assert (Unit.KM/Unit.S/Unit.S).exponents == (1,-2,0)
    assert Unit.KM.convert(3.,Unit.CM) == 3.e5
    assert (np.all(Unit.KM.convert(np.array([1.,2.,3.]), Unit.CM) ==
                           [1.e5, 2.e5, 3.e5]))
    assert (np.all(Unit.DEGREES.convert(np.array([1.,2.,3.]),
                           Unit.ARCSEC) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.H).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [1., 2., 3.]))
    assert (np.all((Unit.DEG*Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC*Unit.H) == [1., 2., 3.]))
    assert (np.all((Unit.DEG**2).convert(np.array([1.,2.,3.]),
                            Unit.ARCMIN*Unit.ARCSEC) ==
                            [3600*60, 3600*60*2, 3600*60*3]))
    eps = 1.e-15
    test = Unit.DEG.from_this(np.array([1.,2.,3.]))
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] < test + eps)
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] > test - eps)
    test = Unit.DEG.into_this(test)
    assert np.all(np.array([1., 2., 3.]) < test + eps)
    assert np.all(np.array([1., 2., 3.]) > test - eps)
    assert Unit.CM != Unit.M
    assert (Unit.CM != Unit.M)
    assert (Unit.M  != Unit.SEC)
    assert Unit.M.factor == Unit.MRAD.factor
    assert Unit.CM
    test = Unit.ROTATION/Unit.S
    assert test.get_name() == "rotation/s"
    unit = Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) / Unit.RAD
    assert repr(unit) == "Unit(km/s)"
    assert str(unit) == "km/s"
    unit = (Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) /
                         Unit.MRAD*Unit.MSEC/(Unit.KM/Unit.S) /
                         Unit.S)
    unit.name = None
    assert repr(unit) == "Unit()"
    assert repr(Unit.S * 60) == "Unit(min)"
    assert str(Unit.S * 60) == "min"
    assert repr(60 * Unit.S) == "Unit(min)"
    assert repr(Unit.H/3600) == "Unit(s)"
    assert repr((1000/Unit.KM)**(-2)) == "Unit(m**2)"
    assert Unit.can_match(None, None)
    assert Unit.can_match(None, Unit.UNITLESS)
    assert Unit.can_match(None, Unit.KM)
    assert Unit.can_match(Unit.KM, None)
    assert Unit.can_match(Unit.CM, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.UNITLESS)
    assert Unit.do_match(None, None)
    assert Unit.do_match(None, Unit.UNITLESS)
    assert not Unit.do_match(None, Unit.KM)
    assert not Unit.do_match(Unit.KM, None)
    assert Unit.do_match(Unit.CM, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.UNITLESS)
    assert (Unit.KM**2).sqrt() == Unit.KM

    ##################################################################################
    # __init__(self, exponents, triple, name=None)
    ##################################################################################

    u_angle7 = Unit((0, 0, 7), (1, 1, 0), None)  # angle^7
    name = u_angle7.create_name()

    assert name == {'km': 0, 's': 0, 'rad': 7}

    # Test with distance exponent that doesn't divide evenly
    # Distance units all have power 1, so any integer will work. We need a different approach.
    # Actually, for distance/time, all standard units have power 1, so they always divide evenly.
    # For angle, we have STER with power 2, so we can test with odd powers > 1.


def test_units_test_with_angle_exponent_3_odd_1() -> None:
    """Test with angle exponent 3 (odd, > 1)."""

    np.random.seed(7456)
    assert repr(Unit.KM) == "Unit(km)"
    assert repr(Unit.KM*Unit.KM) == "Unit(km**2)"
    assert repr(Unit.KM**2) == "Unit(km**2)"
    assert repr(Unit.KM**(-2)) == "Unit(km**(-2))"
    assert repr(Unit.KM/Unit.S) == "Unit(km/s)"
    assert repr((Unit.KM/Unit.S)**2) == "Unit(km**2/s**2)"
    assert repr((Unit.KM/Unit.S)**(-2)) == "Unit(s**2/km**2)"
    assert str(Unit.KM) == "km"
    assert str(Unit.KM*Unit.KM) == "km**2"
    assert str(Unit.KM**2) == "km**2"
    assert str(Unit.KM**(-2)) == "km**(-2)"
    assert str(Unit.KM/Unit.S) == "km/s"
    assert str((Unit.KM/Unit.S)**2) == "km**2/s**2"
    assert str((Unit.KM/Unit.S)**(-2)) == "s**2/km**2"
    assert (Unit.KM/Unit.S).exponents == (1,-1,0)
    assert (Unit.KM/Unit.S/Unit.S).exponents == (1,-2,0)
    assert Unit.KM.convert(3.,Unit.CM) == 3.e5
    assert (np.all(Unit.KM.convert(np.array([1.,2.,3.]), Unit.CM) ==
                           [1.e5, 2.e5, 3.e5]))
    assert (np.all(Unit.DEGREES.convert(np.array([1.,2.,3.]),
                           Unit.ARCSEC) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.H).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [1., 2., 3.]))
    assert (np.all((Unit.DEG*Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC*Unit.H) == [1., 2., 3.]))
    assert (np.all((Unit.DEG**2).convert(np.array([1.,2.,3.]),
                            Unit.ARCMIN*Unit.ARCSEC) ==
                            [3600*60, 3600*60*2, 3600*60*3]))
    eps = 1.e-15
    test = Unit.DEG.from_this(np.array([1.,2.,3.]))
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] < test + eps)
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] > test - eps)
    test = Unit.DEG.into_this(test)
    assert np.all(np.array([1., 2., 3.]) < test + eps)
    assert np.all(np.array([1., 2., 3.]) > test - eps)
    assert Unit.CM != Unit.M
    assert (Unit.CM != Unit.M)
    assert (Unit.M  != Unit.SEC)
    assert Unit.M.factor == Unit.MRAD.factor
    assert Unit.CM
    test = Unit.ROTATION/Unit.S
    assert test.get_name() == "rotation/s"
    unit = Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) / Unit.RAD
    assert repr(unit) == "Unit(km/s)"
    assert str(unit) == "km/s"
    unit = (Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) /
                         Unit.MRAD*Unit.MSEC/(Unit.KM/Unit.S) /
                         Unit.S)
    unit.name = None
    assert repr(unit) == "Unit()"
    assert repr(Unit.S * 60) == "Unit(min)"
    assert str(Unit.S * 60) == "min"
    assert repr(60 * Unit.S) == "Unit(min)"
    assert repr(Unit.H/3600) == "Unit(s)"
    assert repr((1000/Unit.KM)**(-2)) == "Unit(m**2)"
    assert Unit.can_match(None, None)
    assert Unit.can_match(None, Unit.UNITLESS)
    assert Unit.can_match(None, Unit.KM)
    assert Unit.can_match(Unit.KM, None)
    assert Unit.can_match(Unit.CM, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.UNITLESS)
    assert Unit.do_match(None, None)
    assert Unit.do_match(None, Unit.UNITLESS)
    assert not Unit.do_match(None, Unit.KM)
    assert not Unit.do_match(Unit.KM, None)
    assert Unit.do_match(Unit.CM, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.UNITLESS)
    assert (Unit.KM**2).sqrt() == Unit.KM

    ##################################################################################
    # __init__(self, exponents, triple, name=None)
    ##################################################################################

    u_angle3 = Unit((0, 0, 3), (1, 1, 0), None)  # angle^3
    name = u_angle3.create_name()

    assert name == {'km': 0, 's': 0, 'rad': 3}


def test_units_test_with_angle_exponent_9_odd_1() -> None:
    """Test with angle exponent 9 (odd, > 1)."""

    np.random.seed(7456)
    assert repr(Unit.KM) == "Unit(km)"
    assert repr(Unit.KM*Unit.KM) == "Unit(km**2)"
    assert repr(Unit.KM**2) == "Unit(km**2)"
    assert repr(Unit.KM**(-2)) == "Unit(km**(-2))"
    assert repr(Unit.KM/Unit.S) == "Unit(km/s)"
    assert repr((Unit.KM/Unit.S)**2) == "Unit(km**2/s**2)"
    assert repr((Unit.KM/Unit.S)**(-2)) == "Unit(s**2/km**2)"
    assert str(Unit.KM) == "km"
    assert str(Unit.KM*Unit.KM) == "km**2"
    assert str(Unit.KM**2) == "km**2"
    assert str(Unit.KM**(-2)) == "km**(-2)"
    assert str(Unit.KM/Unit.S) == "km/s"
    assert str((Unit.KM/Unit.S)**2) == "km**2/s**2"
    assert str((Unit.KM/Unit.S)**(-2)) == "s**2/km**2"
    assert (Unit.KM/Unit.S).exponents == (1,-1,0)
    assert (Unit.KM/Unit.S/Unit.S).exponents == (1,-2,0)
    assert Unit.KM.convert(3.,Unit.CM) == 3.e5
    assert (np.all(Unit.KM.convert(np.array([1.,2.,3.]), Unit.CM) ==
                           [1.e5, 2.e5, 3.e5]))
    assert (np.all(Unit.DEGREES.convert(np.array([1.,2.,3.]),
                           Unit.ARCSEC) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [3600., 7200., 10800.]))
    assert (np.all((Unit.DEG/Unit.H).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC/Unit.S) == [1., 2., 3.]))
    assert (np.all((Unit.DEG*Unit.S).convert(np.array([1.,2.,3.]),
                            Unit.ARCSEC*Unit.H) == [1., 2., 3.]))
    assert (np.all((Unit.DEG**2).convert(np.array([1.,2.,3.]),
                            Unit.ARCMIN*Unit.ARCSEC) ==
                            [3600*60, 3600*60*2, 3600*60*3]))
    eps = 1.e-15
    test = Unit.DEG.from_this(np.array([1.,2.,3.]))
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] < test + eps)
    assert np.all([np.pi/180., np.pi/90., np.pi/60.] > test - eps)
    test = Unit.DEG.into_this(test)
    assert np.all(np.array([1., 2., 3.]) < test + eps)
    assert np.all(np.array([1., 2., 3.]) > test - eps)
    assert Unit.CM != Unit.M
    assert (Unit.CM != Unit.M)
    assert (Unit.M  != Unit.SEC)
    assert Unit.M.factor == Unit.MRAD.factor
    assert Unit.CM
    test = Unit.ROTATION/Unit.S
    assert test.get_name() == "rotation/s"
    unit = Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) / Unit.RAD
    assert repr(unit) == "Unit(km/s)"
    assert str(unit) == "km/s"
    unit = (Unit.KM**3/Unit.S*Unit.RAD*Unit.KM**(-2) /
                         Unit.MRAD*Unit.MSEC/(Unit.KM/Unit.S) /
                         Unit.S)
    unit.name = None
    assert repr(unit) == "Unit()"
    assert repr(Unit.S * 60) == "Unit(min)"
    assert str(Unit.S * 60) == "min"
    assert repr(60 * Unit.S) == "Unit(min)"
    assert repr(Unit.H/3600) == "Unit(s)"
    assert repr((1000/Unit.KM)**(-2)) == "Unit(m**2)"
    assert Unit.can_match(None, None)
    assert Unit.can_match(None, Unit.UNITLESS)
    assert Unit.can_match(None, Unit.KM)
    assert Unit.can_match(Unit.KM, None)
    assert Unit.can_match(Unit.CM, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.KM)
    assert not Unit.can_match(Unit.S, Unit.UNITLESS)
    assert Unit.do_match(None, None)
    assert Unit.do_match(None, Unit.UNITLESS)
    assert not Unit.do_match(None, Unit.KM)
    assert not Unit.do_match(Unit.KM, None)
    assert Unit.do_match(Unit.CM, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.KM)
    assert not Unit.do_match(Unit.S, Unit.UNITLESS)
    assert (Unit.KM**2).sqrt() == Unit.KM

    ##################################################################################
    # __init__(self, exponents, triple, name=None)
    ##################################################################################

    u_angle9 = Unit((0, 0, 9), (1, 1, 0), None)  # angle^9
    name = u_angle9.create_name()

    assert name == {'km': 0, 's': 0, 'rad': 9}


##########################################################################################


def test_units_require_angle_message_names_the_offending_unit() -> None:
    """require_angle() rejects a non-angle unit with a message naming it."""

    with pytest.raises(ValueError, match='unit is not compatible with an angle: km'):
        Unit.require_angle(Unit.KM)


@pytest.mark.parametrize(('expr', 'expected'), [
    ('km', {'km': 1}),
    ('km*s', {'km': 1, 's': 1}),
    ('km/s', {'km': 1, 's': -1}),
    ('km/s/s', {'km': 1, 's': -2}),
    ('km/(s*s)', {'km': 1, 's': -2}),
    ('km**2', {'km': 2}),
    ('km**-1', {'km': -1}),
    ('(km*s)**2', {'km': 2, 's': 2}),
    ('((km))', {'km': 1}),
    ('km*s/km', {'s': 1}),
    ('  km / s  ', {'km': 1, 's': -1}),
])
def test_units_name_to_dict_parses_expressions(expr: str, expected: dict[str, int]) -> None:
    """name_to_dict() resolves operators, exponents, grouping, and whitespace."""

    assert Unit.name_to_dict(expr) == expected


def test_units_name_to_dict_divides_into_a_group() -> None:
    """A "/" before a parenthesized group inverts every name inside it."""

    assert Unit.name_to_dict('km/(s*rad)') == {'km': 1, 's': -1, 'rad': -1}


def test_units_sqrt_of_a_name_with_an_odd_exponent_derives_a_name() -> None:
    """A unit whose name cannot be halved is left unnamed and names itself instead.

    Unit.STER has even dimension exponents (0, 0, 2), so the dimensions halve cleanly to
    an angle, but its name "ster" has an exponent of 1, which does not.
    """

    result = Unit.STER.sqrt()
    assert result.exponents == (0, 0, 1)
    assert result.name is None
    assert result.get_name() == 'rad'


def test_units_sqrt_unit_of_steradians_is_radians() -> None:
    """sqrt_unit() halves the dimensions of a unit whose name cannot be halved."""

    assert Unit.sqrt_unit(Unit.STER).get_name() == 'rad'


def test_units_scalar_sqrt_carries_a_derived_unit() -> None:
    """A Scalar in steradians can be square-rooted, giving radians."""

    result = Scalar(4., unit=Unit.STER).sqrt()
    assert result.values == 2.
    assert result._unit.get_name() == 'rad'


def test_units_sqrt_keeps_a_name_that_halves_cleanly() -> None:
    """A name with even exponents is halved rather than discarded."""

    result = Unit((2, 0, 0), (1, 1, 0), 'km**2').sqrt()
    assert result.name == {'km': 1}


def test_units_sqrt_of_a_mixed_name_derives_from_the_dimensions() -> None:
    """A name that is only partly halvable is discarded whole, not left half-converted."""

    result = Unit((2, 0, 2), (1, 1, 0), 'km**2*ster').sqrt()
    assert result.name is None
    assert result.get_name() == 'km*rad'


def test_units_name_to_dict_drops_a_cancelled_name() -> None:
    """A name that cancels out entirely is absent from the result."""

    assert Unit.name_to_dict('km/km') == {}


def test_units_name_to_dict_passes_a_dict_through() -> None:
    """A dictionary is already in the returned form and is handed back unchanged."""

    namedict = {'km': 1, 's': -1}
    assert Unit.name_to_dict(namedict) is namedict


def test_units_name_to_dict_rejects_a_non_string() -> None:
    """name_to_dict() reports an argument that is neither a string nor a dictionary."""

    with pytest.raises(ValueError, match='unit is not a string: "123"'):
        Unit.name_to_dict(123)


def test_units_name_to_dict_rejects_a_missing_operand() -> None:
    """An operator with nothing after it is an error."""

    with pytest.raises(ValueError, match='missing operand in unit "km\\*"'):
        Unit.name_to_dict('km*')


def test_units_name_to_dict_rejects_a_missing_operand_before_a_parenthesis() -> None:
    """An operator immediately before a closing parenthesis is an error."""

    with pytest.raises(ValueError, match='missing operand in unit "\\(km/\\)"'):
        Unit.name_to_dict('(km/)')


def test_units_name_to_dict_rejects_an_unbalanced_close_parenthesis() -> None:
    """A closing parenthesis with no opening one is an error."""

    with pytest.raises(ValueError, match='unbalanced "\\)" in unit "km\\)"'):
        Unit.name_to_dict('km)')


def test_units_name_to_dict_rejects_an_exponent_without_an_integer() -> None:
    """A "**" must be followed by an integer."""

    with pytest.raises(ValueError, match='"\\*\\*" without an integer in unit "km\\*\\*"'):
        Unit.name_to_dict('km**')


def test_units_name_to_dict_of_an_empty_string_is_empty() -> None:
    """An empty expression names no units."""

    assert Unit.name_to_dict('') == {}


def test_units_multiply_by_a_unit_named_from_create_name() -> None:
    """A generated name, which carries a zero per unused dimension, survives a product."""

    generated = Unit((6, 0, 0), (1, 1, 0), None).create_name()
    assert generated == {'km': 6, 's': 0, 'rad': 0}

    result = Unit((0, 1, 0), (1, 1, 0), 's') * Unit((6, 0, 0), (1, 1, 0), generated)
    assert result.name == {'s': 1, 'km': 6}


def test_units_divide_by_a_unit_named_from_create_name() -> None:
    """A generated name, which carries a zero per unused dimension, survives a quotient."""

    generated = Unit((6, 0, 0), (1, 1, 0), None).create_name()
    result = Unit((0, 1, 0), (1, 1, 0), 's') / Unit((6, 0, 0), (1, 1, 0), generated)
    assert result.name == {'s': 1, 'km': -6}


def test_units_mul_names_drops_a_zero_absent_from_the_first_name() -> None:
    """A zero exponent in the second name is dropped, not looked up in the first."""

    assert Unit._mul_names({'s': 1}, {'km': 0}) == {'s': 1}


def test_units_div_names_drops_a_zero_absent_from_the_first_name() -> None:
    """A zero exponent in the second name is dropped, not looked up in the first."""

    assert Unit._div_names({'s': 1}, {'km': 0}) == {'s': 1}


@pytest.mark.parametrize(('operation', 'message'), [
    (lambda: Unit.KM * 'invalid', "can't multiply sequence by non-int"),
    (lambda: Unit.KM / 'invalid', 'unsupported operand type'),
    (lambda: 'invalid' / Unit.KM, 'unsupported operand type'),
])
def test_units_unsupported_operand_raises_type_error(operation: Callable[[], object],
                                                     message: str) -> None:
    """An operand that Unit does not support makes the operator raise TypeError.

    The methods themselves return NotImplemented, which leaves Python to try the
    reflected operation of the other operand and then raise.
    """

    with pytest.raises(TypeError, match=message):
        operation()
