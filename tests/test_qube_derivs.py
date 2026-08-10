##########################################################################################
# tests/test_qube_derivs.py
##########################################################################################


import pytest

from polymath import Scalar, Vector


def test_qube_derivs_shape_mismatch_raises_error() -> None:
    """shape mismatch raises error."""

    a = Scalar((1,2,3))
    assert a.derivs == {}

    with pytest.raises(ValueError):
        a.insert_deriv('t', Scalar((1,2,3,4)))

    with pytest.raises(ValueError):
        a.insert_deriv('t', Vector((1,2,3)))

    a = Scalar((1,2,3))
    b = Scalar((2,3,4))
    c = Scalar((3,4,5))
    b.insert_deriv('t', c)
    a.insert_deriv('t', b)
    assert hasattr(a, 'd_dt') == True
    assert hasattr(b, 'd_dt') == True
    assert hasattr(a.d_dt, 'd_dt') == False

    a = Scalar((1,2,3), derivs={'t': Scalar((4,5,6)), 'x': Scalar((5,6,7))})
    assert hasattr(a, 'd_dt') == True
    assert hasattr(a, 'd_dx') == True
    a.delete_deriv('t')
    assert hasattr(a, 'd_dt') == False
    assert hasattr(a, 'd_dx') == True
    assert 'x' in a.derivs
    assert 't' not in a.derivs

    a = Scalar((1,2,3), derivs={'t': Scalar((4,5,6)), 'x': Scalar((5,6,7))})
    assert hasattr(a, 'd_dt') == True
    assert hasattr(a, 'd_dx') == True
    a.delete_derivs()
    assert hasattr(a, 'd_dt') == False
    assert hasattr(a, 'd_dx') == False

    a = Scalar((1,2,3), derivs={'t': Scalar((4,5,6)), 'x': Scalar((5,6,7))})
    assert a.d_dx.readonly == False
    a = a.as_readonly()
    assert a.d_dt.readonly == True
    assert a.d_dx.readonly == True
    with pytest.raises(ValueError):
        a.delete_deriv('t')
    with pytest.raises(ValueError):
        a.delete_derivs()
    with pytest.raises(ValueError):
        a.insert_derivs({'a': Scalar((7,8,9)),
                                                    'b': Scalar((8,9,0)),
                                                    'c': Scalar((8,9,0)),
                                                    'd': Scalar((8,9,0)),
                                                    'e': Scalar((8,9,0)),
                                                    'f': Scalar((8,9,0)),
                                                    'g': Scalar((8,9,0)),
                                                    't': Scalar((8,9,0))})
    assert len(a.derivs) == 2
    a.insert_derivs({'a': Scalar((7,8,9)),
                     'b': Scalar((8,9,0)),
                     'c': Scalar((8,9,0)),
                     'd': Scalar((8,9,0)),
                     'e': Scalar((8,9,0)),
                     'f': Scalar((8,9,0)),
                     'g': Scalar((8,9,0))})
    assert len(a.derivs) == 9
    a.insert_deriv('h', Scalar((7,8,9)))
    assert len(a.derivs) == 10
    with pytest.raises(ValueError):
        a.insert_derivs({'a': Scalar((7,8,9))})


def test_qube_derivs_without_derivs() -> None:
    """without_derivs."""

    a = Scalar((1,2,3))
    assert a.derivs == {}

    a = Scalar((1,2,3))
    a.insert_derivs({'a': Scalar((7,8,9)),
                     'b': Scalar((8,9,0)),
                     'c': Scalar((4,5,6)),
                     't': Scalar((5,6,7))})
    assert a.without_derivs().derivs == {}
    assert a.without_derivs(preserve='xxx').derivs == {}
    assert a.without_derivs(preserve=['xxx','yyy']).derivs == {}
    c = a.without_derivs(preserve=['t','xxx'])
    assert 'a' not in c.derivs
    assert 'b' not in c.derivs
    assert 'c' not in c.derivs
    assert 't' in c.derivs
    assert not hasattr(c, 'd_da')
    assert not hasattr(c, 'd_db')
    assert not hasattr(c, 'd_dc')
    assert hasattr(c, 'd_dt')
    assert not a.readonly
    assert not a.d_da.readonly
    assert not a.d_dt.readonly
    a = a.as_readonly()
    assert a.readonly
    assert a.d_da.readonly
    assert a.d_dt.readonly
    b = a.without_derivs()
    assert b.readonly


##########################################################################################
