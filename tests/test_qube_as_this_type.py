##########################################################################################
# tests/test_qube_as_this_type.py
##########################################################################################

import numpy as np
import pytest

from polymath import Boolean, Pair, Qube, Scalar, Vector, Vector3


def test_qube_as_this_type_scalar_int() -> None:
    """Scalar, int."""

    a = Scalar((1,2,3))
    b = a.as_this_type(7)
    assert b == 7
    assert type(b)
    assert b.is_int()
    b = a.as_this_type(7., coerce=True)
    assert b == 7
    assert type(b)
    assert b.is_int()
    b = a.as_this_type(7., coerce=False)
    assert b == 7
    assert type(b)
    assert b.is_float()
    b = a.as_this_type(Qube(7.), coerce=True)
    assert b == 7
    assert type(b)
    assert b.is_int()
    b = a.as_this_type(Qube(7.), coerce=False)
    assert b == 7
    assert type(b)
    assert b.is_float()
    b = Scalar(7)
    bb = a.as_this_type(b, coerce=True)
    assert (b is bb)
    bb = a.as_this_type(b, coerce=False)
    assert (b is bb)
    b = Scalar(7.)
    bb = a.as_this_type(b, coerce=False)
    assert (b is bb)
    bb = a.as_this_type(b, coerce=True)
    assert (b is not bb)
    b = Boolean(True)
    bb = a.as_this_type(b, coerce=False)
    assert bb == 1
    assert type(bb)
    assert bb.is_int()
    b = Scalar((7,8,9))
    bb = a.as_this_type(b, coerce=True)
    assert (b is bb)
    bb = a.as_this_type(b, coerce=False)
    assert (b is bb)
    b = Scalar((7.,8.,9.))
    bb = a.as_this_type(b, coerce=False)
    assert (b is bb)
    bb = a.as_this_type(b, coerce=True)
    assert (b is not bb)
    b = Boolean([True,False,False,True])
    bb = a.as_this_type(b, coerce=False)
    assert bb == [1,0,0,1]
    assert type(bb)
    assert bb.is_int()

    a = Scalar(1.)
    b = a.as_this_type(7., coerce=True)
    assert b == 7
    assert type(b)
    assert b.is_float()
    b = a.as_this_type(7, coerce=True)
    assert b == 7
    assert type(b)
    assert b.is_float()
    b = a.as_this_type(7, coerce=False)
    assert b == 7
    assert type(b)
    assert b.is_int()
    b = Scalar(7.)
    bb = a.as_this_type(b, coerce=True)
    assert (b is bb)
    bb = a.as_this_type(b, coerce=False)
    assert (b is bb)
    b = Scalar(7)
    bb = a.as_this_type(b, coerce=False)
    assert (b is bb)
    bb = a.as_this_type(b, coerce=True)
    assert (b is not bb)
    b = Boolean(True)
    bb = a.as_this_type(b, coerce=False)
    assert bb == 1
    assert type(bb)
    assert bb.is_int()
    bb = a.as_this_type(b, coerce=True)
    assert bb == 1
    assert type(bb)
    assert bb.is_float()
    b = Scalar((7.,8.,9.))
    bb = a.as_this_type(b, coerce=True)
    assert (b is bb)
    bb = a.as_this_type(b, coerce=False)
    assert (b is bb)
    b = Scalar((7,8,9))
    bb = a.as_this_type(b, coerce=False)
    assert (b is bb)
    bb = a.as_this_type(b, coerce=True)
    assert (b is not bb)
    b = Boolean([True,False,False,True])
    bb = a.as_this_type(b, coerce=False)
    assert bb == [1,0,0,1]
    assert type(bb)
    assert bb.is_int()
    bb = a.as_this_type(b, coerce=True)
    assert bb == [1,0,0,1]
    assert type(bb)
    assert bb.is_float()

    a = Scalar(1.)
    b = Scalar(7)
    db_dt = Scalar(np.arange(4.).reshape(2,2), drank=2)
    b.insert_deriv('t', db_dt)
    bb = a.as_this_type(b, recursive=False, coerce=True)
    assert bb == 7
    assert type(bb)
    assert bb.is_float()
    assert bb.derivs == {}
    bb = a.as_this_type(b, recursive=False, coerce=False)
    assert bb == 7
    assert type(bb)
    assert bb.is_int()
    assert bb.derivs == {}
    bb = a.as_this_type(b, recursive=True, coerce=True)
    assert bb == 7
    assert type(bb)
    assert type(bb.d_dt)
    assert bb.is_float()
    assert bb.d_dt.is_float()
    bb = a.as_this_type(b, recursive=True, coerce=False)
    assert bb == 7
    assert type(bb)
    assert type(bb.d_dt)
    assert bb.is_int()
    assert bb.d_dt.is_float()

    a = Boolean((True,False))
    b = a.as_this_type(7)
    assert b == True
    assert type(b)
    assert b.is_bool()
    b = a.as_this_type(7., coerce=True)
    assert b == True
    assert type(b)
    assert b.is_bool()
    b = a.as_this_type(7., coerce=False)
    assert b == True
    assert type(b)
    assert b.is_bool()
    b = a.as_this_type(Scalar([7.,0.]), coerce=True)
    assert b == [True,False]
    assert type(b)
    b = a.as_this_type(Scalar([7.,0.]), coerce=False)
    assert b == [True,False]
    assert type(b)

    a = Vector((1.,2.,3.))
    with pytest.raises(ValueError):
        a.as_this_type(7)
    b = Scalar((1.,2.,3.))
    with pytest.raises(ValueError):
        a.as_this_type(b)
    b = Boolean((False,True,False))
    with pytest.raises(ValueError):
        a.as_this_type(b)
    b = Vector((1.,2.,3.))
    bb = a.as_this_type(b)
    assert type(bb) == Vector
    b = Vector3((1.,2.,3.))
    bb = a.as_this_type(b)
    assert type(bb) == Vector
    b = Pair((1.,2.))
    bb = a.as_this_type(b)
    assert type(bb) == Vector

    a = Vector3((1.,2.,3.))
    with pytest.raises(ValueError):
        a.as_this_type(7)
    b = Scalar((1.,2.,3.))
    with pytest.raises(ValueError):
        a.as_this_type(b)
    b = Boolean((False,True,False))
    with pytest.raises(ValueError):
        a.as_this_type(b)
    b = Vector((1.,2.,3.))
    bb = a.as_this_type(b)
    assert type(bb) == Vector3
    b = Vector3((1.,2.,3.))
    bb = a.as_this_type(b)
    assert (b is bb)
    b = Pair((1.,2.))
    with pytest.raises(ValueError):
        a.as_this_type(b)
    b = Vector3((1.,2.,3.))
    db_dt = Vector3(np.arange(6.).reshape(3,2), drank=1)
    b.insert_deriv('t', db_dt)
    bb = a.as_this_type(b, recursive=True)
    assert (b is bb)
    b = Vector((1.,2.,3.))
    db_dt = Vector(np.arange(6.).reshape(3,2), drank=1)
    b.insert_deriv('t', db_dt)
    bb = a.as_this_type(b, recursive=True)
    assert (b is not bb)
    assert np.all(bb.values == b.values)
    assert np.all(bb.d_dt.values == b.d_dt.values)
    assert type(b)
    assert type(b.d_dt)


def test_qube_as_this_type_read_only_status() -> None:
    """read-only status."""

    a = Scalar(1.)
    b = Scalar((1,2,3))
    b.as_readonly()
    bb = a.as_this_type(b, coerce=False)
    assert (b is bb)
    assert bb.readonly
    bb = a.as_this_type(b, coerce=True)
    assert (b is not bb)
    assert not bb.readonly
    a = Pair((1.,2.))
    b = Pair((2,3))
    db_dt = Pair(np.arange(4.).reshape(2,2), drank=1)
    b.as_readonly()
    b.insert_deriv('t', db_dt)
    assert b.d_dt.readonly
    bb = a.as_this_type(b, coerce=False)
    assert (b is bb)
    assert bb.readonly
    bb = a.as_this_type(b, coerce=True)
    assert (b is not bb)
    assert not bb.readonly


##########################################################################################
