##########################################################################################
# tests/test_qube_stack.py
##########################################################################################

import numpy as np
import pytest

from polymath import Boolean, Qube, Scalar, Unit


def test_qube_stack_cast_int_to_float() -> None:
    """Cast int to float."""

    a = Scalar(np.arange(10))
    b = Scalar(np.arange(10,20))
    ab = Scalar(np.arange(20).reshape(2,10))
    assert Qube.stack(a,b) == ab
    assert a.is_int()
    assert b.is_int()
    assert ab.is_int()
    assert np.all(Qube.stack(a,b).mask == False)

    b = Scalar(np.arange(10,20.))
    ab = Scalar(np.arange(20.).reshape(2,10))
    assert Qube.stack(a,b) == ab
    assert b.is_float()
    assert ab.is_float()
    assert np.all(Qube.stack(a,b).mask == False)

    c = Boolean(5*[True] + 5*[False])
    d = None
    abcd = Qube.stack(a,b,c,d)
    assert abcd[:2] == ab
    assert abcd[2] == 5*[1.] + 5*[0.]
    assert abcd[3] == 10*[0.]
    assert np.all(abcd.mask == False)
    assert c.is_bool()
    assert abcd.is_float()

    b = Scalar(np.arange(10,20))
    abcd = Qube.stack(a,b,c,d)
    assert abcd[:2] == ab
    assert abcd[2] == 5*[1] + 5*[0]
    assert abcd[3] == 10*[0]
    assert np.all(abcd.mask == False)
    assert abcd.is_int()

    cd = Qube.stack(c,d)
    assert cd[0] == 5*[True] + 5*[False]
    assert cd[1] == 10*[False]
    assert np.all(cd.mask == False)
    assert cd.is_bool()

    b_d_dx = Scalar(np.arange(30.).reshape(10,3), drank=1)
    a_d_dt = Scalar(np.arange(10.) / 10.)
    a.insert_deriv('t', a_d_dt)
    b.insert_deriv('x', b_d_dx)
    abcd = Qube.stack(a,b,c,d)
    assert abcd.d_dt[0] == a_d_dt
    assert abcd.d_dx[1] == b_d_dx
    assert abcd.d_dt[1:] == 0.
    assert abcd.d_dx[0] == [0.,0.,0.]
    assert abcd.d_dx[2:] == [0.,0.,0.]
    a.insert_deriv('t', a_d_dt)
    b.insert_deriv('t', b_d_dx)
    with pytest.raises(ValueError):
        Qube.stack(a, b)
    a = Scalar(np.arange(10))
    b = Scalar(np.arange(10,20))
    a.insert_deriv('t', a_d_dt)
    b.insert_deriv('t', b_d_dx)
    ab = Scalar(np.arange(20).reshape(2,10))
    assert Qube.stack(a,b,recursive=False) == ab
    assert Qube.stack(a,b,recursive=False).derivs == {}

    a = Scalar(np.arange(30.).reshape(10,3), drank=1)
    b = Scalar(np.arange(10.))
    with pytest.raises(ValueError):
        Qube.stack(a, b)
    a = Scalar(np.arange(30.).reshape(10,3), drank=1)
    b = Scalar(np.arange(30.,60.).reshape(10,3), drank=1)
    ab = Qube.stack(a,b)
    assert np.all(ab.values.flatten() == np.arange(60))

    a = Scalar(np.arange(10), unit=Unit.KM)
    b = Scalar(np.arange(10,20))
    ab = Qube.stack(a,b)
    assert ab.units == Unit.KM
    a = Scalar(np.arange(10))
    b = Scalar(np.arange(10,20), unit=Unit.DEG)
    ab = Qube.stack(a,b)
    assert ab.units == Unit.DEG
    a = Scalar(np.arange(10), unit=Unit.KM)
    b = Scalar(np.arange(10,20), unit=Unit.DEG)
    with pytest.raises(ValueError):
        Qube.stack(a, b)


def test_qube_stack_masks() -> None:
    """Masks."""

    a = Scalar(np.arange(10))
    b = Scalar(np.arange(10,20))
    ab = Scalar(np.arange(20).reshape(2,10))
    assert Qube.stack(a,b) == ab
    assert a.is_int()
    assert b.is_int()
    assert ab.is_int()
    assert np.all(Qube.stack(a,b).mask == False)

    a = Scalar(np.arange(10), mask=True)
    b = Scalar(np.arange(10.,20.), mask=True)
    c = Boolean(5*[True] + 5*[False], mask=True)
    d = None
    assert (Qube.stack(a,b,c,d).mask is True)
    a = Scalar(np.arange(10), mask=False)
    b = Scalar(np.arange(10.,20.), mask=False)
    c = Boolean(5*[True] + 5*[False], mask=False)
    d = None
    assert np.all(Qube.stack(a,b,c,d).mask == False)
    a = Scalar(np.arange(10), mask=False)
    b = Scalar(np.arange(10.,20.), mask=True)
    c = Boolean(5*[True] + 5*[False], mask=False)
    d = None
    abcd = Qube.stack(a,b,c,d)
    assert type(abcd.mask) == np.ndarray
    assert abcd[0] == np.arange(10)
    assert abcd[1] == Scalar.MASKED
    assert abcd[2] == [1,1,1,1,1,0,0,0,0,0]
    assert abcd[3] == [0,0,0,0,0,0,0,0,0,0]
    a = Scalar(np.arange(10), mask=[1,1,1,1,1,0,0,0,0,0])
    b = Scalar(np.arange(10.,20.), mask=[1,1,1,1,1,0,0,0,0,0])
    c = Boolean(5*[True] + 5*[False], mask=[1,1,1,1,1,0,0,0,0,0])
    d = None
    abcd = Qube.stack(a,b,c,d)
    assert np.all(abcd[0:3].mask == 3*[[1,1,1,1,1,0,0,0,0,0]])
    assert (abcd[3] == False).all()


def test_qube_stack_broadcasting() -> None:
    """Broadcasting."""

    a = Scalar(np.arange(10))
    b = Scalar(np.arange(10,20))
    ab = Scalar(np.arange(20).reshape(2,10))
    assert Qube.stack(a,b) == ab
    assert a.is_int()
    assert b.is_int()
    assert ab.is_int()
    assert np.all(Qube.stack(a,b).mask == False)

    a = Scalar(np.arange(10).reshape(10,1))
    b = Scalar(11.)
    c = Boolean(5*[True] + 5*[False])
    d = None
    assert Qube.stack(a,b,c,d).shape == (4,10,10)
    a = Scalar(np.arange(10), mask=[0,0,0,0,0,1,1,1,1,1])
    b = Scalar(11., mask=False)
    c = Boolean(5*[True] + 5*[False], mask=True)
    d = None
    abcd = Qube.stack(a,b,c,d)
    assert abcd.shape == (4,10)
    assert abcd.mask.shape == (4,10)
    assert abcd[0][:5] == np.arange(5)
    assert abcd[0][5:] == Scalar.MASKED
    assert abcd[1] == 10*[11.]
    assert abcd[2] == Scalar.MASKED
    assert abcd[3] == 0.
    a = Scalar(np.arange(10), mask=False)
    b = Scalar(np.arange(10.,20.), mask=True)
    c = Boolean(5*[True] + 5*[False], mask=False)
    d = None
    abcd = Qube.stack(a,b,c,d)
    assert type(abcd.mask) == np.ndarray
    assert abcd[0] == np.arange(10)
    assert abcd[1] == Scalar.MASKED
    assert abcd[2] == [1,1,1,1,1,0,0,0,0,0]
    assert abcd[3] == [0,0,0,0,0,0,0,0,0,0]
    a = Scalar(np.arange(10), mask=[1,1,1,1,1,0,0,0,0,0])
    b = Scalar(np.arange(10.,20.), mask=[1,1,1,1,1,0,0,0,0,0])
    c = Boolean(5*[True] + 5*[False], mask=[1,1,1,1,1,0,0,0,0,0])
    d = None
    abcd = Qube.stack(a,b,c,d)
    assert np.all(abcd[0:3].mask == 3*[[1,1,1,1,1,0,0,0,0,0]])
    assert (abcd[3] == False).all()


def test_qube_stack_booleans() -> None:
    """Booleans."""

    a = Scalar(np.arange(10))
    b = Scalar(np.arange(10,20))
    ab = Scalar(np.arange(20).reshape(2,10))
    assert Qube.stack(a,b) == ab
    assert a.is_int()
    assert b.is_int()
    assert ab.is_int()
    assert np.all(Qube.stack(a,b).mask == False)

    c = Boolean(5*[True] + 5*[False])
    d = Scalar(np.arange(10))
    cd = Qube.stack(c,d)
    assert cd.is_int()
    assert type(cd) == Scalar
    d = np.arange(10)
    cd = Qube.stack(c,d)
    assert cd.is_int()
    assert type(cd) == Qube
    d = np.arange(10.)
    cd = Qube.stack(c,d)
    assert cd.is_float()
    assert type(cd) == Qube
    d = 1
    cd = Qube.stack(c,d)
    assert cd.is_int()
    assert type(cd) == Qube
    d = 1.
    cd = Qube.stack(c,d)
    assert cd.is_float()
    assert type(cd) == Qube
    d = True
    cd = Qube.stack(c,d)
    assert cd.is_bool()
    assert type(cd) == Boolean
    d = np.array([True])
    cd = Qube.stack(c,d)
    assert cd.is_bool()
    assert type(cd) == Boolean


##########################################################################################
