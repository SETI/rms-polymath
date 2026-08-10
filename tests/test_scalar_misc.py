##########################################################################################
# test/test_scalar_ops.py
##########################################################################################

import numbers
import numpy as np
import pytest

from polymath import Qube, Scalar, Unit


def test_scalar_misc_constructors() -> None:
    """Constructors."""

    a = np.array(7)             # shapeless array value
    b = Scalar(a)
    assert isinstance(b.vals, numbers.Integral)
    assert (b.vals == 7)
    assert str(b) == 'Scalar(7)'
    a = Scalar([Scalar.MASKED, 4])
    assert a[0] == Scalar.MASKED
    assert a.vals[1] == 4
    assert np.all(a.mask == (True,False))
    a = Scalar([(Scalar.MASKED, 4),(5,6)])
    assert a[0,0] == Scalar.MASKED
    assert a.vals[0,1] == 4
    assert a.vals[1,0] == 5
    assert a.vals[1,1] == 6
    assert np.all(a.mask == [[True,False],[False,False]])

    a = Scalar.zeros((2,3), dtype='int')
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'i'
    assert np.all(a.vals == 0)
    a = Scalar.zeros((2,3), dtype='float')
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'f'
    assert np.all(a.vals == 0)
    a = Scalar.zeros((2,3), dtype='bool')
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'i'    # bool -> int
    assert np.all(a.vals == 0)
    a = Scalar.zeros((2,2), denom=(3,))
    assert a.shape == (2,2)
    assert a.vals.shape == (2,2,3)
    assert np.all(a.vals == 0)
    a = Scalar.zeros((2,2), denom=(3,), mask=[[0,1],[0,0]])
    assert a.shape == (2,2)
    assert a.vals.shape == (2,2,3)
    assert np.all(a.vals == 0)
    assert np.all(a.mask == [[0,1],[0,0]])
    with pytest.raises(ValueError):
        Scalar.zeros((2,3), numer=(3,))

    a = Scalar.ones((2,3), dtype='int')
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'i'
    assert np.all(a.vals == 1)
    a = Scalar.ones((2,3), dtype='float')
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'f'
    assert np.all(a.vals == 1)
    a = Scalar.ones((2,3), dtype='bool')
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'i'    # bool -> int
    assert np.all(a.vals == 1)
    a = Scalar.ones((2,2), denom=(3,))
    assert a.shape == (2,2)
    assert a.vals.shape == (2,2,3)
    assert np.all(a.vals == 1)
    a = Scalar.ones((2,2), denom=(3,), mask=[[0,1],[0,0]])
    assert a.shape == (2,2)
    assert a.vals.shape == (2,2,3)
    assert np.all(a.vals == 1)
    assert np.all(a.mask == [[0,1],[0,0]])
    with pytest.raises(ValueError):
        Scalar.ones((2,3), numer=(3,))

    a = Scalar.filled((2,3), 7)
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'i'
    assert np.all(a.vals == 7)
    a = Scalar.filled((2,3), 7.)
    assert a.shape == (2,3)
    assert a.vals.dtype.kind == 'f'
    assert np.all(a.vals == 7)
    a = Scalar.filled((2,2), 7, denom=(3,))
    assert a.shape == (2,2)
    assert a.vals.shape == (2,2,3)
    assert np.all(a.vals == 7)
    a = Scalar.filled((2,2), 7, denom=(3,), mask=[[0,1],[0,0]])
    assert a.shape == (2,2)
    assert a.vals.shape == (2,2,3)
    assert np.all(a.vals == 7)
    assert np.all(a.mask == [[0,1],[0,0]])
    with pytest.raises(ValueError):
        Scalar.filled(7, (2,3), numer=(3,))

    ints = Scalar((1,2,3))
    test = Scalar(np.array([1,2,3]))
    assert ints == test
    test = Scalar(test)
    assert ints == test
    assert ints == (1,2,3)
    assert ints == [1,2,3]
    assert ints.shape == (3,)
    assert -ints == [-1,-2,-3]
    assert +ints == [1,2,3]
    assert ints == abs(ints)
    assert ints == abs(Scalar(( 1, 2, 3)))
    assert ints == abs(Scalar((-1,-2,-3)))
    assert ints * 2 == [2,4,6]
    assert ints / 2. == [0.5,1,1.5]

    assert ints / 2 == [0.5,1,1.5]         # now truediv
    assert ints + 1 == [2,3,4]
    assert ints - 0.5 == (0.5,1.5,2.5)
    assert ints % 2 == (1,0,1)
    assert ints + Scalar([1,2,3]) == [2,4,6]
    assert ints - Scalar((1,2,3)) == [0,0,0]
    assert ints * [1,2,3] == [1,4,9]
    assert ints / [1,2,3] == [1,1,1]
    assert ints % [1,3,3] == [0,2,0]
    with pytest.raises(ValueError):
        ints.__add__((4,5))
    with pytest.raises(ValueError):
        ints.__sub__((4,5))
    with pytest.raises(ValueError):
        ints.__mul__((4,5))
    with pytest.raises(ValueError):
        ints.__truediv__((4,5))
    with pytest.raises(ValueError):
        ints.__mod__((4,5))
    with pytest.raises(ValueError):
        ints.__add__(Scalar((4,5)))
    with pytest.raises(ValueError):
        ints.__sub__(Scalar((4,5)))
    with pytest.raises(ValueError):
        ints.__mul__(Scalar((4,5)))
    with pytest.raises(ValueError):
        ints.__truediv__(Scalar((4,5)))
    with pytest.raises(ValueError):
        ints.__mod__(Scalar((4,5)))

    ints = Scalar((1,2,3))
    ints += 1
    assert ints == [2,3,4]
    ints -= 1
    assert ints == [1,2,3]
    ints *= 2
    assert ints == [2,4,6]
    ints //= 2
    assert ints == [1,2,3]
    ints *= (3,2,1)
    assert ints == [3,4,3]
    ints //= (1,2,3)
    assert ints == [3,2,1]
    ints += (1,2,3)
    assert ints == 4
    assert ints == [4]
    assert ints == [4,4,4]
    assert ints == Scalar([4,4,4])
    ints -= (3,2,1)
    assert ints == [1,2,3]
    test = Scalar((10,10,10))
    test %= 4
    assert test == 2
    test = Scalar((10,10,10))
    test %= (4,3,2)
    assert test == [2,1,0]
    test = Scalar((10,10,10))
    test %= Scalar((5,4,3))
    assert test == [0,2,1]
    with pytest.raises(ValueError):
        ints.__iadd__((4,5))
    with pytest.raises(ValueError):
        ints.__isub__((4,5))
    with pytest.raises(ValueError):
        ints.__imul__((4,5))
    with pytest.raises(ValueError):
        ints.__imod__((4,5))
    with pytest.raises(ValueError):
        ints.__ifloordiv__((4,5))
    with pytest.raises(ValueError):
        ints.__iadd__(Scalar((4,5)))
    with pytest.raises(ValueError):
        ints.__isub__(Scalar((4,5)))
    with pytest.raises(ValueError):
        ints.__imul__(Scalar((4,5)))
    with pytest.raises(ValueError):
        ints.__imod__(Scalar((4,5)))
    with pytest.raises(ValueError):
        ints.__ifloordiv__(Scalar((4,5)))
    with pytest.raises(TypeError):
        ints.__itruediv__((4,5))
    with pytest.raises(TypeError):
        ints.__itruediv__(Scalar((4,5)))

    floats = Scalar((1.,2.,3.))
    floats += 1
    assert floats == [2,3,4]
    floats -= 1
    assert floats == [1,2,3]
    floats *= 2
    assert floats == [2,4,6]
    floats /= 2
    assert floats == [1,2,3]
    floats *= (3,2,1)
    assert floats == [3,4,3]
    floats /= (1,2,3)
    assert floats == [3,2,1]
    floats += (1,2,3)
    assert floats == 4
    assert floats == [4]
    assert floats == [4,4,4]
    assert floats == Scalar([4,4,4])
    floats -= (3,2,1)
    assert floats == [1,2,3]
    test = Scalar((10,10,10))
    test %= 4
    assert test == 2
    test = Scalar((10,10,10))
    test %= (4,3,2)
    assert test == [2,1,0]
    test = Scalar((10,10,10))
    test %= Scalar((5,4,3))
    assert test == [0,2,1]
    with pytest.raises(ValueError):
        floats.__iadd__((4,5))
    with pytest.raises(ValueError):
        floats.__isub__((4,5))
    with pytest.raises(ValueError):
        floats.__imul__((4,5))
    with pytest.raises(ValueError):
        floats.__itruediv__((4,5))
    with pytest.raises(ValueError):
        floats.__imod__((4,5))
    with pytest.raises(ValueError):
        floats.__ifloordiv__((4,5))
    with pytest.raises(ValueError):
        floats.__iadd__(Scalar((4,5)))
    with pytest.raises(ValueError):
        floats.__isub__(Scalar((4,5)))
    with pytest.raises(ValueError):
        floats.__imul__(Scalar((4,5)))
    with pytest.raises(ValueError):
        floats.__itruediv__(Scalar((4,5)))
    with pytest.raises(ValueError):
        floats.__imod__(Scalar((4,5)))
    with pytest.raises(ValueError):
        floats.__ifloordiv__(Scalar((4,5)))

    assert ints[0] == 1
    floats = ints.as_float()
    assert floats[0] == 1.
    six = Scalar([1,2,3,4,5,6])
    assert six.shape == (6,)
    test = six.copy().reshape((3,1,2))
    assert test.shape == (3,1,2)
    assert test == [[[1,2]],[[3,4]],[[5,6]]]
    assert test.swap_axes(0,1).shape == (1,3,2)
    assert test.swap_axes(0,2).shape == (2,1,3)
    assert test.flatten().shape == (6,)
    four = Scalar([1,2,3,4]).reshape((2,2))
    assert four == [[1,2],[3,4]]
    assert Qube.broadcasted_shape(four,test) == (3,2,2)
    assert four.broadcast_into_shape((3,2,2)) == ([[[1,2],[3,4]],
                            [[1,2],[3,4]],
                            [[1,2],[3,4]]])
    assert test.broadcast_into_shape((3,2,2)) == ([[[1,2],[1,2]],
                            [[3,4],[3,4]],
                            [[5,6],[5,6]]])
    assert four.broadcast_into_shape((3,2,2)) == ([[[1,2],[3,4]],
                      [[1,2],[3,4]],
                      [[1,2],[3,4]]])
    assert test.broadcast_into_shape((3,2,2)) == ([[[1,2],[1,2]],
                      [[3,4],[3,4]],
                      [[5,6],[5,6]]])
    ten = four + test
    assert ten.shape == (3,2,2)
    assert ten == ([[[2, 4], [4, 6]],
                           [[4, 6], [6, 8]],
                           [[6, 8], [8,10]]])
    x24 = four * test
    assert x24.shape == (3,2,2)
    assert x24 == ([[[1, 4], [ 3, 8]],
                           [[3, 8], [ 9,16]],
                           [[5,12], [15,24]]])

    test = Scalar(list(range(6)))
    assert str(test) == "Scalar(0 1 2 3 4 5)"
    test = Scalar(test, mask=(3*[True] + 3*[False]))
    assert str(test) == "Scalar(-- -- -- 3 4 5; mask)"
    assert str(test+1) == "Scalar(-- -- -- 4 5 6; mask)"
    assert str(test-2) == "Scalar(-- -- -- 1 2 3; mask)"
    assert str(test*2) == "Scalar(-- -- -- 6 8 10; mask)"
    assert str(test/2) == "Scalar(-- -- -- 1.5 2.0 2.5; mask)"
    assert str(test%2) == "Scalar(-- -- -- 1 0 1; mask)"
    assert str(test-2.) == "Scalar(-- -- -- 1.0 2.0 3.0; mask)"
    assert str(test+2.) == "Scalar(-- -- -- 5.0 6.0 7.0; mask)"
    assert str(test*2.) == "Scalar(-- -- -- 6.0 8.0 10.0; mask)"
    assert str(test/2.) == "Scalar(-- -- -- 1.5 2.0 2.5; mask)"
    assert str(test + [1, 2, 3, 4, 5, 6]) == "Scalar(-- -- -- 7 9 11; mask)"
    assert str(test - [1, 2, 3, 4, 5, 6]) == "Scalar(-- -- -- -1 -1 -1; mask)"
    assert str(test * [1, 2, 3, 4, 5, 6]) == "Scalar(-- -- -- 12 20 30; mask)"
    assert str(test / [1, 7, 5, 1, 2, 1]) == "Scalar(-- -- -- 3.0 2.0 5.0; mask)"
    assert str(test / [0, 7, 5, 1, 2, 0]) == "Scalar(-- -- -- 3.0 2.0 --; mask)"
    assert str(test % [0, 7, 5, 1, 2, 0]) == "Scalar(-- -- -- 0 0 --; mask)"
    temp = Scalar(6*[1], 5*[False] + [True])
    assert str(temp) == "Scalar(1 1 1 1 1 --; mask)"
    assert str(test + temp) == "Scalar(-- -- -- 4 5 --; mask)"
    foo = test + temp
    assert (foo.vals[0] == test.vals[0] + temp.vals[0])
    foo.vals[0] = 99
    assert foo.vals[0] != test.vals[0] + temp.vals[0]
    assert foo == test + temp
    assert test[5] == 5
    assert test[-1] == 5
    assert test[3:] == [3,4,5]
    assert test[3:5] == [3,4]
    assert test[3:-1] == [3,4]
    assert test[0] == Scalar(0, True)
    assert str(test[0]) == "Scalar(--; mask)"
    assert str(test[0:4]) == "Scalar(-- -- -- 3; mask)"
    assert str(test[0:1]) == "Scalar(--; mask)"
    assert str(test[5]) == "Scalar(5)"
    assert str(test[4:]) == "Scalar(4 5)"
    assert str(test[5:]) == "Scalar(5)"
    assert str(test[0:6:2]) == "Scalar(-- -- 4; mask)"
    mvals = test.mvals
    assert type(mvals) == np.ma.MaskedArray
    assert str(mvals) == "[-- -- -- 3 4 5]"
    temp = Scalar(list(range(6)))
    mvals = temp.mvals
    assert type(mvals) == np.ma.MaskedArray
    assert str(mvals) == "[0 1 2 3 4 5]"
    assert mvals.mask == np.ma.nomask
    temp = Scalar(temp, mask=True)
    assert str(temp) == "Scalar(-- -- -- -- -- --; mask)"
    mvals = temp.mvals
    assert type(mvals) == np.ma.MaskedArray
    assert str(mvals) == "[-- -- -- -- -- --]"

    test = Scalar(list(range(6)))
    assert test == np.arange(6)
    km = Scalar(list(range(6)), unit=Unit.KM)
    cm = Scalar(np.arange(6), unit=Unit.CM)
    assert np.all(km.values == cm.values)
    cm = cm.into_unit()
    EPS = 1.e-15
    assert np.all(np.abs(km.values - cm/1.e5) < 1.e5*EPS)
    with pytest.raises(ValueError):
        km.set_unit(Unit.SECONDS)


##########################################################################################
