##########################################################################################
# tests/test_indices.py
##########################################################################################

import warnings
import numpy as np
import pytest

from polymath import Scalar, Pair, Vector, Matrix, Boolean, Qube


def test_indices_an_unmasked_scalar() -> None:
    """An unmasked Scalar."""

    def make_masked(orig, mask_list):
        ret = orig.copy()
        ret[np.array(mask_list)] = np.ma.masked
        return ret
    def extract(a, indices):
        ret = []
        for index in indices:
            ret.append(a[index])

        # NOTE: can raise UserWarning:
        #    Warning: converting a masked element to nan.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = np.ma.array(ret)

        return result
    def compare_a_b_1d(a, b, class_):
        """Input a is a Qube subclass made from MaskedArray b, at least 1-D."""

        # Traditional indexing
        assert a == b
        assert a[1] == b[1]
        assert a[-1] == b[-1]
        assert a[1:5] == b[1:5]
        assert a[1:5:2] == b[1:5:2]
        assert a[-5:] == b[-5:]
        assert a[:] == b[:]
        assert a[...] == b[...]
        assert a[...,:] == b[...,:]
        assert a[::-1] == b[::-1]

        # Single Scalar
        assert a[Scalar(1)] == b[1]
        assert a[Scalar(1,True)] == make_masked(b, [1])[1]

        # Two elements
        assert a[Scalar((1,3))] == b[1:4:2]
        assert a[Scalar((1,3),(True,False))] == make_masked(b, [1])[1:4:2]
        assert a[Scalar((1,3),(True,False))] == Qube.stack(a[3].as_all_masked(), a[3])
        assert a[Scalar((1,3),True)] == make_masked(b, [1,3])[1:4:2]
        assert a[Scalar((1,3),True)] == class_.zeros((), denom=a.denom, numer=a.numer, mask=True)
        assert a[Scalar((1,3),True)].shape == (2,) + a.shape[1:]

        # Boolean
        assert a[True] == b
        assert a[True].shape == a.shape
        assert a[False].shape == (0,) + a.shape[1:]
        assert a[Boolean(True)] == b
        assert a[Boolean(True)].shape == a.shape
        assert a[Boolean(False)].shape == (0,) + a.shape[1:]
        assert a[Boolean.MASKED].shape == (1,) + a.shape[1:]
        assert a[Boolean.MASKED].mask == True
    def compare_a_b_2d(a, b, class_):
        """Input a is a Qube subclass made from MaskedArray b, at least 2-D."""

        assert a[Pair((1,1))] == b[1,1]

        assert a[Pair((1,1),True)] == make_masked(b, [[1,1]])[1,1]
        assert a[Pair(((1,1),(2,2),(3,3)))] == extract(b, ((1,1),(2,2),(3,3)))
        assert a[Pair(((1,1),(2,2),(3,3)),False)] == extract(b, ((1,1),(2,2),(3,3)))
        assert a[Pair(((1,1),(2,2),(3,3)),True)] == make_masked(extract(b, ((1,1),(2,2),(3,3))), [0,1,2])
        assert a[Pair(((1,1),(2,2),(3,3)),(True,False,False))] == make_masked(extract(b, ((1,1),(2,2),(3,3))), [0])
        assert a[Pair(((1,1),(2,2),(3,3)),(False,True,False))] == make_masked(extract(b, ((1,1),(2,2),(3,3))), [1])
        assert a[Pair(((1,1),(2,2),(3,3)),(False,False,True))] == make_masked(extract(b, ((1,1),(2,2),(3,3))), [2])
    def compare_a_b_3d(a, b, class_):
        """Input a is a Qube subclass made from MaskedArray b, at least 3-D.
        """

        # Indexed by 3-D Vector
        assert a[Vector((1,1,1))] == b[1,1,1]
        assert a[Vector((1,1,1),True)] == make_masked(b, [[1,1,1]])[1,1,1]
        assert a[Vector(((1,1,1),(2,2,2),(3,3,3)))] == extract(b, ((1,1,1),(2,2,2),(3,3,3)))
        assert a[Vector(((1,1,1),(2,2,2),(3,3,3)),False)] == extract(b, ((1,1,1),(2,2,2),(3,3,3)))
        assert a[Vector(((1,1,1),(2,2,2),(3,3,3)),True)] == (make_masked(extract(b, ((1,1,1),(2,2,2),(3,3,3))),
                                     [0,1,2]))
        assert a[Vector(((1,1,1),(2,2,2),(3,3,3)),(True,0,0))] == (make_masked(extract(b, ((1,1,1),(2,2,2),(3,3,3))),
                                     [0]))
        assert a[Vector(((1,1,1),(2,2,2),(3,3,3)),(0,True,0))] == (make_masked(extract(b, ((1,1,1),(2,2,2),(3,3,3))),
                                     [1]))
        assert a[Vector(((1,1,1),(2,2,2),(3,3,3)),(0,0,True))] == (make_masked(extract(b, ((1,1,1),(2,2,2),(3,3,3))),
                                     [2]))

        # Indexed by mixed types
        assert a[(0,Scalar(3),0)] == 15.
        assert a[(0,Scalar(3))] == [15,16,17,18,19]
        assert a[(0,Scalar(3,True),0)] == Scalar.MASKED
        assert a[(0,Scalar(3,True))].shape == (5,)
        assert np.all(a[(0,Scalar(3,True))].mask == True)

        assert a[(Ellipsis, Scalar([0,1],False))] == b[...,(0,1)]
        assert np.all(a[(Ellipsis, Scalar([0,1],True))].mask == True)

        indx = (Scalar([1,2]), Ellipsis, Scalar([0,1]))
        assert a[indx] == b[(1,2),...,(0,1)]

        indx = (Scalar([1,2],True), Ellipsis, Scalar([0,1]))
        assert np.all(a[indx].mask == True)

        indx = (Scalar([1,2],True), Ellipsis, Scalar([0,1],True))
        assert np.all(a[indx].mask == True)
    def check_derivs_1d(c):
        """Alternative ways of indexing a 1-D derivative."""

        assert c[1].d_dt == c.d_dt[1]
        assert c[-1].d_dt == c.d_dt.vals[-1]
        assert c[1:5].d_dt == c.d_dt[1:5]
        assert c[1:5:2].d_dt == c.d_dt[1:5:2]
        assert c[-5:].d_dt == c.d_dt[-5:]
        assert c[:].d_dt == c.d_dt
        assert c[...].d_dt == c.d_dt
        assert c[::-1].d_dt == c.d_dt[::-1]

        assert c[1].d_dxy == c.d_dxy[1]
        assert c[-1].d_dxy == c.d_dxy[-1]
        assert c[1:3].d_dxy == c.d_dxy[1:3]
        assert c[1:4:2].d_dxy == c.d_dxy[1:4:2]
        assert c[-3:].d_dxy == c.d_dxy[-3:]
        assert c.d_dxy[:] == c.d_dxy
        assert c[:].d_dxy == c.d_dxy
        assert c[...].d_dxy == c.d_dxy
        assert c[::-1].d_dxy == c.d_dxy[::-1]
    def check_derivs_2d(c, ellipses=True):
        """Alternative ways of indexing a 2-D derivative."""

        assert c[1,0].d_dt == c.d_dt[1,0]
        assert c[-1,0].d_dt == c.d_dt.vals[-1,0]
        assert c[1:5,3].d_dt == c.d_dt[1:5,3]
        assert c[:-1,1:5:2].d_dt == c.d_dt[:-1,1:5:2]
        assert c[-1,-5:].d_dt == c.d_dt[-1,-5:]
        assert c[:,0].d_dt == c[:,0].d_dt
        assert c[:,0:].d_dt == c[:,0:].d_dt
        assert c[:,-1].d_dt == c[:,-1].d_dt
        assert c[:,-1:].d_dt == c[:,-1:].d_dt
        assert c[::-1,:2].d_dt == c.d_dt[::-1,:2]
        if ellipses:
            assert c[...,2].d_dt == c[...,2].d_dt
            assert c[-2,...].d_dt == c[-2,...].d_dt
            assert c[:-2,...,1].d_dt == c[:-2,...,1].d_dt

        assert c[Scalar(1),0].d_dt == c.d_dt[1,0]
        assert c[Scalar(-1),0].d_dt == c.d_dt.vals[-1,0]
        assert c[1:5,Scalar((3,4))].d_dt == c.d_dt[1:5,3:5]
        assert c[-1,-5:].d_dt == c.d_dt[Scalar(-1),-5:]
        if ellipses:
            assert c[...,Scalar(2)].d_dt == c[...,2].d_dt
            assert c[Scalar(-2),...].d_dt == c[-2,...].d_dt
            assert c[:-2,...,Scalar(1)].d_dt == c[:-2,...,1].d_dt
            assert c[Scalar(0),...,Scalar(-1)].d_dt == c.d_dt[0,-1]
            assert c[Scalar((1,0)),...,Scalar(-1)].d_dt == c.d_dt[Pair(((1,-1),(0,-1)))]

        assert c[Pair((1,0))].d_dt == c.d_dt[1,0]
        assert c[Pair((-1,0))].d_dt == c.d_dt.vals[-1,0]
        assert c[Pair([(1,3),(2,3),(3,3),(4,3)])].d_dt == c.d_dt[1:5,3]

    b = np.ma.arange(10)
    a = Scalar(b.data, False)
    c = a.copy()
    c.insert_deriv('t', Scalar([5,4,3,2,1,0,9,8,7,6]))
    c.insert_deriv('xy', Scalar(-2*np.arange(20.).reshape(10,2), drank=1))
    compare_a_b_1d(a, b, Scalar)
    check_derivs_1d(c)

    b = np.ma.arange(10)
    b[:] = np.ma.masked
    a = Scalar(b, True)
    c = a.copy()
    c.insert_deriv('t', Scalar([5,4,3,2,1,0,9,8,7,6]))
    c.insert_deriv('xy', Scalar(-2*np.arange(20.).reshape(10,2), drank=1))
    compare_a_b_1d(a, b, Scalar)
    check_derivs_1d(c)

    b = np.ma.arange(10)
    b[3] = np.ma.masked
    a = Scalar(b)
    c = a.copy()
    c.insert_deriv('t', Scalar([5,4,3,2,1,0,9,8,7,6],
                               mask=[0,0,0,1,0,0,0,0,0,0]))
    c.insert_deriv('xy', Scalar(-2*np.arange(20.).reshape(10,2), drank=1,
                                mask=[0,0,0,1,0,0,0,0,0,0]))
    compare_a_b_1d(a, b, Scalar)
    check_derivs_1d(c)

    b = np.ma.arange(25).reshape(5,5)
    a = Scalar(b, False)
    c = a.copy()
    c.insert_deriv('t', Scalar(np.random.randn(5,5)))
    c.insert_deriv('xy', Scalar(np.random.randn(5,5,2), drank=1))
    compare_a_b_1d(a, b, Scalar)
    check_derivs_1d(c)

    b = np.ma.arange(25).reshape(5,5)
    a = Scalar(b)
    c = a.copy()
    c.insert_deriv('t', Scalar(np.random.randn(5,5)))
    c.insert_deriv('xy', Scalar(np.random.randn(5,5,2), drank=1))
    compare_a_b_1d(a, b, Scalar)
    compare_a_b_2d(a, b, Scalar)
    check_derivs_1d(c)
    check_derivs_2d(c)

    b = np.ma.arange(25).reshape(5,5)
    b[1,1] = np.ma.masked
    a = Scalar(b)
    c = a.copy()
    c.insert_deriv('t', Scalar(np.random.randn(5,5)))
    c.insert_deriv('xy', Scalar(np.random.randn(5,5,2), drank=1))
    compare_a_b_1d(a, b, Scalar)
    compare_a_b_2d(a, b, Scalar)
    compare_a_b_1d(a, b, Scalar)
    compare_a_b_2d(a, b, Scalar)
    check_derivs_1d(c)
    check_derivs_2d(c)

    b = np.ma.arange(125).reshape(5,5,5)
    a = Scalar(b)
    c = a.copy()
    c.insert_deriv('t', Scalar(np.random.randn(5,5,5)))
    c.insert_deriv('xy', Scalar(np.random.randn(5,5,5,2), drank=1))
    compare_a_b_1d(a, b, Scalar)
    compare_a_b_2d(a, b, Scalar)
    check_derivs_1d(c)
    check_derivs_2d(c, ellipses=False)

    b = np.ma.arange(72).reshape(6,6,2)
    a = Scalar(b)
    c = a.copy()
    c.insert_deriv('t', Scalar(np.random.randn(6,6,2)))
    c.insert_deriv('xy', Scalar(np.random.randn(6,6,2,2), drank=1))
    compare_a_b_1d(a, b, Scalar)
    compare_a_b_2d(a, b, Scalar)
    check_derivs_1d(c)
    check_derivs_2d(c, ellipses=False)

    b = np.ma.arange(75).reshape(5,5,3)
    b[1,1,1] = np.ma.masked
    a = Scalar(b)
    c = a.copy()
    c.insert_deriv('t', Scalar(np.random.randn(5,5,3)))
    c.insert_deriv('xy', Scalar(np.random.randn(5,5,3,2), drank=1))
    compare_a_b_1d(a, b, Scalar)
    compare_a_b_2d(a, b, Scalar)
    check_derivs_1d(c)
    check_derivs_2d(c, ellipses=False)

    b = np.ma.arange(125).reshape(5,5,5)
    a = Scalar(b)
    c = a.copy()
    c.insert_deriv('t', Scalar(np.random.randn(5,5,5)))
    c.insert_deriv('xy', Scalar(np.random.randn(5,5,5,2), drank=1))
    compare_a_b_1d(a, b, Scalar)
    compare_a_b_2d(a, b, Scalar)
    compare_a_b_3d(a, b, Scalar)
    check_derivs_1d(c)
    check_derivs_2d(c, ellipses=False)

    b = np.ma.arange(20).reshape(5,2,2)
    a = Matrix(b)
    c = a.copy()
    c.insert_deriv('t', Matrix(np.random.randn(5,2,2)))
    c.insert_deriv('xy', Matrix(np.random.randn(5,2,2,2), drank=1))
    compare_a_b_1d(a, b, Matrix)
    check_derivs_1d(c)

    b = np.ma.arange(100).reshape(5,5,2,2)
    a = Matrix(b)
    c = a.copy()
    c.insert_deriv('t', Matrix(np.random.randn(5,5,2,2)))
    c.insert_deriv('xy', Matrix(np.random.randn(5,5,2,2,2), drank=1))
    compare_a_b_1d(a, b, Matrix)
    compare_a_b_2d(a, b, Matrix)
    check_derivs_1d(c)
    check_derivs_2d(c)

    a = Pair(np.arange(6).reshape((3,2)), mask=[False, False, True])
    assert a[2] == Pair.MASKED
    assert a[np.array([True,False,True])] == [Pair((0,1)),Pair.MASKED]
    assert a[Boolean([True,False,True])] == [Pair((0,1)),Pair.MASKED]
    a = a.insert_deriv('t', Pair(-np.arange(6).reshape((3,2)), mask=a.mask))
    assert a[2].d_dt == Pair.MASKED
    assert a[np.array([True,False,True])].d_dt == [Pair((0,-1)),Pair.MASKED]
    assert a[Boolean([True,False,True])].d_dt == [Pair((0,-1)),Pair.MASKED]

    a = Scalar(0.)
    assert a[True] == a
    assert a[..., True] == a
    assert a[..., True].shape == ()
    assert a[..., True, None, None] == a
    assert a[..., True, None, None].shape == (1,1)
    assert a[None, ..., True, None] == a
    assert a[None, ..., None, True].shape == (1,1)
    assert a[None, ..., None] == a
    assert a[None, ..., None].shape == (1,1)
    assert a[False].shape == (0,)
    assert a[..., False].shape == (0,)
    assert a[..., False, None, None].shape == (0,1,1)
    assert a[None, ..., False, None].shape == (1,0,1)
    BM = Boolean.MASKED
    assert a[BM] == Scalar.MASKED
    assert a[BM].shape == ()
    assert a[..., BM] == Scalar.MASKED
    assert a[..., BM].shape == ()
    assert a[..., BM, None, None] == Scalar.MASKED
    assert a[..., BM, None, None].shape == (1,1)
    assert a[None, ..., BM, None] == Scalar.MASKED
    assert a[None, ..., BM, None].shape == (1,1)
    a.insert_deriv('xy', Scalar((1.,2.), drank=1))
    assert a[True].d_dxy == a.d_dxy
    assert a[True].d_dxy.shape == ()
    assert a[..., True].d_dxy == a.d_dxy
    assert a[..., True].d_dxy.shape == ()
    assert a[..., True, None, None].d_dxy == a.d_dxy
    assert a[..., True, None, None].d_dxy.shape == (1,1)
    assert a[None, ..., True, None].d_dxy == a.d_dxy
    assert a[None, ..., None, True].d_dxy.shape == (1,1)
    assert a[None, ..., None].d_dxy == a.d_dxy
    assert a[None, ..., None].d_dxy.shape == (1,1)
    assert a[False].d_dxy.shape == (0,)
    assert a[..., False].d_dxy.shape == (0,)
    assert a[..., False, None, None].d_dxy.shape == (0,1,1)
    assert a[None, ..., False, None].d_dxy.shape == (1,0,1)
    dxy_masked = Scalar((0.,0.), drank=1, mask=True)
    assert a[BM].d_dxy == dxy_masked
    assert a[BM].d_dxy.shape == ()
    assert a[..., BM].d_dxy == dxy_masked
    assert a[..., BM].d_dxy.shape == ()
    assert a[..., BM, None, None].d_dxy == dxy_masked
    assert a[..., BM, None, None].d_dxy.shape == (1,1)
    assert a[None, ..., BM, None].d_dxy == dxy_masked
    assert a[None, ..., BM, None].d_dxy.shape == (1,1)
    with pytest.raises(IndexError):
        a.__getitem__((Ellipsis, None, Ellipsis))
    with pytest.raises(IndexError):
        a.__getitem__((True, False))
    with pytest.raises(IndexError):
        a.__getitem__((True, True))

    # __setitem__

    a = Scalar(1.)
    assert a == 1
    a[True] = 7
    assert a == 7
    a[False] = -7
    assert a == 7
    a[Boolean(True)] = 4
    assert a == 4
    a[Boolean(False)] = -7
    assert a == 4
    a[Boolean.MASKED] = -7
    assert a == 4

    a = Scalar(np.arange(3))
    a[True] = np.arange(4,7)
    assert a == np.arange(4,7)
    a[..., True] = np.arange(3)
    assert a == np.arange(3)
    a[None, None, ..., True] = np.arange(4,7)
    assert a == np.arange(4,7)
    a[None, ..., True, None] = np.arange(3).reshape(3,1)
    assert a == np.arange(3)
    a = Scalar(np.arange(4,7))
    a[False] = np.arange(3)
    assert a == np.arange(4,7)
    a[..., False] = np.arange(3)
    assert a == np.arange(4,7)
    a[None, ..., False, None] = np.arange(3).reshape(3,1)
    assert a == np.arange(4,7)
    a[Boolean(True)] = np.arange(8,11)
    assert a == np.arange(8,11)
    a[Boolean(False)] = np.arange(3)
    assert a == np.arange(8,11)
    a[Boolean.MASKED] = np.arange(3)
    assert a == np.arange(8,11)
    a[np.array([True, True, False])] = 7
    assert a == [7,7,10]
    a[Boolean([False, True, True])] = -7
    assert a == [7,-7,-7]
    a[Boolean([False, True, True], mask=(0,0,1))] = 3
    assert a == [7,3,-7]
    assert a.derivs == {}
    five = Scalar(5, derivs={'t': Scalar(-5)})
    a[Boolean([False, False, True], mask=(0,0,1))] = five
    assert a.derivs == {}
    a[Boolean([False, True, True], mask=(0,0,1))] = five
    assert a.derivs == {'t': Scalar([0,-5,0])}

    b = np.zeros(10)
    a = Scalar(b)
    a[2] = 1
    assert a == Scalar((0,0,1,0,0,0,0,0,0,0))
    a[Scalar(3)] = 1
    assert a == Scalar((0,0,1,1,0,0,0,0,0,0))
    a[Scalar(4,True)] = 1
    assert np.all(a.values == (0,0,1,1,0,0,0,0,0,0))
    assert not np.any(a.mask)
    a[Scalar((5,6,7),(True,False,True))] = 2
    assert np.all(a.values == (0,0,1,1,0,0,2,0,0,0))
    assert not np.any(a.mask)
    a[Scalar(1)] = Scalar(3,True)
    assert np.all(a.values == (0,3,1,1,0,0,2,0,0,0))
    assert np.all(a.mask   == (0,1,0,0,0,0,0,0,0,0))
    a[Scalar(0,True)] = a[2] + 3
    assert np.all(a.values == (0,3,1,1,0,0,2,0,0,0))
    assert np.all(a.mask   == (0,1,0,0,0,0,0,0,0,0))
    a[Scalar(0,False)] = a[2] + 3
    assert np.all(a.values == (4,3,1,1,0,0,2,0,0,0))
    assert np.all(a.mask   == (0,1,0,0,0,0,0,0,0,0))
    a[Scalar((0,2,4))] = Scalar(4,True)
    assert np.all(a.values == (4,3,4,1,4,0,2,0,0,0))
    assert np.all(a.mask   == (1,1,1,0,1,0,0,0,0,0))
    a[Scalar((0,2,4))] = Scalar((5,6,7))
    assert np.all(a.values == (5,3,6,1,7,0,2,0,0,0))
    assert np.all(a.mask   == (0,1,0,0,0,0,0,0,0,0))
    a[Scalar((-1,-2,-3))] = a[Scalar((0,1,2))]
    assert np.all(a.values == (5,3,6,1,7,0,2,6,3,5))
    assert np.all(a.mask   == (0,1,0,0,0,0,0,0,1,0))
    a[Scalar((5,6,5),(True,False,False))] = Scalar((5,6,7))
    assert np.all(a.values == (5,3,6,1,7,7,6,6,3,5))
    assert np.all(a.mask   == (0,1,0,0,0,0,0,0,1,0))
    a[Scalar((5,6,5),(False,False,True))] = Scalar((5,6,7))
    assert np.all(a.values == (5,3,6,1,7,5,6,6,3,5))
    assert np.all(a.mask   == (0,1,0,0,0,0,0,0,1,0))
    a[:] = 9
    assert a == Scalar([9]*10)

    a = Scalar(((0,0,0),(0,0,0)))
    a[Pair((1,2))] = 1
    assert a == Scalar([[0,0,0],[0,0,1]])
    a[Pair((1,2),True)] = 2
    assert np.all(a.values == [[0,0,0],[0,0,1]])
    assert not np.any(a.mask)
    a[Pair((1,2),False)] = 2
    assert np.all(a.values == [[0,0,0],[0,0,2]])
    assert not np.any(a.mask)
    a[Pair((1,2))] = Scalar(0,True)
    assert np.all(a.values == [[0,0,0],[0,0,0]])
    assert np.all(a.mask   == [[0,0,0],[0,0,1]])
    a[Scalar(1,True)] = Scalar(1,True)
    assert np.all(a.values == [[0,0,0],[0,0,0]])
    assert np.all(a.mask   == [[0,0,0],[0,0,1]])
    a[Scalar(1,False)] = Scalar(1,False)
    assert np.all(a.values == [[0,0,0],[1,1,1]])
    assert not np.any(a.mask)
    a[Scalar(1)] = Scalar(1,True)
    assert np.all(a.values == [[0,0,0],[1,1,1]])
    assert np.all(a.mask   == [[0,0,0],[1,1,1]])
    a[Scalar(1)] = Scalar(2)
    assert np.all(a.values == [[0,0,0],[2,2,2]])
    assert not np.any(a.mask)
    a[Pair(((0,0),(0,1),(0,2)),True)] = 'abc'  # would raise an error if not for the mask
    assert np.all(a.values == [[0,0,0],[2,2,2]])
    assert not np.any(a.mask)
    a[Pair(((0,0),(1,1)))] = 7
    assert np.all(a.values == [[7,0,0],[2,7,2]])
    assert not np.any(a.mask)
    a[Pair(((0,0),(-1,-1)))] = 8
    assert np.all(a.values == [[8,0,0],[2,7,8]])
    assert not np.any(a.mask)

    a = Matrix(np.zeros(16).reshape(2,2,2,2))
    a[Pair((1,1))] = Matrix([[1,2],[3,4]])
    assert a == (Matrix([[[[0,0],[0,0]], [[0,0],[0,0]]],
                                [[[0,0],[0,0]], [[1,2],[3,4]]]]))
    assert (np.all(a.values == [[[[0,0],[0,0]], [[0,0],[0,0]]],
                                        [[[0,0],[0,0]], [[1,2],[3,4]]]]))
    assert np.all(a.mask   == False)
    a[Pair((1,1))] = Matrix([[4,5],[6,7]],True)
    assert (np.all(a.values == [[[[0,0],[0,0]], [[0,0],[0,0]]],
                                        [[[0,0],[0,0]], [[4,5],[6,7]]]]))
    assert np.all(a.mask   == [[0,0],[0,1]])
    a[Pair((1,1),True)] = Matrix([[5,5],[5,5]])
    assert (np.all(a.values == [[[[0,0],[0,0]], [[0,0],[0,0]]],
                                        [[[0,0],[0,0]], [[4,5],[6,7]]]]))
    assert np.all(a.mask   == [[0,0],[0,1]])
    a[Pair((1,1),False)] = Matrix([[5,5],[5,5]])
    assert (np.all(a.values == [[[[0,0],[0,0]], [[0,0],[0,0]]],
                                        [[[0,0],[0,0]], [[5,5],[5,5]]]]))
    assert not np.any(a.mask)
    a[...,1] = Matrix([[5,6],[7,8]])
    assert a == (Matrix([[[[0,0],[0,0]], [[5,6],[7,8]]],
                                [[[0,0],[0,0]], [[5,6],[7,8]]]]))
    assert not np.any(a.mask)
    a[...,Scalar(1)] = Matrix([[1,2],[3,4]])
    assert a == (Matrix([[[[0,0],[0,0]], [[1,2],[3,4]]],
                                [[[0,0],[0,0]], [[1,2],[3,4]]]]))
    assert not np.any(a.mask)
    a[...,Scalar(1,True)] = Matrix([[8,8],[8,8]])
    assert a == (Matrix([[[[0,0],[0,0]], [[1,2],[3,4]]],
                                [[[0,0],[0,0]], [[1,2],[3,4]]]]))
    assert not np.any(a.mask)
    a[...,1] = Matrix([[8,8],[8,8]])
    assert (np.all(a.values == [[[[0,0],[0,0]], [[8,8],[8,8]]],
                                        [[[0,0],[0,0]], [[8,8],[8,8]]]]))
    assert not np.any(a.mask)
    a[...,0] = Matrix([[9,9],[9,9]],True)
    assert (np.all(a.values[:,1] == [[[8,8],[8,8]],
                                              [[8,8],[8,8]]]))
    assert not np.any(a.mask[:,1])
    assert np.all(a.mask[:,0])
    a[Pair((0,0))] = Matrix([[5,5],[5,5]],False)
    a[Pair((-1,0))] = Matrix([[6,6],[6,6]],False)
    assert (np.all(a.values == [[[[5,5],[5,5]], [[8,8],[8,8]]],
                                        [[[6,6],[6,6]], [[8,8],[8,8]]]]))
    assert not np.any(a.mask)
    a[Pair((1,0))] = Matrix([[7,7],[7,7]],True)
    assert (np.all(a.values == [[[[5,5],[5,5]], [[8,8],[8,8]]],
                                        [[[7,7],[7,7]], [[8,8],[8,8]]]]))
    assert np.all(a.mask   == [[0,0],[1,0]])

    a = Scalar(0.)
    a[False] = 7
    assert a == 0.
    assert a.is_float()
    a[True] = 7
    assert a == 7.
    assert a.is_float()
    a[..., np.newaxis, False] = 3
    assert a == 7.
    assert a.is_float()
    a[..., np.newaxis, True] = 3
    assert a == 3.
    assert a.is_float()
    a = Scalar(0.)
    a.insert_deriv('xy', Scalar((2,3), drank=1))
    a[False] = 7
    assert a.d_dxy == Scalar((2,3), drank=1)
    a[..., True] = 7
    assert a.d_dxy == Scalar((0,0), drank=1)
    a = Scalar(0.)
    a.insert_deriv('xy', Scalar((2,3), drank=1))
    b = Scalar(7.)
    b.insert_deriv('ab', Scalar((4,3), drank=1))
    a[None, ..., False] = b
    assert a.d_dxy == Scalar((2,3), drank=1)
    assert 'ab' not in a.derivs
    a[None, ..., True] = b
    assert a.d_dxy == Scalar((0,0), drank=1)
    assert a.d_dab == Scalar((4,3), drank=1)

    # Additional coverage tests for missing lines

    a = Matrix(np.arange(24).reshape(2, 3, 2, 2))

    with pytest.raises(IndexError):
        # This will raise an IndexError, though the exact message may vary
        _ = a[0, 0, 0, 0, 0]  # Too many indices for the array shape

    a = Scalar(np.arange(24).reshape(2, 3, 4))
    idx = (Scalar([0, 1]), Ellipsis, Scalar([0, 2]))
    b = a[idx]
    assert b.shape == (2, 3)

    a = Scalar(7.)
    with pytest.raises(IndexError) as cm:
        a[0] = 5
    assert isinstance(cm.value, IndexError)

    a = Scalar([1., 2., 3.])
    a.insert_deriv('t', Scalar([10., 20., 30.]))
    b = Scalar(4.)  # Use a scalar value, not an array
    a[0] = b
    assert a.values[0] == 4.
    assert a.d_dt.values[0] == 0.

    a = Scalar(np.arange(24).reshape(2, 3, 4))
    idx = (Scalar([0, 1]), 1, Scalar([0, 2]))
    b = a[idx]

    assert b.shape == (2,)

    # moveaxis in __setitem__
    # Testing moveaxis in __setitem__ is complex due to shape matching requirements
    # The moveaxis logic in __getitem__ is tested above
    # For __setitem__, the moveaxis code paths are difficult to test without
    # triggering shape mismatches, so we skip a direct test here
    # The code paths are still exercised through other __setitem__ tests

    a = Scalar([1., 2., 3.])
    mask = np.array([True, False, True])
    b = Scalar([10., 20., 30.])
    a[mask] = b[mask]
    assert a.values[0] == 10.
    assert a.values[2] == 30.
    assert a.values[1] == 2.

    a = Scalar(np.arange(12).reshape(3, 4))
    idx = ([0, 1], [2, 3])
    b = a[idx]

    assert b.shape == (2,)

    a = Scalar([1., 2., 3.])
    with pytest.raises(IndexError) as cm:
        _ = a[..., ...]
    assert 'only have a single ellipsis' in str(cm.value)

    a = Scalar([1., 2., 3.])
    with pytest.raises(IndexError):
        # This raises an error about multiple ellipses
        # The correction < 0 case is rare and hard to trigger directly
        _ = a[..., 0, ...]

    a = Scalar([1., 2., 3.])
    with pytest.raises(IndexError) as cm:
        _ = a[Scalar(1.5)]
    assert 'floating-point indexing is not permitted' in str(cm.value)

    a = Scalar(np.arange(12).reshape(3, 4))
    with pytest.raises(IndexError) as cm:
        _ = a[Boolean(np.array([[True, False], [False, True]]))]
    assert 'boolean index did not match' in str(cm.value)

    a = Scalar(np.arange(12).reshape(3, 4))
    mask = Boolean(np.array([True, False, True]), mask=[False, True, False])
    b = a[mask]

    assert b.shape == (3, 4)
    assert np.all(b.mask[1])  # The second row should be masked

    a = Scalar(np.arange(12).reshape(3, 4))
    idx = Scalar([0, 2])
    b = a[idx]
    assert b.shape == (2, 4)
    assert np.allclose(b.values[0], a.values[0])
    assert np.allclose(b.values[1], a.values[2])

    a = Scalar(np.arange(12).reshape(3, 4))
    idx = Scalar([0, 5, 2])
    b = a[idx]
    assert b.shape == (3, 4)
    assert np.all(b.mask[1])  # Index 5 is out of bounds, so it should be masked

    a = Scalar([1., 2., 3.])
    with pytest.raises(IndexError) as cm:
        _ = a['invalid']
    assert 'invalid index type' in str(cm.value)


def test_indices_masked_index_when_every_element_of_the_axis_is_used() -> None:
    """A masked index value still yields a masked result when no axis element is spare."""

    a = Scalar([10., 11., 12.])
    index = Scalar([0, 1, 2, 0], [False, False, False, True])
    result = a[index]

    assert list(result.mask) == [False, False, False, True]
    assert result.values[0] == 10.
    assert result.values[1] == 11.
    assert result.values[2] == 12.


def test_indices_masked_index_avoids_the_elements_the_index_selects() -> None:
    """A masked index value is redirected away from the elements the index selects."""

    a = Scalar([10., 11., 12., 13.])
    index = Scalar([1, 2, 1], [False, False, True])
    result = a[index]

    assert list(result.mask) == [False, False, True]
    assert result.values[0] == 11.
    assert result.values[1] == 12.
    # The value under the mask is unspecified, but it must not alias an element that the
    # index genuinely selects
    assert result.values[2] not in (11., 12.)


##########################################################################################
