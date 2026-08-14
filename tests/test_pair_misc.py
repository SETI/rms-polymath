##########################################################################################
# tests/Test_Pair_misc.py
# Old Pair tests, updated by MRS 2/18/14
##########################################################################################

import numpy as np
import pytest

from polymath import Pair, Scalar


def test_pair_misc_basic_comparisons_and_indexing() -> None:
    """Basic comparisons and indexing."""

    pairs = Pair([[1,2],[3,4],[5,6]])
    assert pairs.numer == (2,)
    assert pairs.shape == (3,)
    assert pairs.rank == 1
    test = [[1,2],[3,4],[5,6]]
    assert pairs == test
    test = Pair(test)
    assert pairs == test
    assert (pairs == test)
    assert (pairs == test)
    assert (~(pairs != test)).all()
    assert (pairs == test).all() == True
    assert (pairs != test) == False
    assert (pairs == test) == (True,  True,  True)
    assert (pairs != test) == (False, False, False)
    assert (pairs == test).all() == Scalar(True)
    assert (pairs != test).all() == Scalar(False)
    assert (pairs == test) == Scalar((True,  True,  True))
    assert (pairs != test) == Scalar((False, False, False))
    assert pairs[0] == (1,2)
    assert pairs[0] == [1,2]
    assert pairs[0] == Pair([1,2])
    assert pairs[0:1] == (1,2)
    assert pairs[0:1] == [[1,2]]
    assert pairs[0:1] == Pair([[1,2]])
    assert pairs[0:2] == ((1,2),(3,4))
    assert pairs[0:2] == [[1,2],[3,4]]
    assert pairs[0:2] == Pair([[1,2],[3,4]])

    assert +pairs == pairs
    assert -pairs == Pair([[-1,-2],[-3,-4],(-5,-6)])

    pairs = Pair([[1,2],[3,4],[5,6]])
    assert pairs + (2,2) == [[3,4],[5,6],(7,8)]
    assert pairs + (2,2) == Pair([[3,4],[5,6],(7,8)])
    assert pairs - (2,2) == [[-1,0],[1,2],[3,4]]
    assert pairs - (2,2) == Pair([[-1,0],[1,2],[3,4]])
    assert pairs.element_mul((2,2)) == [[2,4],[6,8],[10,12]]
    assert pairs.element_mul((2,2)) == Pair([[2,4],[6,8],[10,12]])
    assert pairs.element_mul((1,2)) == [[1,4],[3,8],[5,12]]
    assert pairs.element_mul((1,2)) == Pair([[1,4],[3,8],[5,12]])
    assert pairs.element_mul(Pair((1,2))) == [[1,4],[3,8],[5,12]]
    assert pairs.element_mul(Pair((1,2))) == Pair([[1,4],[3,8],[5,12]])
    assert pairs * 2 == [[2,4],[6,8],[10,12]]
    assert pairs * 2 == [[2,4],[6,8],[10,12]]
    assert pairs * Scalar(2) == [[2,4],[6,8],[10,12]]
    assert pairs * Scalar(2) == [[2,4],[6,8],[10,12]]
    assert pairs * (1,2,3) == [[1,2],[6,8],[15,18]]
    assert pairs * Scalar((1,2,3)) == [[1,2],[6,8],[15,18]]
    assert pairs.element_div((2,2)) == [[0.5,1],[1.5,2],[2.5,3]]
    assert pairs.element_div((2,2)) == Pair([[0.5,1],[1.5,2],[2.5,3]])
    assert pairs.element_div((1,2)) == [[1,1],[3,2],[5,3]]
    assert pairs.element_div((1,2)) == Pair([[1,1],[3,2],[5,3]])
    assert pairs.element_div(Pair((1,2))) == [[1,1],[3,2],[5,3]]
    assert pairs.element_div(Pair((1,2))) == Pair([[1,1],[3,2],[5,3]])
    assert pairs / 2 == [[0.5,1],[1.5,2],[2.5,3]]
    assert pairs / 2 == Pair([[0.5,1],[1.5,2],[2.5,3]])
    assert pairs / Scalar(2) == [[0.5,1],[1.5,2],[2.5,3]]
    assert pairs / Scalar(2) == Pair([[0.5,1],[1.5,2],[2.5,3]])
    assert pairs / (1,2,2) == [[1,2],[1.5,2],[2.5,3]]
    assert pairs / Scalar((1,2,2)) == [[1,2],[1.5,2],[2.5,3]]
    with pytest.raises(TypeError):
        pairs.__add__(2)
    with pytest.raises(TypeError):
        pairs.__sub__(2)
    with pytest.raises(TypeError):
        pairs.__add__(Scalar(2))
    with pytest.raises(TypeError):
        pairs.__sub__(Scalar(2))

    test = pairs.copy()
    test += (2,2)
    assert test == [[3,4],[5,6],(7,8)]
    test -= (2,2)
    assert test == [[1,2],[3,4],[5,6]]
    test *= (1,2,3)
    assert test == [[1,2],[6,8],[15,18]]
    test //= (1,2,3)
    assert test == [[1,2],[3,4],[5,6]]
    test *= 2
    assert test == [[2,4],[6,8],[10,12]]
    test //= 2
    assert test == [[1,2],[3,4],[5,6]]
    test += Pair((2,2))
    assert test == [[3,4],[5,6],(7,8)]
    test -= Pair((2,2))
    assert test == [[1,2],[3,4],[5,6]]
    test *= Scalar((1,2,3))
    assert test == [[1,2],[6,8],[15,18]]
    test //= Scalar((1,2,3))
    assert test == [[1,2],[3,4],[5,6]]
    test *= Scalar(2)
    assert test == [[2,4],[6,8],[10,12]]
    test //= Scalar(2)
    assert test == [[1,2],[3,4],[5,6]]

    test = pairs.as_float()
    test += (2,2)
    assert test == [[3,4],[5,6],(7,8)]
    test -= (2,2)
    assert test == [[1,2],[3,4],[5,6]]
    test *= (1,2,3)
    assert test == [[1,2],[6,8],[15,18]]
    test /= (1,2,3)
    assert test == [[1,2],[3,4],[5,6]]
    test *= 2
    assert test == [[2,4],[6,8],[10,12]]
    test /= 2
    assert test == [[1,2],[3,4],[5,6]]
    test += Pair((2,2))
    assert test == [[3,4],[5,6],(7,8)]
    test -= Pair((2,2))
    assert test == [[1,2],[3,4],[5,6]]
    test *= Scalar((1,2,3))
    assert test == [[1,2],[6,8],[15,18]]
    test /= Scalar((1,2,3))
    assert test == [[1,2],[3,4],[5,6]]
    test *= Scalar(2)
    assert test == [[2,4],[6,8],[10,12]]
    test /= Scalar(2)
    assert test == [[1,2],[3,4],[5,6]]


def test_pair_misc_other_functions() -> None:
    """Other functions."""

    pairs = Pair([[1,2],[3,4],[5,6]])
    eps = 3.e-16
    lo = 1. - eps
    hi = 1. + eps

    assert pairs.to_scalar(0) == Scalar((1,3,5))
    assert pairs.to_scalar(1) == Scalar((2,4,6))
    assert pairs.to_scalar(-1) == Scalar((2,4,6))
    assert pairs.to_scalar(-2) == Scalar((1,3,5))

    assert pairs.to_scalars() == ((Scalar((1,3,5)),
                                          Scalar((2,4,6))))

    assert pairs.swapxy() == Pair(((2,1),(4,3),(6,5)))

    assert pairs.dot((1,0)) == pairs.to_scalar(0)
    assert pairs.dot((0,1)) == pairs.to_scalar(1)
    assert pairs.dot((1,1)) == pairs.to_scalar(0) + pairs.to_scalar(1)

    assert pairs.norm() == np.sqrt((5.,25.,61.))
    assert pairs.norm() == Scalar(np.sqrt((5.,25.,61.)))
    assert (pairs.unit().norm() > lo).all()
    assert (pairs.unit().norm() < hi).all()
    assert (pairs.sep(pairs.unit()) > -eps).all()
    assert (pairs.sep(pairs.unit()) <  eps).all()

    axes = Pair([(1,0),(0,1)])
    axes2 = axes.reshape((2,1))
    assert axes.cross(axes2) == [[0,-1],[1,0]]

    assert (axes.sep((1,1)) > np.pi/4. - eps).all()
    assert (axes.sep((1,1)) < np.pi/4. + eps).all()
    angles = np.arange(0., np.pi, 0.01)
    vecs = Pair.from_scalars(np.cos(angles), np.sin(angles))
    assert (Pair([2,0]).sep(vecs) > angles - 3*eps).all()
    assert (Pair([2,0]).sep(vecs) < angles + 3*eps).all()
    vecs = Pair.from_scalars(np.cos(angles), -np.sin(angles))
    assert (Pair([2,0]).sep(vecs) > angles - 3*eps).all()
    assert (Pair([2,0]).sep(vecs) < angles + 3*eps).all()

    # cross_scalars()
#         pair = Pair.cross_scalars(np.arange(10), np.arange(5))
#         self.assertEqual(pair.shape, [10,5])
#         self.assertTrue(np.all(pair.vals[9,:,0] == 9))
#         self.assertTrue(np.all(pair.vals[:,4,1] == 4))
#
#         pair = Pair.cross_scalars(np.arange(12).reshape(3,4), np.arange(5))
#         self.assertEqual(pair.shape, [3,4,5])
#         self.assertTrue(np.all(pair.vals[2,3,:,0] == 11))
#         self.assertTrue(np.all(pair.vals[:,:,4,1] == 4))


def test_pair_misc_new_tests_2_1_12_mrs() -> None:
    """New tests 2/1/12 (MRS)."""

    test = Pair(np.arange(6).reshape(3,2))
    assert str(test) == "Pair([0 1]\n [2 3]\n [4 5])"
    test =  Pair(np.arange(6).reshape(3,2), mask=[False, False, True])
    assert str(test) == "Pair([0 1]\n [2 3]\n [-- --]; mask)"
    assert str(test*2) == "Pair([0 2]\n [4 6]\n [-- --]; mask)"
    assert str(test/2) == "Pair([0.0 0.5]\n [1.0 1.5]\n [-- --]; mask)"
    assert str(test%2) == "Pair([0 1]\n [0 1]\n [-- --]; mask)"
    assert str(test + (1,0)) == "Pair([1 1]\n [3 3]\n [-- --]; mask)"
    assert str(test - (0,1)) == "Pair([0 0]\n [2 2]\n [-- --]; mask)"
    assert str(test + test) == "Pair([0 2]\n [4 6]\n [-- --]; mask)"
    assert str(test + np.arange(6).reshape(3,2)) == "Pair([0 2]\n [4 6]\n [-- --]; mask)"
    temp = Pair(np.arange(6).reshape(3,2), [True, False, False])
    assert str(test + temp) == "Pair([-- --]\n [4 6]\n [-- --]; mask)"
    assert str(test - 2*temp) == "Pair([-- --]\n [-2 -3]\n [-- --]; mask)"
    assert str(test.element_mul(temp)) == "Pair([-- --]\n [4 9]\n [-- --]; mask)"
    assert str(test.element_div(temp)) == "Pair([-- --]\n [1.0 1.0]\n [-- --]; mask)"
    temp = Pair(np.arange(6).reshape(3,2), [True, False, False])
    assert str(temp) == "Pair([-- --]\n [2 3]\n [4 5]; mask)"
    assert str(temp[0]) == "Pair(-- --; mask)"
    assert str(temp[1]) == "Pair(2 3)"
    assert str(temp[0:2]) == "Pair([-- --]\n [2 3]; mask)"
    assert str(temp[0:1]) == "Pair([-- --]; mask)"
    assert str(temp[1:2]) == "Pair([2 3])"
    test = Pair(np.arange(6).reshape(3,2))
    assert test == Pair(np.arange(6).reshape(3,2))
    mvals = test.mvals
    assert mvals.mask == np.ma.nomask
    assert test == mvals
    test = Pair(np.arange(6).reshape(3,2), [False, False, True])
    mvals = test.mvals
    assert str(mvals) == "[[0 1]\n [2 3]\n [-- --]]"
    assert test.mask.shape == (3,)
    assert mvals.mask.shape == (3,2)


##########################################################################################
