##########################################################################################
# tests/test_vector3_misc.py
# Old Vector3 tests, updated by MRS 2/18/14
##########################################################################################

import numpy as np
import pytest

from polymath import Boolean, Scalar, Vector3, Pair


def test_vector3_misc_basic_comparisons_and_indexing() -> None:
    """Basic comparisons and indexing."""

    np.random.seed(2222)

    vecs = Vector3([[1,2,3],[3,4,5],[5,6,7]])
    assert vecs.numer == (3,)
    assert vecs.shape == (3,)
    assert vecs.rank == 1
    test = [[1,2,3],[3,4,5],[5,6,7]]
    assert vecs == test
    test = Vector3(test)
    assert vecs == test
    assert (vecs == test)
    assert (vecs == test)
    assert (vecs == test) == True
    assert (vecs != test) == False
    assert (vecs == test) == (True,  True,  True)
    assert (vecs != test) == (False, False, False)
    assert (vecs == test) == Boolean(True)
    assert (vecs != test) == Boolean(False)
    assert (vecs == test) == Boolean((True,  True,  True))
    assert (vecs != test) == Boolean((False, False, False))
    assert (vecs == [1,2,3]) == Boolean((True, False, False))
    assert vecs[0] == (1,2,3)
    assert vecs[0] == [1,2,3]
    assert vecs[0] == Vector3([1,2,3])
    assert vecs[0:1] == (1,2,3)
    assert vecs[0:1] == [[1,2,3]]
    assert vecs[0:1] == Vector3([[1,2,3]])
    assert vecs[0:2] == ((1,2,3),(3,4,5))
    assert vecs[0:2] == [[1,2,3],[3,4,5]]
    assert vecs[0:2] == Vector3([[1,2,3],[3,4,5]])

    assert +vecs == vecs
    assert -vecs == Vector3([[-1,-2,-3],[-3,-4,-5],(-5,-6,-7)])

    vecs = Vector3([[1,2,3],[3,4,5],[5,6,7]])
    assert vecs + (0,1,2) == [[1,3,5],[3,5,7],(5,7,9)]
    assert vecs + (0,1,2) == Vector3([[1,3,5],[3,5,7],(5,7,9)])
    assert vecs - (0,1,2) == [[1,1,1],[3,3,3],[5,5,5]]
    assert vecs - (0,1,2) == Vector3([[1,1,1],[3,3,3],[5,5,5]])
    assert vecs.element_mul((1,2,3)) == [[1,4,9],[3,8,15],[5,12,21]]
    assert vecs.element_mul((1,2,3)) == Vector3([[1,4,9],[3,8,15],[5,12,21]])
    assert vecs.element_mul(Vector3((1,2,3))) == [[1,4,9],[3,8,15],[5,12,21]]
    assert vecs.element_mul(Vector3((1,2,3))) == Vector3([[1,4,9],[3,8,15],[5,12,21]])
    assert vecs * 2 == [[2,4,6],[6,8,10],[10,12,14]]
    assert vecs * 2 == Vector3([[2,4,6],[6,8,10],[10,12,14]])
    assert vecs * Scalar(2) == [[2,4,6],[6,8,10],[10,12,14]]
    assert vecs * Scalar(2) == (Vector3([[2,4,6],[6,8,10],
                                                             [10,12,14]]))
    assert vecs.element_div((1,1,2)) == [[1,2,1.5],[3,4,2.5],[5,6,3.5]]
    assert vecs.element_div(Vector3((1,1,2))) == [[1,2,1.5],[3,4,2.5],[5,6,3.5]]
    assert vecs / 2 == [[0.5,1,1.5],[1.5,2,2.5],[2.5,3,3.5]]
    assert vecs / Scalar(2) == [[0.5,1,1.5],[1.5,2,2.5], [2.5,3,3.5]]
    with pytest.raises(TypeError):
        vecs.__add__(1)
    with pytest.raises(TypeError):
        vecs.__add__(Scalar(1))
    with pytest.raises(ValueError):
        vecs.__add__((1,2))
    with pytest.raises(TypeError):
        vecs.__add__(Pair((1,2)))
    with pytest.raises(TypeError):
        vecs.__sub__(1)
    with pytest.raises(TypeError):
        vecs.__sub__(Scalar(1))
    with pytest.raises(ValueError):
        vecs.__sub__((1,2))
    with pytest.raises(TypeError):
        vecs.__sub__(Pair((1,2)))
    with pytest.raises(ValueError):
        vecs.__mul__((1,2))
    with pytest.raises(TypeError):
        vecs.__mul__(Pair((1,2)))
    with pytest.raises(ValueError):
        vecs.__truediv__((1,2))
    with pytest.raises(TypeError):
        vecs.__truediv__(Pair((1,2)))

    vecs = Vector3([[1,2,3],[3,4,5],[5,6,7]])
    test = vecs.copy()
    test += (1,2,3)
    assert test == [[2,4,6],[4,6,8],(6,8,10)]
    test -= (1,2,3)
    assert test == vecs
    test = test.element_mul((1,2,3))
    assert test == [[1,4,9],[3,8,15],[5,12,21]]
    test = test.element_div((1,2,3))
    assert test == vecs
    test *= 2
    assert test == [[2,4,6],[6,8,10],[10,12,14]]
    test /= 2
    assert test == vecs
    test *= Scalar(2)
    assert test == [[2,4,6],[6,8,10],[10,12,14]]
    test /= Scalar(2)
    assert test == vecs
    test *= Scalar((1,2,3))
    assert test == [[1,2,3],[6,8,10],[15,18,21]]
    test /= Scalar((1,2,3))
    assert test == vecs
    with pytest.raises(TypeError):
        test.__iadd__(Scalar(1))
    with pytest.raises(TypeError):
        test.__iadd__(1)
    with pytest.raises(ValueError):
        test.__iadd__((1,2))
    with pytest.raises(TypeError):
        test.__isub__(Scalar(1))
    with pytest.raises(TypeError):
        test.__isub__(1)
    with pytest.raises(ValueError):
        test.__isub__((1,2,3,4))
    with pytest.raises(TypeError):
        test.__imul__(Pair((1,2)))
    with pytest.raises(ValueError):
        test.__imul__((1,2,3,4))
    with pytest.raises(TypeError):
        test.__itruediv__(Pair((1,2)))
    with pytest.raises(ValueError):
        test.__itruediv__((1,2,3,4))

    # Other functions...

    assert vecs.to_scalar(0) == Scalar((1,3,5))
    assert vecs.to_scalar(1) == Scalar((2,4,6))
    assert vecs.to_scalar(2) == Scalar((3,5,7))
    assert vecs.to_scalar(-1) == Scalar((3,5,7))
    assert vecs.to_scalar(-2) == Scalar((2,4,6))
    assert vecs.to_scalar(-3) == Scalar((1,3,5))

    assert vecs.to_scalars() == ((Scalar((1,3,5)),
                                         Scalar((2,4,6)),
                                         Scalar((3,5,7))))

    assert vecs.dot((1,0,0)) == vecs.to_scalar(0)
    assert vecs.dot((0,1,0)) == vecs.to_scalar(1)
    assert vecs.dot((0,0,1)) == vecs.to_scalar(2)
    assert vecs.dot((1,1,0)) == vecs.to_scalar(0) + vecs.to_scalar(1)


def test_vector3_misc_norm() -> None:
    """norm()."""

    np.random.seed(2222)

    v = Vector3([[[1,2,3],[2,3,4]],[[0,1,2],[3,4,5]]])
    assert v.norm() == np.sqrt([[14,29],[5,50]])


def test_vector3_misc_cross_ucross() -> None:
    """cross(), ucross()."""

    np.random.seed(2222)

    a = Vector3([[[1,0,0]],[[0,2,0]],[[0,0,3]]])
    b = Vector3([ [0,3,3] , [2,0,2] , [1,1,0] ])
    axb = a.cross(b)
    assert a.shape == (3,1)
    assert b.shape == (3,)
    assert axb.shape == (3,3)
    assert axb[0,0] == ( 0,-3, 3)
    assert axb[0,1] == ( 0,-2, 0)
    assert axb[0,2] == ( 0, 0, 1)
    assert axb[1,0] == ( 6, 0, 0)
    assert axb[1,1] == ( 4, 0,-4)
    assert axb[1,2] == ( 0, 0,-2)
    assert axb[2,0] == (-9, 0, 0)
    assert axb[2,1] == ( 0, 6, 0)
    assert axb[2,2] == (-3, 3, 0)
    axb = a.ucross(b)
    assert axb[0,0] == Vector3(( 0,-3, 3)).unit()
    assert axb[0,1] == Vector3(( 0,-2, 0)).unit()
    assert axb[0,2] == Vector3(( 0, 0, 1)).unit()
    assert axb[1,0] == Vector3(( 6, 0, 0)).unit()
    assert axb[1,1] == Vector3(( 4, 0,-4)).unit()
    assert axb[1,2] == Vector3(( 0, 0,-2)).unit()
    assert axb[2,0] == Vector3((-9, 0, 0)).unit()
    assert axb[2,1] == Vector3(( 0, 6, 0)).unit()
    assert axb[2,2] == Vector3((-3, 3, 0)).unit()


def test_vector3_misc_perp_proj_sep() -> None:
    """perp, proj, sep."""

    np.random.seed(2222)
    eps = 3.e-16

    a = Vector3(np.random.rand(2,1,4,1,3))
    b = Vector3(np.random.rand(  3,4,2,3))
    aperp = a.perp(b)
    aproj = a.proj(b)
    assert aperp.shape == (2,3,4,2)
    assert aproj.shape == (2,3,4,2)
    eps = 3.e-14
    assert (aperp.sep(b) > np.pi/2 - eps).all()
    assert (aperp.sep(b) < np.pi/2 + eps).all()
    assert (aproj.sep(b) % np.pi > -eps).all()
    assert (aproj.sep(b) % np.pi <  eps).all()
    assert np.all((a - aperp - aproj).vals > -eps)
    assert np.all((a - aperp - aproj).vals <  eps)

    # Note: the sep(reverse=True) option is not tested here


def test_vector3_misc_new_tests_2_1_12_mrs() -> None:
    """New tests 2/1/12 (MRS)."""

    np.random.seed(2222)

    test = Vector3(np.arange(6).reshape(2,3))
    str_test = str(test).replace('  ', ' ').replace('[ ','[')
    assert str_test == "Vector3([0. 1. 2.]\n [3. 4. 5.])"
    test = Vector3(np.arange(6).reshape(2,3), mask = [True, False])
    assert str(test) == "Vector3([-- -- --]\n [3.0 4.0 5.0]; mask)"
    assert str(test*2) == "Vector3([-- -- --]\n [6.0 8.0 10.0]; mask)"
    assert str(test/2) == "Vector3([-- -- --]\n [1.5 2.0 2.5]; mask)"
    assert str(test + (1,0,2)) == "Vector3([-- -- --]\n [4.0 4.0 7.0]; mask)"
    assert str(test - (1,0,2)) == "Vector3([-- -- --]\n [2.0 4.0 3.0]; mask)"
    assert str(test - 2*test) == "Vector3([-- -- --]\n [-3.0 -4.0 -5.0]; mask)"
    assert str(test + np.arange(6).reshape(2,3)) == "Vector3([-- -- --]\n [6.0 8.0 10.0]; mask)"
    assert str(test[0]) == "Vector3(-- -- --; mask)"
    assert str(test[1]).replace('( ','(').replace('  ',' ') == "Vector3(3. 4. 5.)"
    assert str(test[0:2]) == "Vector3([-- -- --]\n [3.0 4.0 5.0]; mask)"
    assert str(test[0:1]) == "Vector3([-- -- --]; mask)"
    assert str(test[1:2]).replace('[ ','[').replace('  ',' ') == "Vector3([3. 4. 5.])"


##########################################################################################
