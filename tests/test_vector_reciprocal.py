##########################################################################################
# tests/test_vector_reciprocal.py
##########################################################################################

import numpy as np
import pytest

from polymath import Pair, Vector, Vector3


def test_vector_reciprocal_print_np_abs_diffs_max_the_tolerance_is_set_by_float64_round() -> None:
    """print(np.abs(diffs).max()) # The tolerance is set by float64 round-off, not by any property of reciprocal(): # np.linalg.inv() on this same seeded data gives a bit-identical error. The worst # of these 100 random matrices has a condition number of ~3500, which puts the # round-trip error at ~3.e-13, so 1.e-12 leaves a modest safety margin."""

    np.random.seed(4912)
    vec = Pair([[1,0],[0,2]], drank=1)
    inverse = vec.reciprocal()
    assert np.all(inverse == [[1,0],[0,0.5]])
    assert type(inverse) is type(vec)
    vec = Vector3([[0,1,0],[0,0,2],[4,0,0]], drank=1)
    inverse = vec.reciprocal()
    assert np.all(inverse == [[0,0,0.25],[1,0,0],[0,0.5,0]])
    assert type(inverse) is type(vec)
    N = 100
    vec = Vector(np.random.randn(N,4,4), drank=1)
    inverse = vec.reciprocal()
    product = vec.vals @ inverse.vals
    diffs = product - [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

    assert (np.abs(diffs).max() < 1.e-12)

    vec = Pair(np.zeros((2,2)), drank=1)
    with pytest.raises(ValueError) as cm:
        inverse = vec.reciprocal(nozeros=True)
    assert str(cm.value) == 'Matrix.inverse() input is singular'
    inverse = vec.reciprocal()
    assert inverse.mask

    with pytest.raises(TypeError) as cm:
        inverse = Vector3(np.arange(9).reshape(3,3)).reciprocal()
    assert str(cm.value) == ('Vector3.reciprocal() is not supported '
                                        'unless it represents a Jacobian')


##########################################################################################
