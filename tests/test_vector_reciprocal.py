##########################################################################################
# tests/test_vector_reciprocal.py
##########################################################################################

import numpy as np
import pytest

from polymath import Pair, Vector, Vector3


def test_vector_reciprocal_inverts_pairs_vector3s_and_random_vectors() -> None:
    """reciprocal() inverts the denominators of a Pair, a Vector3, and random 4x4 Vectors.

    The tolerance reflects float64 round-off, not reciprocal(): np.linalg.inv() gives a
    bit-identical error on the same data. The worst of the 100 random matrices has a
    condition number of about 3500, which puts the round-trip error at about 3e-13, so
    1e-12 leaves a modest margin.
    """

    np.random.seed(4912)
    vec = Pair([[1,0],[0,2]], drank=1)
    inverse = vec.reciprocal()
    assert inverse == [[1,0],[0,0.5]]
    assert type(inverse) is type(vec)
    vec = Vector3([[0,1,0],[0,0,2],[4,0,0]], drank=1)
    inverse = vec.reciprocal()
    assert inverse == [[0,0,0.25],[1,0,0],[0,0.5,0]]
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
