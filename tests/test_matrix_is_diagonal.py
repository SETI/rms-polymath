##########################################################################################
# tests/test_matrix_is_diagonal.py
##########################################################################################

import numpy as np
import pytest

from polymath import Matrix


def test_matrix_is_diagonal_must_be_square() -> None:
    """must be square."""

    np.random.seed(6216)
    N = 4
    mats = np.random.randn(N,5,5)
    assert Matrix(mats).is_diagonal() == False
    mats = np.zeros((N,4,4))
    assert Matrix(mats).is_diagonal() == True

    mats = np.empty((N,2,3))
    with pytest.raises(ValueError):
        Matrix(mats).is_diagonal()

    mats = np.empty((N,3,3,2))
    with pytest.raises(ValueError):
        Matrix(mats, drank=1).is_diagonal()


def test_matrix_is_diagonal_delta_0() -> None:
    """delta = 0."""

    np.random.seed(6216)
    N = 4
    mats = np.random.randn(N,5,5)
    assert Matrix(mats).is_diagonal() == False
    mats = np.zeros((N,4,4))
    assert Matrix(mats).is_diagonal() == True

    mats = np.zeros((N,3,3))
    for i in range(N):
        for j in range(3):
            mats[i,j,j] = np.random.randn()
    assert Matrix(mats).is_diagonal() == True
    mats[0,0,1] = 1.e-14
    assert Matrix(mats).is_diagonal() == [False] + (N-1)*[True]

    assert Matrix(mats).is_diagonal(delta=3.e-13) == True

    assert Matrix(np.random.randn(N,5,5),True).is_diagonal() == True
    assert Matrix(np.random.randn(5,5),True).is_diagonal() == True

    assert Matrix(mats).is_diagonal() == [False] + (N-1)*[True]
    mask = [True] + (N-1) * [False]
    assert Matrix(mats,mask).is_diagonal() == True


##########################################################################################
