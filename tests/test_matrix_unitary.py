##########################################################################################
# tests/test_matrix_unitary.py
##########################################################################################

import numpy as np

from polymath import Matrix3, Matrix


def test_matrix_unitary_matrices_10_perturbed_from_unitary() -> None:
    """Matrices 10% perturbed from unitary."""

    np.random.seed(2163)

    N = 100
    SCALE = 0.1
    euler = (np.random.rand(N) * 2.*np.pi,
             np.random.rand(N) * 2.*np.pi,
             np.random.rand(N) * 2.*np.pi)
    a = Matrix(Matrix3.from_euler(*euler))
    a += SCALE * Matrix(np.random.randn(N,3,3))
    b = a.unitary()
    assert b.count_masked() == 0


def test_matrix_unitary_matrices_30_perturbed_from_unitary() -> None:
    """Matrices 30% perturbed from unitary."""

    np.random.seed(2163)

    N = 100
    SCALE = 0.3
    euler = (np.random.rand(N) * 2.*np.pi,
             np.random.rand(N) * 2.*np.pi,
             np.random.rand(N) * 2.*np.pi)
    a = Matrix(Matrix3.from_euler(*euler))
    a += SCALE * Matrix(np.random.randn(N,3,3))
    b = a.unitary()
    assert (b.count_masked() <= 30)


##########################################################################################
