##########################################################################################
# tests/test_quaternion_euler.py
##########################################################################################

import numpy as np

from polymath import Quaternion


def test_quaternion_euler_quaternion_to_euler_and_back_one_quaternion() -> None:
    """Quaternion to Euler and back, one Quaternion."""

    np.random.seed(7599)

    for code in Quaternion._AXES2TUPLE:
        a = Quaternion(np.random.rand(4)).unit()
        euler = a.to_euler(code)
        b = Quaternion.from_euler(*euler, axes=code)
    DEL = 1.e-14
    for j in range(4):
        assert a.values[j] == b.values[j] or abs(a.values[j] - b.values[j]) <= DEL


def test_quaternion_euler_quaternion_to_euler_and_back_n_quaternions() -> None:
    """Quaternion to Euler and back, N Quaternions."""

    np.random.seed(7599)

    N = 100
    for code in Quaternion._AXES2TUPLE:
        a = Quaternion(np.random.rand(N,4)).unit()
        euler = a.to_euler(code)
        b = Quaternion.from_euler(*euler, axes=code)
    DEL = 1.e-14
    for i in range(N):
        for j in range(4):
            assert a.values[i,j] == b.values[i,j] or abs(a.values[i,j] - b.values[i,j]) <= DEL


def test_quaternion_euler_quaternion_to_matrix3_to_euler_and_back() -> None:
    """Quaternion to Matrix3 to Euler and back."""

    np.random.seed(7599)

    N = 100
    for code in Quaternion._AXES2TUPLE:
        a = Quaternion(np.random.rand(N,4)).unit()
        mats = a.to_matrix3()
        euler = mats.to_euler(code)
        b = Quaternion.from_euler(*euler, axes=code)
    DEL = 1.e-14
    for i in range(N):
        for j in range(4):
            assert a.values[i,j] == b.values[i,j] or abs(a.values[i,j] - b.values[i,j]) <= DEL


##########################################################################################
