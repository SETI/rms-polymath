##########################################################################################
# tests/test_matrix3_euler.py
##########################################################################################

import numpy as np

from polymath import Matrix3


def test_matrix3_euler_round_trip_through_euler_angles_returns_the_same_matrix() -> None:
    """Conversion to Euler angles and back always returns the same matrix."""

    np.random.seed(5072)
    DEL = 1.e-12
    N = 30
    euler = (np.random.rand(N) * 2.*np.pi,
             np.random.rand(N) * 2.*np.pi,
             np.random.rand(N) * 2.*np.pi)
    a = Matrix3.from_euler(*euler)

    for code in Matrix3._AXES2TUPLE:
        angles = a.to_euler(axes=code)
        b = Matrix3.from_euler(*angles, axes=code)

        assert np.abs(a.values - b.values).max() < DEL


##########################################################################################
