##########################################################################################
# tests/test_matrix3_euler.py
##########################################################################################

import numpy as np

from polymath import Matrix3


def test_matrix3_euler_conversion_to_euler_angles_and_back_always_returns_the_same_() -> None:
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
