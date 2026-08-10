##########################################################################################
# tests/test_matrix3_pickle.py: Tests of Matrix3.__getstate__ and __setstate__
##########################################################################################

import numpy as np
import pickle
import pytest

from polymath import Matrix, Matrix3, Quaternion


def _rotations(shape: tuple[int, ...]) -> Matrix3:
    """An array of random rotation matrices with the given shape."""

    angles = np.random.randn(*(shape + (3,)))
    return Matrix3.from_euler(angles[..., 0], angles[..., 1], angles[..., 2], 'rzxz')


def _tangent(matrix: Matrix3, denom: tuple[int, ...] = ()) -> np.ndarray:
    """A derivative tangent to the space of rotations, as any rotation's must be.

    The derivative of a rotation matrix M always takes the form W M, where W is
    antisymmetric.
    """

    drank = len(denom)
    skew = np.random.randn(*(matrix.shape + (3, 3) + denom))
    skew = skew - np.swapaxes(skew, len(matrix.shape), len(matrix.shape) + 1)

    skew = np.moveaxis(skew, (-2 - drank, -1 - drank), (-2, -1))
    values = matrix.values.reshape(matrix.shape + drank * (1,) + (3, 3))
    return np.moveaxis(np.matmul(skew, values), (-2, -1), (-2 - drank, -1 - drank))


def _uses_quaternion(matrix: Matrix3) -> bool:
    """True if this object pickles via the quaternion encoding."""

    return 'QUATERNION_ENCODING' in matrix.__getstate__()


def test_matrix3_pickle_uses_the_quaternion_encoding() -> None:
    """A large array of rotation matrices is encoded as a quaternion."""

    np.random.seed(8021)

    assert _uses_quaternion(_rotations((500,)))


def test_matrix3_pickle_is_smaller_than_the_default_encoding() -> None:
    """The quaternion encoding is less than half the size of the default encoding."""

    np.random.seed(8021)

    matrix = _rotations((500,))
    quaternion_size = len(pickle.dumps(matrix))
    default_size = len(pickle.dumps(Matrix(matrix)))

    assert quaternion_size < 0.5 * default_size


def test_matrix3_pickle_round_trip() -> None:
    """Values survive the quaternion encoding to within the conversion precision."""

    np.random.seed(8021)

    matrix = _rotations((500,))
    restored = pickle.loads(pickle.dumps(matrix))

    assert type(restored) is Matrix3
    assert restored.shape == matrix.shape
    assert np.abs(restored.values - matrix.values).max() <= 1.e-14


def test_matrix3_pickle_round_trip_multidimensional() -> None:
    """A multidimensional array survives the quaternion encoding."""

    np.random.seed(8021)

    matrix = _rotations((20, 7))
    assert _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    assert restored.shape == (20, 7)
    assert np.abs(restored.values - matrix.values).max() <= 1.e-14


def test_matrix3_pickle_round_trip_masked() -> None:
    """A partially masked array survives the quaternion encoding."""

    np.random.seed(8021)

    matrix = _rotations((500,)).mask_where(np.arange(500) % 5 == 0)
    assert _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    antimask = matrix.antimask

    assert np.all(restored.mask == matrix.mask)
    assert np.abs(restored.values[antimask] - matrix.values[antimask]).max() <= 1.e-14


def test_matrix3_pickle_preserves_readonly() -> None:
    """The read-only status survives the quaternion encoding."""

    np.random.seed(8021)

    matrix = _rotations((500,)).as_readonly()
    restored = pickle.loads(pickle.dumps(matrix))

    assert restored.readonly


def test_matrix3_pickle_round_trip_at_180_degrees() -> None:
    """A 180-degree rotation, whose scalar quaternion component is zero, survives."""

    np.random.seed(8021)

    quaternion = Quaternion(np.zeros((500, 4)))
    quaternion.values[:, 1] = 1.            # (0, 1, 0, 0): 180 degrees about x
    matrix = quaternion.to_matrix3()
    assert _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    assert np.abs(restored.values - matrix.values).max() == 0.


@pytest.mark.parametrize('delta', [1.e-2, 1.e-6, 1.e-10, 0.])
def test_matrix3_pickle_precision_near_180_degrees(delta: float) -> None:
    """Precision does not degrade as a rotation approaches 180 degrees."""

    np.random.seed(8021)

    angles = np.zeros((500, 3))
    angles[:, 0] = np.pi - delta
    angles[:, 1] = np.linspace(0., 0.001, 500)
    matrix = Matrix3.from_euler(angles[:, 0], angles[:, 1], angles[:, 2], 'rzxz')

    restored = pickle.loads(pickle.dumps(matrix))
    assert np.abs(restored.values - matrix.values).max() <= 1.e-14


def test_matrix3_pickle_round_trip_derivative() -> None:
    """A derivative tangent to the space of rotations survives the encoding."""

    np.random.seed(8021)

    matrix = _rotations((500,))
    deriv = _tangent(matrix)
    matrix.insert_deriv('t', Matrix(deriv))
    assert _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    assert np.abs(restored.d_dt.values - deriv).max() <= 1.e-13


def test_matrix3_pickle_round_trip_derivative_with_denominator() -> None:
    """A derivative with a denominator survives the encoding."""

    np.random.seed(8021)

    matrix = _rotations((500,))
    deriv = _tangent(matrix, (2,))
    matrix.insert_deriv('uv', Matrix(deriv, drank=1))
    assert _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    assert restored.d_duv.denom == (2,)
    assert np.abs(restored.d_duv.values - deriv).max() <= 1.e-13


def test_matrix3_pickle_round_trip_masked_derivative() -> None:
    """A derivative of a partially masked array survives the encoding."""

    np.random.seed(8021)

    matrix = _rotations((500,))
    deriv = _tangent(matrix)
    matrix = matrix.mask_where(np.arange(500) % 3 == 0)
    matrix.insert_deriv('t', Matrix(deriv))
    assert _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    antimask = matrix.antimask
    assert np.abs(restored.d_dt.values[antimask] - deriv[antimask]).max() <= 1.e-13


@pytest.mark.parametrize('digits', ['double', 'single', 10, 7])
def test_matrix3_pickle_honors_pickle_digits(digits: object) -> None:
    """Every supported precision setting round-trips through the quaternion encoding."""

    np.random.seed(8021)

    matrix = _rotations((500,))
    matrix.set_pickle_digits(digits, 'fpzip')
    assert _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    tolerance = {'double': 1.e-14, 'single': 1.e-6}.get(digits, 1.e-6)
    assert np.abs(restored.values - matrix.values).max() <= tolerance


def test_matrix3_pickle_falls_back_when_small() -> None:
    """An object below the size cutoff uses the lossless default encoding."""

    np.random.seed(8021)

    matrix = _rotations((5,))
    assert not _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    assert np.abs(restored.values - matrix.values).max() == 0.


def test_matrix3_pickle_falls_back_when_fully_masked() -> None:
    """A fully masked object uses the default encoding."""

    np.random.seed(8021)

    matrix = _rotations((500,)).mask_where(np.ones(500, dtype='bool'))
    assert not _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    assert np.all(restored.mask)


def test_matrix3_pickle_falls_back_when_not_a_rotation() -> None:
    """A matrix that is not a proper rotation uses the lossless default encoding."""

    np.random.seed(8021)

    matrix = Matrix3(np.random.randn(500, 3, 3))
    assert not _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    assert np.abs(restored.values - matrix.values).max() == 0.


def test_matrix3_pickle_falls_back_when_reflected() -> None:
    """A matrix with determinant -1 uses the lossless default encoding."""

    np.random.seed(8021)

    values = _rotations((500,)).values.copy()
    values[:, 0] *= -1.                     # orthogonal, but determinant -1
    matrix = Matrix3(values)
    assert not _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    assert np.abs(restored.values - matrix.values).max() == 0.


def test_matrix3_pickle_falls_back_with_a_denominator() -> None:
    """An object with a denominator uses the lossless default encoding."""

    np.random.seed(8021)

    matrix = Matrix3(np.random.randn(500, 3, 3, 2), drank=1)
    assert not _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    assert np.abs(restored.values - matrix.values).max() == 0.


def test_matrix3_pickle_falls_back_with_a_nontangent_derivative() -> None:
    """A derivative off the space of rotations uses the lossless default encoding."""

    np.random.seed(8021)

    matrix = _rotations((500,))
    deriv = np.random.randn(500, 3, 3)
    matrix.insert_deriv('t', Matrix(deriv))
    assert not _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    assert np.abs(restored.d_dt.values - deriv).max() == 0.


def test_matrix3_pickle_falls_back_with_a_nontangent_denominator_derivative() -> None:
    """A derivative with a denominator off the space of rotations falls back."""

    np.random.seed(8021)

    matrix = _rotations((500,))
    deriv = np.random.randn(500, 3, 3, 2)
    matrix.insert_deriv('uv', Matrix(deriv, drank=1))
    assert not _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    assert np.abs(restored.d_duv.values - deriv).max() == 0.


def test_matrix3_pickle_accepts_a_zero_derivative() -> None:
    """A derivative of zero is tangent to the space of rotations."""

    np.random.seed(8021)

    matrix = _rotations((500,))
    matrix.insert_deriv('t', Matrix(np.zeros((500, 3, 3))))
    assert _uses_quaternion(matrix)

    restored = pickle.loads(pickle.dumps(matrix))
    assert np.abs(restored.d_dt.values).max() <= 1.e-13


##########################################################################################
