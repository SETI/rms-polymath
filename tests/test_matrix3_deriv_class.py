##########################################################################################
# tests/test_matrix3_deriv_class.py: Tests of the class used for a Matrix3 derivative
##########################################################################################

import numpy as np

from polymath import Matrix, Matrix3, Qube, Scalar, Vector, Vector3


def _rotations(n: int) -> Matrix3:
    """An array of n random rotation matrices."""

    angles = np.random.randn(n, 3)
    return Matrix3.from_euler(angles[:, 0], angles[:, 1], angles[:, 2], 'rzxz')


def _with_deriv(matrix: Matrix3, values: np.ndarray) -> Matrix3:
    """A copy of a Matrix3 carrying the given derivative."""

    obj = matrix.copy()
    obj.insert_deriv('t', Matrix(values))
    return obj


def test_matrix3_deriv_class_is_matrix() -> None:
    """A Matrix3 names Matrix as the class of its derivatives.

    A derivative of a rotation matrix is not itself a rotation matrix: it is not
    orthogonal, and unlike a rotation matrix it can be added to another one.
    """

    assert Matrix3._DERIV_CLASS is Matrix


def test_matrix3_deriv_class_substituted_in_a_class_list() -> None:
    """Matrix3 is replaced by Matrix among candidate classes for a derivative."""

    assert Qube._deriv_classes((Matrix3, Matrix)) == (Matrix, Matrix)
    assert Qube._deriv_classes(Matrix3) == (Matrix,)


def test_matrix3_deriv_class_leaves_other_classes_alone() -> None:
    """A class with no constraint of its own is its own derivative class."""

    assert Qube._deriv_classes((Vector3, Scalar)) == (Vector3, Scalar)
    assert Qube._deriv_classes(Vector) == (Vector,)


def test_matrix3_product_of_two_matrices_with_derivatives() -> None:
    """A product of two rotation matrices that both carry derivatives is computable.

    Both terms of the product rule are rotation matrix derivatives, and adding them
    together is what fails if they are typed as rotation matrices.
    """

    np.random.seed(2266)

    a = _with_deriv(_rotations(5), np.random.randn(5, 3, 3))
    b = _with_deriv(_rotations(5), np.random.randn(5, 3, 3))
    product = a * b

    assert type(product) is Matrix3
    assert ('t' in product.derivs)
    assert type(product.d_dt) is Matrix


def test_matrix3_product_derivative_obeys_the_product_rule() -> None:
    """The derivative of a matrix product matches a finite difference of the product."""

    np.random.seed(2266)

    da = np.random.randn(5, 3, 3)
    db = np.random.randn(5, 3, 3)
    a = _rotations(5)
    b = _rotations(5)
    product = _with_deriv(a, da) * _with_deriv(b, db)

    eps = 1.e-6
    ahead = Matrix(a.values + eps * da) * Matrix(b.values + eps * db)
    behind = Matrix(a.values - eps * da) * Matrix(b.values - eps * db)
    expected = (ahead.values - behind.values) / (2. * eps)

    assert np.abs(product.d_dt.values - expected).max() <= 1.e-8


def test_matrix3_product_with_one_derivative() -> None:
    """A product with a derivative on one side alone also yields a Matrix derivative."""

    np.random.seed(2266)

    a = _with_deriv(_rotations(5), np.random.randn(5, 3, 3))
    product = a * _rotations(5)

    assert type(product) is Matrix3
    assert type(product.d_dt) is Matrix


def test_matrix3_product_value_is_still_a_rotation() -> None:
    """The product itself remains a Matrix3, because it is still a rotation."""

    np.random.seed(2266)

    a = _with_deriv(_rotations(5), np.random.randn(5, 3, 3))
    b = _with_deriv(_rotations(5), np.random.randn(5, 3, 3))
    product = a * b
    identity = np.matmul(product.values, np.swapaxes(product.values, -1, -2))

    assert type(product) is Matrix3
    assert np.abs(identity - np.identity(3)).max() <= 1.e-14


def test_matrix3_times_vector3_keeps_a_vector3_derivative() -> None:
    """A class with no constraint of its own keeps its class in the derivative."""

    np.random.seed(2266)

    a = _with_deriv(_rotations(5), np.random.randn(5, 3, 3))
    v = Vector3(np.random.randn(5, 3))
    v.insert_deriv('t', Vector3(np.random.randn(5, 3)))
    product = a * v

    assert type(product) is Vector3
    assert type(product.d_dt) is Vector3


def test_matrix3_outer_product_with_derivatives_on_both_sides() -> None:
    """An outer product cast to Matrix3 gives its derivative the Matrix class."""

    np.random.seed(2266)

    u = Vector3(np.random.randn(5, 3))
    u.insert_deriv('t', Vector3(np.random.randn(5, 3)))
    v = Vector3(np.random.randn(5, 3))
    v.insert_deriv('t', Vector3(np.random.randn(5, 3)))
    result = Qube.outer(u, v, classes=(Matrix3, Matrix))

    assert type(result) is Matrix3
    assert type(result.d_dt) is Matrix


##########################################################################################
