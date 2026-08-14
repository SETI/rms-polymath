##########################################################################################
# tests/test_scalar_quadratic.py
##########################################################################################

import numpy as np

from polymath import Scalar


def test_scalar_quadratic_arrays_of_various_sizes() -> None:
    """Arrays of various sizes."""

    np.random.seed(7108)

    a = np.random.randn(8)
    b = np.random.randn(3,8)
    c = np.random.randn(4,1,1)
    (x0, x1) = Scalar.solve_quadratic(a, b, c)
    assert x0.shape == (4,3,8)
    assert (abs(x0.eval_quadratic(a,b,c)).median() < 1.e-15)
    assert (abs(x0.eval_quadratic(a,b,c)).max()    < 3.e-13)
    assert (abs(x1.eval_quadratic(a,b,c)).median() < 1.e-15)
    assert (abs(x1.eval_quadratic(a,b,c)).max()    < 3.e-13)
    assert np.all(x0.mask == x1.mask)


def test_scalar_quadratic_check_with_one_linear_case() -> None:
    """Check with one linear case."""

    np.random.seed(7108)

    a = np.random.randn(20)
    b = np.random.randn(20)
    c = np.random.randn(20)
    a[0] = 0.
    (x0, x1) = Scalar.solve_quadratic(a, b, c)
    assert (abs(x0.eval_quadratic(a,b,c)).median() < 1.e-15)
    assert (abs(x0.eval_quadratic(a,b,c)).max()    < 3.e-13)
    assert (abs(x1.eval_quadratic(a,b,c)).median() < 1.e-15)
    assert (abs(x1.eval_quadratic(a,b,c)).max()    < 3.e-13)
    assert np.all(x0[1:].mask == x1[1:].mask)
    assert np.all(x1[0].mask)


def test_scalar_quadratic_check_with_two_single_solution_quadratic_cases() -> None:
    """Check with two single-solution quadratic cases."""

    np.random.seed(7108)

    a = np.random.randn(20)
    b = np.random.randn(20)
    c = np.random.randn(20)
    (b[0], c[0]) = (0, 0)
    (a[1], b[1], c[1]) = (1, -2, 1)
    (x0, x1) = Scalar.solve_quadratic(a, b, c)
    assert (abs(x0.eval_quadratic(a,b,c)).median() < 1.e-15)
    assert (abs(x0.eval_quadratic(a,b,c)).max()    < 3.e-13)
    assert (abs(x1.eval_quadratic(a,b,c)).median() < 1.e-15)
    assert (abs(x1.eval_quadratic(a,b,c)).max()    < 3.e-13)
    assert x0[0] == 0.
    assert x0[1] == 1.
    assert np.all(x0[2:].mask == x1[2:].mask)
    assert np.all(x1[:2].mask)


def test_scalar_quadratic_single_values() -> None:
    """Single values."""

    np.random.seed(7108)

    for _k in range(100):
        a = np.random.randn()
        b = np.random.randn()
        c = np.random.randn()

        (x0, x1) = Scalar.solve_quadratic(a, b, c)

        assert x0.shape == ()
        if not x0.mask:
            assert (x0.eval_quadratic(a,b,c) < 3.e-13)
            assert (x1.eval_quadratic(a,b,c) < 3.e-13)
            assert (x0.mask == x1.mask)


def test_scalar_quadratic_single_linear_case() -> None:
    """Single linear case."""

    np.random.seed(7108)

    a = 0.
    b = np.random.randn()
    c = np.random.randn()
    (x0, x1) = Scalar.solve_quadratic(a, b, c)
    assert (x0.eval_quadratic(a,b,c) < 3.e-13)
    assert x1.mask


def test_scalar_quadratic_single_quadratic_case_with_one_solution() -> None:
    """Single quadratic case with one solution."""

    np.random.seed(7108)

    (x0, x1) = Scalar.solve_quadratic(1., -2., 1.)
    assert x0 == 1.
    assert x1.mask


def test_scalar_quadratic_derivatives_wrt_a() -> None:
    """Derivatives wrt a."""

    np.random.seed(7108)

    a = Scalar(np.random.randn(8))
    b = Scalar(np.random.randn(3,8))
    c = Scalar(np.random.randn(4,1,1))
    a.insert_deriv('t', Scalar(np.random.randn(8)))
    x = Scalar.solve_quadratic(a, b, c)
    assert (abs(x[0].eval_quadratic(a,b,c)).median() < 1.e-15)
    assert (abs(x[0].eval_quadratic(a,b,c)).max()    < 3.e-13)
    assert (abs(x[1].eval_quadratic(a,b,c)).median() < 1.e-15)
    assert (abs(x[1].eval_quadratic(a,b,c)).max()    < 1.e-13)
    assert ('t' in x[0].derivs)
    assert ('t' in x[1].derivs)
    da = 1.e-5 * a
    for k in range(2):
        dx = 0.5 * (Scalar.solve_quadratic(a + da, b, c)[k] -
                    Scalar.solve_quadratic(a - da, b, c)[k])
        assert (abs(dx * a.d_dt - x[k].d_dt * da).median() < 3.e-14)


def test_scalar_quadratic_derivatives_wrt_b() -> None:
    """Derivatives wrt b."""

    np.random.seed(7108)

    a = Scalar(np.random.randn(8))
    b = Scalar(np.random.randn(3,8))
    c = Scalar(np.random.randn(4,1,1))
    b.insert_deriv('t', Scalar(np.random.randn(3,8)))
    x = Scalar.solve_quadratic(a, b, c)
    assert (abs(x[0].eval_quadratic(a,b,c)).median() < 1.e-15)
    assert (abs(x[0].eval_quadratic(a,b,c)).max()    < 3.e-13)
    assert (abs(x[1].eval_quadratic(a,b,c)).median() < 1.e-15)
    assert (abs(x[1].eval_quadratic(a,b,c)).max()    < 3.e-13)
    assert ('t' in x[0].derivs)
    assert ('t' in x[1].derivs)
    db = 1.e-5 * b
    for k in range(2):
        dx = 0.5 * (Scalar.solve_quadratic(a, b+db, c)[k] -
                    Scalar.solve_quadratic(a, b-db, c)[k])
        assert (abs(dx * b.d_dt - x[k].d_dt * db).median() < 3.e-14)


def test_scalar_quadratic_derivatives_wrt_c() -> None:
    """Derivatives wrt c."""

    np.random.seed(7108)

    a = Scalar(np.random.randn(8))
    b = Scalar(np.random.randn(3,8))
    c = Scalar(np.random.randn(4,1,1))
    c.insert_deriv('t', Scalar(np.random.randn(4,1,1)))
    x = Scalar.solve_quadratic(a, b, c)
    assert (abs(x[0].eval_quadratic(a,b,c)).median() < 1.e-15)
    assert (abs(x[0].eval_quadratic(a,b,c)).max()    < 3.e-13)
    assert (abs(x[1].eval_quadratic(a,b,c)).median() < 1.e-15)
    assert (abs(x[1].eval_quadratic(a,b,c)).max()    < 3.e-13)
    assert ('t' in x[0].derivs)
    assert ('t' in x[1].derivs)
    dc = 1.e-5 * c
    for k in range(2):
        dx = 0.5 * (Scalar.solve_quadratic(a, b, c+dc)[k] -
                    Scalar.solve_quadratic(a, b, c-dc)[k])
        assert (abs(dx * c.d_dt - x[k].d_dt * dc).median() < 1.e-14)


##########################################################################################
