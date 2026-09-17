##########################################################################################
# tests/test_qube_errstate_independence.py: detection of an out-of-domain value does not
# depend on how the caller has configured NumPy's floating-point error reporting.
#
# Each of these operations reports a bad input by raising a ValueError. It must do so
# whether the caller has asked NumPy to ignore floating-point errors, warn about them, or
# raise on them, so the guard cannot rely on a warning being emitted.
##########################################################################################

from collections.abc import Callable, Iterator
from typing import Any, Literal

import numpy as np
import pytest

from polymath import Matrix, Scalar, Vector

# numpy.errstate accepts only these words, as a Literal in NumPy's own stubs
_ErrKind = Literal['ignore', 'warn', 'print', 'raise']

SETTINGS: tuple[_ErrKind, ...] = ('ignore', 'warn', 'print', 'raise')

OPERATIONS: dict[str, Callable[[], Any]] = {
    'arcsin': lambda: Scalar([2.]).arcsin(check=False),
    'arccos': lambda: Scalar([2.]).arccos(check=False),
    'sqrt': lambda: Scalar([-1.]).sqrt(check=False),
    'log_of_zero': lambda: Scalar([0.]).log(check=False),
    'log_of_negative': lambda: Scalar([-1.]).log(check=False),
    'exp': lambda: Scalar([1000.]).exp(check=False),
    'reciprocal': lambda: Scalar([0.]).reciprocal(nozeros=True),
    'reciprocal_shapeless': lambda: Scalar(0.).reciprocal(nozeros=True),
    'inverse': lambda: Matrix([[1., 0.], [0., 0.]]).inverse(nozeros=True),
    'solve': lambda: Matrix([[1., 0.], [0., 0.]]).solve(Vector([1., 1.]), nozeros=True),
}


def _cases() -> Iterator[tuple[str, _ErrKind]]:
    """Every operation paired with every NumPy error setting."""

    for name in OPERATIONS:
        for setting in SETTINGS:
            yield (name, setting)


@pytest.mark.parametrize(('name', 'setting'), list(_cases()),
                         ids=[f'{n}-{s}' for n, s in _cases()])
def test_qube_errstate_independence_raises_value_error(name: str,
                                                      setting: _ErrKind) -> None:
    """A bad input raises ValueError under any NumPy error setting."""

    with np.errstate(all=setting), pytest.raises(ValueError):
        OPERATIONS[name]()


def test_qube_errstate_independence_of_the_default_state() -> None:
    """The same holds with NumPy left in its default state."""

    with pytest.raises(ValueError, match='outside domain'):
        Scalar([2.]).arccos(check=False)


@pytest.mark.parametrize('setting', SETTINGS)
def test_qube_errstate_independence_of_power_suppression(setting: _ErrKind) -> None:
    """Zero to a negative power is masked, not raised, under any NumPy error setting."""

    with np.errstate(all=setting):
        result = Scalar([0., 2.]) ** Scalar([-1., 2.])

    assert np.array_equal(result.mask, [True, False])


@pytest.mark.parametrize('setting', SETTINGS)
def test_qube_errstate_independence_of_power_values(setting: _ErrKind) -> None:
    """The valid element of that power survives under any NumPy error setting."""

    with np.errstate(all=setting):
        result = Scalar([0., 2.]) ** Scalar([-1., 2.])

    assert result.vals[1] == 4.


@pytest.mark.parametrize('setting', SETTINGS)
def test_qube_errstate_independence_restores_the_caller_setting(setting: _ErrKind) -> None:
    """An operation leaves the caller's NumPy error setting as it found it."""

    with np.errstate(all=setting):
        before = np.geterr()
        with pytest.raises(ValueError):
            Scalar([2.]).arccos(check=False)

        assert np.geterr() == before


@pytest.mark.parametrize('setting', SETTINGS)
def test_qube_errstate_independence_of_the_checked_path(setting: _ErrKind) -> None:
    """With check=True the bad value is masked instead, under any error setting."""

    with np.errstate(all=setting):
        result = Scalar([2., 0.5]).arccos()

    assert np.array_equal(result.mask, [True, False])

##########################################################################################
