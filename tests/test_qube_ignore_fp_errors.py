##########################################################################################
# tests/test_qube_ignore_fp_errors.py
##########################################################################################

import inspect

from typing import Any

import numpy as np

from polymath import Qube


@Qube.ignore_fp_errors
def _state() -> Any:
    """The NumPy error state seen inside a decorated function."""

    return np.geterr()


@Qube.ignore_fp_errors
def _nested_state() -> Any:
    """The error state seen by a decorated function called from another one."""

    return _state()


class _Decorated:
    """A class whose method carries the decorator."""

    @Qube.ignore_fp_errors
    def state(self, *, extra: int = 0) -> Any:
        """The error state seen inside a decorated method."""

        return np.geterr()


def test_qube_ignore_fp_errors_is_an_errstate() -> None:
    """The attribute is a NumPy errstate object."""

    assert isinstance(Qube.ignore_fp_errors, np.errstate)


def test_qube_ignore_fp_errors_ignores_divide() -> None:
    """A divide-by-zero is suppressed inside a decorated function."""

    assert _state()['divide'] == 'ignore'


def test_qube_ignore_fp_errors_ignores_overflow() -> None:
    """An overflow is suppressed inside a decorated function."""

    assert _state()['over'] == 'ignore'


def test_qube_ignore_fp_errors_ignores_underflow() -> None:
    """An underflow is suppressed inside a decorated function."""

    assert _state()['under'] == 'ignore'


def test_qube_ignore_fp_errors_ignores_invalid() -> None:
    """An invalid operation is suppressed inside a decorated function."""

    assert _state()['invalid'] == 'ignore'


def test_qube_ignore_fp_errors_covers_every_category() -> None:
    """Every category NumPy defines is suppressed."""

    assert sorted(_state().values()) == 4 * ['ignore']


def test_qube_ignore_fp_errors_is_reusable() -> None:
    """One shared object decorates repeated calls, unlike a `with` statement."""

    assert [_state()['divide'] for _ in range(3)] == 3 * ['ignore']


def test_qube_ignore_fp_errors_decorates_more_than_one_function() -> None:
    """The same object decorates several functions."""

    assert _nested_state()['divide'] == 'ignore'


def test_qube_ignore_fp_errors_nests() -> None:
    """A decorated function may call another one."""

    assert _nested_state() == _state()


def test_qube_ignore_fp_errors_decorates_a_method() -> None:
    """The decorator works on a method, keyword arguments included."""

    assert _Decorated().state(extra=1)['divide'] == 'ignore'


def test_qube_ignore_fp_errors_preserves_the_signature() -> None:
    """The wrapper keeps the signature, which stubtest and Sphinx both read."""

    assert list(inspect.signature(_Decorated.state).parameters) == ['self', 'extra']


def test_qube_ignore_fp_errors_preserves_the_docstring() -> None:
    """The wrapper keeps the docstring, which Sphinx reads."""

    assert _Decorated.state.__doc__ == 'The error state seen inside a decorated method.'


def test_qube_ignore_fp_errors_restores_the_previous_state() -> None:
    """The state after the call is the state before it."""

    before = np.geterr()
    _state()
    assert np.geterr() == before


def test_qube_ignore_fp_errors_overrides_a_caller_setting() -> None:
    """A caller's own setting does not apply inside a decorated function."""

    with np.errstate(invalid='raise'):
        assert _state()['invalid'] == 'ignore'


def test_qube_ignore_fp_errors_restores_a_caller_setting() -> None:
    """A caller's own setting is restored when the decorated function returns."""

    with np.errstate(invalid='raise'):
        _state()
        assert np.geterr()['invalid'] == 'raise'


def test_qube_ignore_fp_errors_restores_the_state_after_an_exception() -> None:
    """The state is restored even when the decorated function raises."""

    @Qube.ignore_fp_errors
    def boom() -> None:
        raise ValueError('boom')

    before = np.geterr()
    try:
        boom()
    except ValueError:
        pass

    assert np.geterr() == before

##########################################################################################
