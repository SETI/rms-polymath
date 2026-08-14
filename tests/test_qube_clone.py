##########################################################################################
# tests/test_qube_clone.py: Tests of Qube.clone and Qube.wod
##########################################################################################

import numpy as np

from polymath import Qube, Scalar, Unit, Vector3


def test_qube_clone_copies_every_descriptive_attribute() -> None:
    """A clone carries every attribute that describes the object."""

    np.random.seed(4409)

    a = Vector3(np.random.randn(5, 3), np.random.rand(5) < 0.5, unit=Unit.KM)
    b = a.clone()

    for attr in Qube._TRANSFERABLE_ATTRS:
        assert np.all(getattr(b, attr) == getattr(a, attr)), attr


def test_qube_clone_transfer_list_covers_the_whole_object() -> None:
    """No attribute of a constructed object is missing from the transfer list."""

    np.random.seed(4409)

    a = Vector3(np.random.randn(5, 3), unit=Unit.KM)
    a.insert_deriv('t', Vector3(np.random.randn(5, 3)))
    known = set(Qube._TRANSFERABLE_ATTRS) | set(Qube._OPTIONAL_ATTRS)
    known |= {'_derivs', '_cache'}
    extras = {name for name in a.__dict__ if not name.startswith('d_d')} - known

    assert extras == set()


def test_qube_clone_gives_the_copy_its_own_derivative_dictionary() -> None:
    """A clone does not share its derivative dictionary with the original."""

    np.random.seed(4409)

    a = Scalar(np.random.randn(5))
    a.insert_deriv('t', Scalar(np.random.randn(5)))
    b = a.clone()
    b.insert_deriv('u', Scalar(np.random.randn(5)))

    assert ('u' in b.derivs)
    assert ('u' not in a.derivs)


def test_qube_clone_without_recursion_drops_the_derivatives() -> None:
    """A clone made without recursion carries no derivatives."""

    np.random.seed(4409)

    a = Scalar(np.random.randn(5))
    a.insert_deriv('t', Scalar(np.random.randn(5)))

    assert not a.clone(recursive=False).derivs


def test_qube_clone_preserves_a_named_derivative() -> None:
    """A named derivative survives a clone made without recursion."""

    np.random.seed(4409)

    a = Scalar(np.random.randn(5))
    a.insert_deriv('t', Scalar(np.random.randn(5)))
    a.insert_deriv('u', Scalar(np.random.randn(5)))
    b = a.clone(recursive=False, preserve='t')

    assert ('t' in b.derivs)
    assert ('u' not in b.derivs)


def test_qube_clone_preserves_the_pickle_digits() -> None:
    """The pickle precision set on an object survives a clone.

    The pickler clones an object before encoding it, so losing this attribute would
    silently discard a precision setting.
    """

    np.random.seed(4409)

    a = Scalar(np.random.randn(5))
    a.set_pickle_digits(8, 'mean')
    b = a.clone()

    assert b.pickle_digits() == (8., 8.)
    assert b.pickle_reference() == ('mean', 'mean')


def test_qube_clone_of_an_object_without_pickle_digits_has_none() -> None:
    """An object that never set a pickle precision produces a clone without one."""

    np.random.seed(4409)

    b = Scalar(np.random.randn(5)).clone()

    assert not hasattr(b, '_pickle_digits')


def test_qube_wod_copies_the_values_and_drops_the_derivatives() -> None:
    """The derivative-free copy keeps the values and the mask but no derivatives."""

    np.random.seed(4409)

    values = np.random.randn(5, 3)
    mask = np.random.rand(5) < 0.5
    a = Vector3(values, mask, unit=Unit.KM)
    a.insert_deriv('t', Vector3(np.random.randn(5, 3)))
    b = a.wod

    assert not b.derivs
    assert np.all(b.values == values)
    assert np.all(b.mask == mask)
    assert b.unit_ == Unit.KM
    assert type(b) is Vector3


def test_qube_wod_of_an_object_without_derivatives_returns_it() -> None:
    """An object with no derivatives is its own derivative-free copy."""

    np.random.seed(4409)

    a = Vector3(np.random.randn(5, 3))

    assert a.wod is a


def test_qube_wod_leaves_the_original_intact() -> None:
    """Taking the derivative-free copy does not disturb the original."""

    np.random.seed(4409)

    a = Scalar(np.random.randn(5))
    a.insert_deriv('t', Scalar(np.random.randn(5)))
    b = a.wod

    assert not b.derivs
    assert ('t' in a.derivs)


##########################################################################################
