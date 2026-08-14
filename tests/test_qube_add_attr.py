##########################################################################################
# tests/test_qube_add_attr.py: Tests of Qube.add_attr
##########################################################################################

import copy
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from polymath import Qube, Scalar, Vector, Vector3


def attr(obj: Qube, name: str) -> Any:
    """The value of an attribute that exists only at run time.

    An attribute added by :meth:`Qube.add_attr` is invisible to a type checker, because
    the stubs cannot describe it, so it is read indirectly here.

    Parameters:
        obj (Qube): The object carrying the attribute.
        name (str): The name of the attribute.

    Returns:
        Any: The value of the attribute.
    """

    return getattr(obj, name)


def test_qube_add_attr_assigns_the_value() -> None:
    """An added attribute is readable under its own name."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label', 'north')

    assert attr(a, 'label') == 'north'


def test_qube_add_attr_defaults_to_none() -> None:
    """An added attribute defaults to a value of None."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label')

    assert attr(a, 'label') is None


def test_qube_add_attr_returns_the_object() -> None:
    """The method returns the object to which the attribute was added."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))

    assert a.add_attr('label', 'north') is a


def test_qube_add_attr_replaces_a_value_it_added_before() -> None:
    """An attribute added by this method can be given a new value."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label', 'north')
    a.add_attr('label', 'south')

    assert attr(a, 'label') == 'south'


def test_qube_add_attr_allows_direct_assignment_afterward() -> None:
    """An added attribute is writable like any other attribute."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label', 'north')
    a.label = 'south'  # type: ignore[attr-defined]  # add_attr() created it above

    assert attr(a.clone(), 'label') == 'south'


@pytest.mark.parametrize('name', ['shape', 'clone', 'derivs', '_values', '_added_attrs'])
def test_qube_add_attr_refuses_to_shadow_an_existing_attribute(name: str) -> None:
    """An attribute that the object already has cannot be replaced."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))

    with pytest.raises(ValueError, match=f'attribute "{name}" already exists'):
        a.add_attr(name, 'north')


@pytest.mark.parametrize('name', ['d_d', 'd_dt', 'd_dsomething'])
def test_qube_add_attr_refuses_a_derivative_name(name: str) -> None:
    """A name beginning with "d_d" is reserved for derivatives."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))

    with pytest.raises(ValueError, match='reserved for derivatives'):
        a.add_attr(name, 'north')


def test_qube_add_attr_allows_a_name_that_merely_starts_with_d() -> None:
    """A name that falls short of the "d_d" prefix is allowed."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('d_t', 'north')

    assert attr(a, 'd_t') == 'north'


def test_qube_add_attr_requires_a_string_name() -> None:
    """A name that is not a string raises TypeError."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))

    with pytest.raises(TypeError, match='attribute name is not a string'):
        a.add_attr(7, 'north')  # type: ignore[arg-type]  # deliberately not a string


@pytest.mark.parametrize('name', ['', 'two words', '9lives'])
def test_qube_add_attr_requires_an_identifier(name: str) -> None:
    """A name that is not a valid Python identifier raises ValueError."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))

    with pytest.raises(ValueError, match='invalid attribute name'):
        a.add_attr(name, 'north')


def test_qube_add_attr_survives_a_clone() -> None:
    """A clone carries the added attribute."""

    np.random.seed(2701)

    a = Vector3(np.random.randn(5, 3))
    a.add_attr('label', 'north')

    assert attr(a.clone(), 'label') == 'north'


def test_qube_add_attr_survives_a_clone_without_derivatives() -> None:
    """A clone made without recursion carries the added attribute."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.insert_deriv('t', Scalar(np.random.randn(5)))
    a.add_attr('label', 'north')

    assert attr(a.clone(recursive=False), 'label') == 'north'


def test_qube_add_attr_survives_a_copy() -> None:
    """A deep copy carries the added attribute."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label', 'north')

    assert attr(a.copy(), 'label') == 'north'


def test_qube_add_attr_survives_a_readonly_copy() -> None:
    """A read-only copy carries the added attribute."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5)).as_readonly()
    a.add_attr('label', 'north')

    assert attr(a.copy(readonly=True), 'label') == 'north'


def test_qube_add_attr_survives_the_copy_module() -> None:
    """A copy made by the copy module carries the added attribute."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label', 'north')

    assert attr(copy.copy(a), 'label') == 'north'


def test_qube_add_attr_survives_a_deepcopy() -> None:
    """A deep copy made by the copy module carries the added attribute."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label', 'north')

    assert attr(copy.deepcopy(a), 'label') == 'north'


def test_qube_add_attr_survives_a_pickle_round_trip() -> None:
    """An unpickled object carries the added attribute."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label', 'north')

    assert attr(pickle.loads(pickle.dumps(a)), 'label') == 'north'


def test_qube_add_attr_survives_the_wod_property() -> None:
    """The derivative-free version of an object carries the added attribute."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.insert_deriv('t', Scalar(np.random.randn(5)))
    a.add_attr('label', 'north')

    assert attr(a.wod, 'label') == 'north'


def test_qube_add_attr_invalidates_a_cached_wod() -> None:
    """A copy cached before the attribute was added is not returned afterward."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.insert_deriv('t', Scalar(np.random.randn(5)))
    _ = a.wod                                   # cache the derivative-free version
    a.add_attr('label', 'north')

    assert attr(a.wod, 'label') == 'north'


NEW_VALUE_OPS: list[tuple[str, Callable[[Scalar], Any]]] = [
    ('neg',      lambda a: -a),
    ('abs',      lambda a: abs(a)),
    ('add',      lambda a: a + 1),
    ('sub',      lambda a: a - 1),
    ('mul',      lambda a: a * 2),
    ('div',      lambda a: a / 2),
    ('floordiv', lambda a: a // 2),
    ('mod',      lambda a: a % 2),
    ('constant', lambda a: a.as_all_constant()),
]


@pytest.mark.parametrize(('name', 'op'), NEW_VALUE_OPS, ids=[n for n, _ in NEW_VALUE_OPS])
def test_qube_add_attr_is_dropped_by_an_operation_on_the_values(
        name: str, op: Callable[[Scalar], Any]) -> None:
    """An operation that computes new values does not carry the added attribute."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label', 'north')

    assert not hasattr(op(a), 'label')


def test_qube_add_attr_is_dropped_by_the_integer_conversion_of_a_vector() -> None:
    """Conversion to integer indices does not carry the added attribute."""

    a = Vector([[-1, 2, 3], [4, 5, 6]])
    a.add_attr('label', 'north')

    assert not hasattr(a.int(), 'label')


def test_qube_add_attr_survives_the_unary_plus_operator() -> None:
    """The unary "+" operator is a copy, so it carries the added attribute."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label', 'north')

    assert attr(+a, 'label') == 'north'


def test_qube_add_attr_survives_a_change_of_mask() -> None:
    """An operation that changes only the mask carries the added attribute."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label', 'north')

    assert attr(a.as_all_masked(), 'label') == 'north'


def test_qube_add_attr_on_a_derivative_survives_an_operation_on_the_values() -> None:
    """An attribute added to a derivative is carried by an operation that keeps it."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.insert_deriv('t', Scalar(np.random.randn(5)))
    a.derivs['t'].add_attr('label', 'north')

    assert attr((a + 1).derivs['t'], 'label') == 'north'


def test_qube_add_attr_leaves_the_original_alone_when_a_clone_adds_one() -> None:
    """An attribute added to a clone does not appear on the original."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    b = a.clone()
    b.add_attr('label', 'north')

    assert not hasattr(a, 'label')


def test_qube_add_attr_leaves_the_clone_alone_when_the_original_adds_one() -> None:
    """An attribute added after a clone was made does not appear on the clone."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label', 'north')
    b = a.clone()
    a.add_attr('extra', 'south')

    assert not hasattr(b, 'extra')


def test_qube_add_attr_leaves_the_clone_alone_when_the_original_is_reassigned() -> None:
    """A value assigned after a clone was made does not appear on the clone."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    a.add_attr('label', 'north')
    b = a.clone()
    a.add_attr('label', 'south')

    assert attr(b, 'label') == 'north'


def test_qube_add_attr_carries_the_value_by_reference() -> None:
    """A clone shares the value of an added attribute with the original."""

    np.random.seed(2701)

    a = Scalar(np.random.randn(5))
    values = [1, 2, 3]
    a.add_attr('label', values)

    assert attr(a.clone(), 'label') is values

##########################################################################################
