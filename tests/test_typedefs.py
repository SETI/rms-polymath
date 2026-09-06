##########################################################################################
# tests/test_typedefs.py
##########################################################################################

import ast
import types
import typing
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from polymath import Boolean, Matrix, Matrix3, Pair, Quaternion, Qube, Scalar, Vector, Vector3
from polymath import typedefs
from polymath.typedefs import (BooleanLike, MaskType, Matrix3Like, MatrixLike, PairLike,
                               QuaternionLike, QubeLike, ScalarLike, ValsType, Vector3Like,
                               VectorLike)

SRC = Path(typedefs.__file__).parent

# Every alias that names what a constructor accepts, with the class it belongs to and an
# item shape that class accepts
LIKE_ALIASES: dict[str, tuple[Any, type[Qube], tuple[int, ...]]] = {
    'BooleanLike':    (BooleanLike, Boolean, ()),
    'ScalarLike':     (ScalarLike, Scalar, ()),
    'PairLike':       (PairLike, Pair, (2,)),
    'VectorLike':     (VectorLike, Vector, (2,)),
    'Vector3Like':    (Vector3Like, Vector3, (3,)),
    'MatrixLike':     (MatrixLike, Matrix, (2, 2)),
    'Matrix3Like':    (Matrix3Like, Matrix3, (3, 3)),
    'QuaternionLike': (QuaternionLike, Quaternion, (4,)),
    'QubeLike':       (QubeLike, Qube, ()),
}

# The aliases whose classes hold one number per item, so a lone number converts
NUMBER_ALIASES = ('BooleanLike', 'ScalarLike', 'QubeLike')

NUMBER_TYPES = (float, int, bool, np.bool_)


def _array_member(alias: Any) -> Any:
    """The one member of a union alias that is a parameterized numpy.ndarray."""

    members = [arg for arg in typing.get_args(alias) if typing.get_origin(arg) is np.ndarray]
    assert len(members) == 1
    return members[0]


def _shape_args(alias: Any) -> tuple[Any, ...]:
    """The arguments of the shape type of the array member of a union alias."""

    shape, _ = typing.get_args(_array_member(alias))
    return typing.get_args(shape)


def test_all_names_every_public_alias() -> None:
    """The public names are exactly the eleven aliases, and each is a union."""

    expected = ['BooleanLike', 'MaskType', 'Matrix3Like', 'MatrixLike', 'PairLike',
                'QuaternionLike', 'QubeLike', 'ScalarLike', 'ValsType', 'Vector3Like',
                'VectorLike']
    assert typedefs.__all__ == expected

    public_unions = sorted(name for name in dir(typedefs)
                           if not name.startswith('_')
                           and isinstance(getattr(typedefs, name), types.UnionType))
    assert public_unions == expected


@pytest.mark.parametrize('name', sorted(LIKE_ALIASES))
def test_like_alias_accepts_qube(name: str) -> None:
    """Every constructor alias accepts any PolyMath object."""

    alias, _, _ = LIKE_ALIASES[name]
    assert Qube in typing.get_args(alias)


@pytest.mark.parametrize('name', sorted(LIKE_ALIASES))
def test_like_alias_accepts_nested_sequences(name: str) -> None:
    """Every constructor alias accepts a list or a tuple of numbers."""

    alias, _, _ = LIKE_ALIASES[name]
    origins = [typing.get_origin(arg) for arg in typing.get_args(alias)]
    assert origins.count(list) == 1
    assert origins.count(tuple) == 1


@pytest.mark.parametrize('name', sorted(LIKE_ALIASES))
def test_like_alias_array_dtype(name: str) -> None:
    """The array member of every constructor alias holds numbers or truth values."""

    alias, _, _ = LIKE_ALIASES[name]
    _, dtype = typing.get_args(_array_member(alias))
    assert typing.get_origin(dtype) is np.dtype
    assert typing.get_args(dtype) == (np.number[Any] | np.bool_,)


@pytest.mark.parametrize('name', NUMBER_ALIASES)
def test_number_alias_accepts_single_numbers(name: str) -> None:
    """The aliases for rank-0 classes accept a lone Python or NumPy number."""

    alias, _, _ = LIKE_ALIASES[name]
    members = typing.get_args(alias)
    for number_type in NUMBER_TYPES:
        assert number_type in members
    assert _shape_args(alias) == (int, Ellipsis)


@pytest.mark.parametrize('name', sorted(set(LIKE_ALIASES) - set(NUMBER_ALIASES)))
def test_item_alias_rejects_single_numbers(name: str) -> None:
    """The aliases for classes with item axes do not accept a lone number."""

    alias, _, _ = LIKE_ALIASES[name]
    members = typing.get_args(alias)
    for number_type in NUMBER_TYPES:
        assert number_type not in members


@pytest.mark.parametrize(
    ('name', 'trailing'),
    [('PairLike', (2,)), ('Vector3Like', (3,)), ('QuaternionLike', (4,)),
     ('Matrix3Like', (3, 3))],
)
def test_fixed_item_alias_names_trailing_axes(name: str, trailing: tuple[int, ...]) -> None:
    """An alias for a fixed item shape fixes the trailing axes of the array as literals."""

    alias, _, item = LIKE_ALIASES[name]
    leading, *literals = _shape_args(alias)
    assert typing.get_origin(leading) is tuple
    assert typing.get_args(leading) == (int, Ellipsis)
    assert [typing.get_args(literal) for literal in literals] == [(n,) for n in trailing]
    assert trailing == item


@pytest.mark.parametrize(('name', 'minimum_axes'), [('VectorLike', 1), ('MatrixLike', 2)])
def test_free_item_alias_requires_minimum_axes(name: str, minimum_axes: int) -> None:
    """An alias for a free item shape requires at least the class's rank in axes."""

    alias, _, item = LIKE_ALIASES[name]
    *leading, trailing = _shape_args(alias)
    assert leading == [int] * minimum_axes
    assert typing.get_origin(trailing) is tuple
    assert typing.get_args(trailing) == (int, Ellipsis)
    assert minimum_axes == len(item)


def test_valstype_names_what_values_returns() -> None:
    """ValsType is a number or a numeric array, matching the values property."""

    members = typing.get_args(ValsType)
    assert members[:4] == NUMBER_TYPES
    assert typing.get_origin(members[4]) is np.ndarray
    assert len(members) == 5

    assert type(Scalar(1.5).values) in members
    assert type(Scalar(2).values) in members
    assert type(Boolean(True).values) in members
    assert type(Scalar([1., 2.]).values) is np.ndarray


def test_masktype_names_what_mask_returns() -> None:
    """MaskType is a truth value or a boolean array, matching the mask property."""

    members = typing.get_args(MaskType)
    assert members[:2] == (bool, np.bool_)
    _, dtype = typing.get_args(members[2])
    assert typing.get_args(dtype) == (np.bool_,)
    assert len(members) == 3

    assert type(Scalar(1.).mask) is bool
    mask = Scalar([1., 2.], mask=[True, False]).mask
    assert type(mask) is np.ndarray
    assert mask.dtype == np.bool_


@pytest.mark.parametrize('name', sorted(LIKE_ALIASES))
def test_constructor_accepts_each_kind_of_member(name: str) -> None:
    """Each kind of value an alias names is accepted by the corresponding constructor."""

    np.random.seed(4471)
    _, cls, item = LIKE_ALIASES[name]
    array: Any = np.random.randn(2, *item)
    if cls is Boolean:
        array = array > 0.

    from_array = cls(array)
    assert from_array.shape == (2,)
    assert from_array.numer == item

    from_list = cls(array.tolist())
    assert from_list.shape == (2,)
    assert np.all(from_list.values == from_array.values)

    from_qube = cls(from_array)
    assert type(from_qube) is cls
    assert np.all(from_qube.values == from_array.values)

    if name in NUMBER_ALIASES:
        single = cls(True) if cls is Boolean else cls(1.5)
        assert single.shape == ()


def test_aliases_usable_in_annotations() -> None:
    """An alias is an ordinary object that an annotation can refer to directly."""

    def speed(velocity: Vector3Like) -> Scalar:
        return Vector3.as_vector3(velocity).norm()

    assert speed.__annotations__['velocity'] is Vector3Like
    assert speed([3., 4., 0.]) == 5.
    assert speed(np.array([0., 0., 2.])) == 2.
    assert speed(Vector3.XAXIS) == 1.


def _alias_definitions(path: Path) -> dict[str, str]:
    """Every type alias assigned in a module, keyed by name, as a normalized source dump."""

    tree = ast.parse(path.read_text())
    aliases = {}
    for node in tree.body:
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and isinstance(node.annotation, ast.Name)
                and node.annotation.id == 'TypeAlias' and node.value is not None):
            aliases[node.target.id] = ast.dump(node.value)
    return aliases


def _all_list(path: Path) -> list[str]:
    """The value of a module's __all__ list, read from its source."""

    tree = ast.parse(path.read_text())
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == '__all__' and isinstance(node.value, ast.List)):
            return [ast.literal_eval(elt) for elt in node.value.elts]
    raise AssertionError(f'no __all__ in {path}')


def test_stub_mirrors_module() -> None:
    """typedefs.pyi defines the same aliases, in the same terms, as typedefs.py."""

    module = SRC / 'typedefs.py'
    stub = SRC / 'typedefs.pyi'
    assert _all_list(stub) == _all_list(module)
    assert _alias_definitions(stub) == _alias_definitions(module)
    assert set(_alias_definitions(module)) >= set(typedefs.__all__)


def test_stub_takes_qube_from_the_package() -> None:
    """The stub imports Qube from the package, the only supported import path."""

    tree = ast.parse((SRC / 'typedefs.pyi').read_text())
    imports = [(node.module, [alias.name for alias in node.names])
               for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert ('polymath', ['Qube']) in imports
    assert all(module != 'polymath.qube' for module, _ in imports)
