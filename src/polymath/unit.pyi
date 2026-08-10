##########################################################################################
# polymath/unit.pyi
##########################################################################################
"""Type stub for :mod:`polymath.unit`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. These stubs describe the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
taken from the docstrings wherever those state one unambiguously, and are left as `Any`
where they do not, rather than guessed at.
"""

import builtins
from typing import Any, Self

from polymath.qube import Qube, _ShapeOrTuple

class Unit:
    ARCHOUR: Unit
    ARCHOURS: Unit
    ARCMIN: Unit
    ARCMINUTE: Unit
    ARCMINUTES: Unit
    ARCSEC: Unit
    ARCSECOND: Unit
    ARCSECONDS: Unit
    CENTIMETER: Unit
    CENTIMETERS: Unit
    CM: Unit
    CYCLE: Unit
    CYCLES: Unit
    D: Unit
    DAY: Unit
    DAYS: Unit
    DEG: Unit
    DEGREE: Unit
    DEGREES: Unit
    H: Unit
    HOUR: Unit
    HOURS: Unit
    KILOMETER: Unit
    KILOMETERS: Unit
    KM: Unit
    M: Unit
    METER: Unit
    METERS: Unit
    MICRON: Unit
    MICRONS: Unit
    MILLIMETER: Unit
    MILLIMETERS: Unit
    MILLIRAD: Unit
    MIN: Unit
    MINUTE: Unit
    MINUTES: Unit
    MM: Unit
    MRAD: Unit
    MS: Unit
    MSEC: Unit
    RAD: Unit
    RADIAN: Unit
    RADIANS: Unit
    REV: Unit
    REVS: Unit
    ROTATION: Unit
    ROTATIONS: Unit
    S: Unit
    SEC: Unit
    SECOND: Unit
    SECONDS: Unit
    STER: Unit
    UNITLESS: Unit
    def __copy__(self) -> Self: ...
    def __div__(self, arg: Any) -> Any: ...
    def __eq__(self, arg: object) -> Any: ...
    def __init__(self, exponents: _ShapeOrTuple, triple: _ShapeOrTuple,
        name: Any = ...) -> None: ...
    def __mul__(self, arg: Any) -> Unit: ...
    def __ne__(self, arg: object) -> Any: ...
    def __pow__(self, power: float | builtins.int | bool) -> Unit: ...
    def __rdiv__(self, arg: Any) -> Any: ...
    def __repr__(self) -> str: ...
    def __rmul__(self, arg: Any) -> Any: ...
    def __rtruediv__(self, arg: Any) -> Unit: ...
    def __str__(self) -> str: ...
    def __truediv__(self, arg: Any) -> Unit: ...
    @staticmethod
    def as_unit(arg: Any) -> Any: ...
    @staticmethod
    def can_match(first: Unit | None, second: Unit | None) -> bool: ...
    def convert(self, value: Any, unit: Unit | None, info: str = ...) -> Any: ...
    def copy(self) -> Unit: ...
    def create_name(self) -> str | dict[str, Qube]: ...
    @staticmethod
    def div_names(name1: Any, name2: Any) -> Any: ...
    @staticmethod
    def div_units(arg1: Unit | None, arg2: Unit | None, name: Any = ...) -> Any: ...
    @staticmethod
    def do_match(first: Unit | None, second: Unit | None) -> bool: ...
    def from_this(self, value: Any) -> Any: ...
    @staticmethod
    def from_unit(unit: Unit | None, value: Any) -> Any: ...
    @property
    def from_unit_factor(self) -> Any: ...
    def get_name(self) -> Any: ...
    def into_this(self, value: Any) -> Any: ...
    @staticmethod
    def into_unit(unit: Unit | None, value: Any) -> Any: ...
    @property
    def into_unit_factor(self) -> Any: ...
    @staticmethod
    def is_angle(arg: Unit | None) -> bool: ...
    @staticmethod
    def is_unitless(arg: Unit | None) -> bool: ...
    @staticmethod
    def mul_units(arg1: Unit | None, arg2: Unit | None, name: Any = ...) -> Any: ...
    @staticmethod
    def name_power(name: Any, power: float | builtins.int | bool) -> Any: ...
    @staticmethod
    def name_to_dict(name: Any) -> dict[str, Qube]: ...
    @staticmethod
    def name_to_str(namedict: Any) -> str: ...
    @staticmethod
    def require_angle(arg: Unit | None, info: str = ...) -> Any: ...
    @staticmethod
    def require_compatible(first: Unit | None, second: Unit | None,
        info: str = ...) -> Any: ...
    @staticmethod
    def require_match(first: Unit | None, second: Unit | None,
        info: str = ...) -> Any: ...
    @staticmethod
    def require_unitless(arg: Unit | None, info: str = ...) -> Any: ...
    def set_name(self, name: Any) -> Any: ...
    def sqrt(self, name: Any = ...) -> Unit: ...
    @staticmethod
    def sqrt_unit(unit: Unit | None, name: Any = ...) -> Any: ...
    @staticmethod
    def unit_power(unit: Unit | None, power: float | builtins.int | bool,
        name: Any = ...) -> Any: ...

##########################################################################################
