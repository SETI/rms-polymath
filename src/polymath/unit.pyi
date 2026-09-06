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

from typing import Any


__all__ = ['Unit']

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
    def __copy__(self) -> Unit: ...
    def __div__(self, arg: Unit | float | int) -> None: ...
    def __eq__(self, arg: object) -> bool: ...
    def __init__(self, exponents: tuple[int, int, int], triple: tuple[int, int, int],
        name: str | dict | None = ...) -> None: ...  # type: ignore[type-arg]
    def __mul__(self, arg: Unit | float | int) -> None: ...
    def __ne__(self, arg: object) -> bool: ...
    def __pow__(self, power: int | float) -> Unit: ...
    def __rdiv__(self, arg: float | int | None) -> Unit: ...
    def __repr__(self) -> str: ...
    def __rmul__(self, arg: Unit | float | int) -> None: ...
    def __rtruediv__(self, arg: float | int | None) -> Unit: ...
    def __str__(self) -> str: ...
    def __truediv__(self, arg: Unit | float | int) -> None: ...
    @staticmethod
    def as_unit(arg: Unit | str | None) -> Unit | None: ...
    @staticmethod
    def can_match(first: Unit | None, second: Unit | None) -> bool: ...
    def convert(self, value: Any, unit: Unit | None, info: str = ...) -> Any: ...
    def copy(self) -> Unit: ...
    def create_name(self) -> str | dict: ...  # type: ignore[type-arg]
    @staticmethod
    def div_units(arg1: Unit | None, arg2: Unit | None) -> Unit | None: ...
    @staticmethod
    def do_match(first: Unit | None, second: Unit | None) -> bool: ...
    def from_this(self, value: Any) -> Any: ...
    @staticmethod
    def from_unit(unit: Unit | None, value: Any) -> Any: ...
    @property
    def from_unit_factor(self) -> float | int: ...
    def get_name(self) -> str | dict | None: ...  # type: ignore[type-arg]
    def into_this(self, value: Any) -> Any: ...
    @staticmethod
    def into_unit(unit: Unit | None, value) -> None: ...  # type: ignore[no-untyped-def]
    @property
    def into_unit_factor(self) -> float | int: ...
    @staticmethod
    def is_angle(arg: Unit | None) -> bool: ...
    @staticmethod
    def is_unitless(arg: Unit | None) -> bool: ...
    @staticmethod
    def mul_units(arg1: Unit | None, arg2: Unit | None) -> Unit | None: ...
    @staticmethod
    def name_to_dict(expr: str | dict) -> dict: ...  # type: ignore[type-arg]
    @staticmethod
    def name_to_str(namedict: dict | None) -> str: ...  # type: ignore[type-arg]
    @staticmethod
    def require_angle(arg: Unit | None, info: str = ...) -> None: ...
    @staticmethod
    def require_compatible(first: Unit | None, second: Unit | None, info: str = ...
        ) -> None: ...
    @staticmethod
    def require_match(first: Unit | None, second: Unit | None, info: str = ...
        ) -> None: ...
    @staticmethod
    def require_unitless(arg: Unit | None, info: str = ...) -> None: ...
    def set_name(self, name: str | dict) -> Unit: ...  # type: ignore[type-arg]
    def sqrt(self) -> Unit: ...
    @staticmethod
    def sqrt_unit(unit: Unit | None) -> Unit | None: ...
    @staticmethod
    def unit_power(unit: Unit | None, power: int | float) -> Unit | None: ...

##########################################################################################
