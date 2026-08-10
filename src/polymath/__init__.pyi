##########################################################################################
# polymath/__init__.pyi
##########################################################################################
"""Type stub for the PolyMath package namespace.

The `src` tree carries no inline annotations, so type information for public
symbols is published through stub files instead. Each class is described
by the stub alongside its own module.

Each import uses the redundant `X as X` form, which is how a stub marks a name
as re-exported rather than merely imported for internal use.
"""

from polymath.boolean import Boolean as Boolean
from polymath.matrix import Matrix as Matrix
from polymath.matrix3 import Matrix3 as Matrix3
from polymath.pair import Pair as Pair
from polymath.polynomial import Polynomial as Polynomial
from polymath.quaternion import Quaternion as Quaternion
from polymath.qube import Qube as Qube
from polymath.scalar import Scalar as Scalar
from polymath.unit import Unit as Unit
from polymath.vector import Vector as Vector
from polymath.vector3 import Vector3 as Vector3

__version__: str

__all__ = ['Boolean', 'Matrix', 'Matrix3', 'Pair', 'Polynomial', 'Quaternion', 'Qube',
           'Scalar', 'Unit', 'Vector', 'Vector3']

##########################################################################################
