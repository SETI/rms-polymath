##########################################################################################
# polymath/matrix.py: Matrix subclass ofse PolyMath base class
##########################################################################################

import math
import numpy as np
import warnings

from polymath.qube    import Qube
from polymath.scalar  import Scalar
from polymath.boolean import Boolean
from polymath.vector  import Vector
from polymath.vector3 import Vector3
from polymath.unit    import Unit


class Matrix(Qube):
    """A Qube of arbitrary 2-D matrices.

    This class represents arbitrary 2D matrices in the PolyMath framework and provides
    operations for matrix arithmetic, transposition, and inversion.
    """

    _NRANK = 2          # The number of numerator axes.
    _NUMER = None       # Shape of the numerator.
    _FLOATS_OK = True   # True to allow floating-point numbers.
    _INTS_OK = False    # True to allow integers.
    _BOOLS_OK = False   # True to allow booleans.
    _UNITS_OK = True    # True to allow units; False to disallow them.
    _DERIVS_OK = True   # True to allow derivatives and denominators; False to disallow.

    _DEBUG = False      # Set to True for some debugging tasks
    _DELTA = np.finfo(float).eps * 3     # Cutoff used in unary()

    @staticmethod
    def as_matrix(arg, *, recursive=True):
        """Convert the argument to a Matrix if possible.

        Parameters:
            arg: The object to convert to a Matrix.
            recursive (bool, optional): True to include derivatives in the result.

        Returns:
            Matrix: The argument converted to a Matrix.
        """

        if type(arg) is Matrix:
            return arg if recursive else arg.wod

        if isinstance(arg, Qube):

            # Convert a Vector with drank=1 to a Matrix
            if isinstance(arg, Vector) and arg._drank == 1:
                return arg.join_items([Matrix])

            arg = Matrix(arg._values, arg._mask, example=arg)
            return arg if recursive else arg.wod

        return Matrix(arg)

    def row_vector(self, row, *, recursive=True, classes=(Vector3, Vector)):
        """The selected row of a Matrix as a Vector.

        If the Matrix is M x N, then this will return a Vector of length N. By default, if
        N == 3, it will return a Vector3 object instead.

        Parameters:
            row: Index of the row to return.
            recursive (bool, optional): True to return corresponding vectors of
                derivatives.
            classes (tuple, optional): A list of classes; an instance of the first
                suitable class is returned.

        Returns:
            Vector or Vector3: The selected row as a vector.
        """

        return self.extract_numer(0, row, recursive=recursive, classes=classes)

    def row_vectors(self, *, recursive=True, classes=(Vector3, Vector)):
        """A tuple of Vector objects, one for each row of this Matrix.

        If the Matrix is M x N, then this will return M Vectors of length N. By default,
        if N == 3, it will return Vector3 objects instead.

        Parameters:
            recursive (bool, optional): True to return corresponding vectors of
                derivatives.
            classes (tuple, optional): A list of classes; instances of the first
                suitable class are returned.

        Returns:
            tuple: A tuple of Vector objects, one for each row.
        """

        vectors = []
        for row in range(self._numer[0]):
            vectors.append(self.extract_numer(0, row, recursive=recursive,
                                              classes=classes))

        return tuple(vectors)

    def column_vector(self, column, *, recursive=True, classes=(Vector3, Vector)):
        """The selected column of a Matrix as a Vector.

        If the Matrix is M x N, then this will return a Vector of length M. By default, if
        M == 3, it will return a Vector3 object instead.

        Parameters:
            column: Index of the column to return.
            recursive (bool, optional): True to return corresponding vectors of
                derivatives.
            classes (tuple, optional): A list of classes; an instance of the first
                suitable class is returned.

        Returns:
            Vector or Vector3: The selected column as a vector.
        """

        return self.extract_numer(1, column, recursive=recursive, classes=classes)

    def column_vectors(self, recursive=True, classes=(Vector3, Vector)):
        """A tuple of Vector objects, one for each column of this Matrix.

        If the Matrix is M x N, then this will return N Vectors of length M. By default,
        if M == 3, it will return Vector3 objects instead.

        Parameters:
            recursive (bool, optional): True to return corresponding vectors of
                derivatives.
            classes (tuple, optional): A list of classes; instances of the first suitable
                class are returned.

        Returns:
            tuple: A tuple of Vector objects, one for each column.
        """

        vectors = []
        for col in range(self._numer[1]):
            vectors.append(self.extract_numer(1, col, recursive=recursive,
                                              classes=classes))

        return tuple(vectors)

    def to_vector(self, axis, indx, *, recursive=True, classes=()):
        """One of the components of a Matrix as a Vector.

        Parameters:
            axis: Axis index from which to extract vector.
            indx: Index of the vector along this axis.
            classes (list, optional): A list of the Vector subclasses to return. The first
                valid one will be used.
            recursive (bool, optional): True to extract the derivatives as well.

        Returns:
            Vector: One component of the Matrix as a Vector.
        """

        return self.extract_numer(axis, indx, list(classes) + [Vector],
                                  recursive=recursive)

    def to_scalar(self, /, indx0, indx1, *, recursive=True):
        """One of the elements of a Matrix as a Scalar.

        Parameters:
            indx0 (int): Index along the first matrix axis.
            indx1 (int): Index along the second matrix axis.
            recursive (bool, optional): True to extract the derivatives as well.

        Returns:
            Scalar: One element of the Matrix as a Scalar.
        """

        vector = self.extract_numer(0, indx0, Vector, recursive=recursive)
        return vector.extract_numer(0, indx1, Scalar, recursive=recursive)

    @staticmethod
    def from_scalars(*args, recursive=True, shape=None, classes=()):
        """Construct a Matrix or subclass by combining scalars.

        Parameters:
            *args: Any number of Scalars or arguments that can be casted to Scalars. They
                need not have the same shape, but it must be possible to broadcast them to
                the same shape. A value of None is converted to a zero-valued Scalar that
                matches the denominator shape of the other arguments.
            recursive (bool, optional): True to include all the derivatives. The returned
                object will have derivatives representing the union of all the derivatives
                found amongst the scalars.
            shape (tuple, optional): The Matrix's item shape. If not specified but the
                number of Scalars is a perfect square, a square matrix is returned.
                If specified, the number of scalar arguments must equal
                shape[0] * shape[1]. Each scalar argument can be a single value or an
                array that will be broadcast to match the other arguments.
            classes (list, optional): An arbitrary list defining the preferred class of
                the returned object. The first suitable class in the list will be used.
                Default is [Matrix].

        Returns:
            Matrix: A Matrix constructed from the given scalars.

        Raises:
            TypeError: If the input would result in an int matrix, which is not allowed.
            ValueError: If the number of Scalars does not match the specified shape.
        """

        # Create the Vector object
        vector = Vector.from_scalars(*args, recursive=recursive)

        # Int matrices are disallowed
        if vector.is_int():
            raise TypeError('Matrix.from_scalars() requires objects with data type float')

        # Determine the shape
        if shape is not None:
            if len(shape) != 2:
                raise ValueError(f'invalid Matrix item shape: {shape}')

            size = shape[0] * shape[1]
            if len(args) != size:
                raise ValueError('incorrect number of Scalars for Matrix.from_scalars() '
                                 f'with shape {shape}: expected {size}, got {len(args)}')
            shape = tuple(shape)

        else:
            dim = int(np.sqrt(len(args)))
            size = dim * dim
            if size != len(args):
                raise ValueError('incorrect number of Scalars for Matrix.from_scalars() '
                                 'with square shape')
            shape = (dim, dim)

        return vector.reshape_numer(shape, list(classes) + [Matrix], recursive=recursive)

    def is_diagonal(self, *, delta=0.):
        """A Boolean equal to True where the matrix is diagonal.

        Masked matrices return True. For arrays of matrices, returns a Boolean array
        with the same shape as the array, where each element indicates whether the
        corresponding matrix is diagonal.

        Parameters:
            delta (float, optional): The fractional limit on what can be treated as
                equivalent to zero in the off-diagonal terms. It is scaled by the RMS
                value of all the elements in the matrix.

        Returns:
            Boolean: True where the matrix is diagonal.

        Raises:
            ValueError: If the matrix is not square or has denominators.
        """

        size = self.item[0]
        if size != self.item[1]:
            raise ValueError(f'{type(self).__name__}.is_diagonal() requires a square '
                             f'matrix; shape is {self._numer}')

        if self._drank:
            raise ValueError(f'{type(self).__name__}.is_diagonal() does not support '
                             'denominators')

        # If necessary, calculate the matrix RMS
        if delta != 0.:
            # rms, scaled to be unity for an identity matrix
            rms = (np.sqrt(np.sum(np.sum(self._values**2, axis=-1), axis=-1)) / size)

        # Flatten the value array
        values = self._values.reshape(self._shape + (size * size,))

        # Slice away the last element
        sliced = values[..., :-1]

        # Reshape so that only elemenents in the first column can be nonzero
        reshaped = sliced.reshape(self._shape + (size-1, size + 1))

        # Slice away the first column
        sliced = reshaped[..., 1:]

        # Convert back to 1-D items
        reshaped = sliced.reshape(self._shape + ((size - 1) * size,))

        # Compare
        if delta == 0:
            compare = (reshaped == 0.)
        else:
            compare = (np.abs(reshaped) <= (delta * rms)[..., np.newaxis])

        compare = np.all(compare, axis=-1)

        # Apply mask
        if np.shape(compare) == ():
            if self._mask:
                compare = True
        elif np.shape(self._mask) == ():
            if self._mask:
                compare.fill(True)
        else:
            compare[self._mask] = True

        return Boolean(compare)

    def transpose(self, *, recursive=True):
        """The transpose of this matrix.

        Parameters:
            recursive (bool, optional): True to include the transposed derivatives; False
                to return an object without derivatives.

        Returns:
            Matrix: Transpose of this matrix.
        """

        return self.transpose_numer(0, 1, recursive=recursive)

    @property
    def T(self):  # noqa: N802  # mirrors the NumPy .T attribute
        """The transpose of this matrix.

        Returns:
            Matrix: Transpose of this matrix with derivatives included.
        """

        return self.transpose_numer(0, 1, recursive=True)

    def inverse(self, *, recursive=True, nozeros=False):
        """The inverse of this matrix.

        The returned object will have the same subclass as this object. Matrices with
        determinant equal to zero are masked.

        Parameters:
            recursive (bool, optional): True to include the derivatives of the inverse.
            nozeros (bool, optional): False to mask out any matrices with zero-valued
                determinants. Set to True only if you know in advance that all
                determinants are nonzero.

        Returns:
            Matrix: Inverse of this matrix. It will have the same subclass as this object.
                Matrices with a determinant equal to zero will be masked.

        Raises:
            ValueError: If the matrix is not square or has denominators.
            ValueError: If `nozeros` is True but a determinant of zero is encountered.
        """

        # Validate array
        if self._numer[0] != self._numer[1]:
            raise ValueError(f'{type(self).__name__}.inverse() requires a square matrix; '
                             f'shape is {self._numer}')

        if self._drank:
            raise ValueError(f'{type(self).__name__}.inverse() does not support '
                             'denominators')

        # Check determinant if necessary
        new_mask = self._mask
        old_values = self._values
        if not nozeros:
            det = np.linalg.det(old_values)

            # Mask out un-invertible matrices and replace with identify matrices.
            # The substitution goes into a copy; this object must not be modified.
            mask = (det == 0.)
            if np.any(mask):
                old_values = old_values.copy()
                old_values[mask] = np.diag(np.ones(self._numer[0]))
                new_mask = Qube.or_(self._mask, mask)

        # Invert the array
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                new_values = np.linalg.inv(old_values)
            except (RuntimeWarning, np.linalg.LinAlgError) as err:
                raise ValueError(f'{type(self).__name__}.inverse() input is singular'
                                 ) from err

        # Construct the result
        obj = Matrix(new_values, new_mask, unit=Unit.unit_power(self._unit, -1))

        # Fill in derivatives
        if recursive and self._derivs:
            new_derivs = {}

            # -M^-1 * dM/dt * M^-1
            for key, deriv in self._derivs.items():
                new_derivs[key] = -obj * deriv * obj

            obj.insert_derivs(new_derivs)

        return obj

    def unitary(self):
        """The nearest unitary matrix as a Matrix3.

        This method only works for 3x3 matrices. For other matrix sizes, a ValueError
        is raised.

        Uses the algorithm from
        https://wikipedia.org/wiki/Orthogonal_matrix#Nearest_orthogonal_matrix

        Returns:
            Matrix3: The nearest unitary (orthogonal) matrix.

        Raises:
            ValueError: If the matrix has denominators or is not 3x3.
        """

        # Algorithm from
        #    https://wikipedia.org/wiki/Orthogonal_matrix#Nearest_orthogonal_matrix
        max_iters = 10      # Adequate iterations unless convergence is failing

        m0 = self.wod
        if m0._drank:
            raise ValueError(f'{type(self).__name__}.unitary() does not support '
                             'denominators')

        if m0._numer != (3, 3):
            raise ValueError(f'{type(self).__name__}.unitary() requires 3x3 matrix as '
                             'input')

        # Iterate...
        m0 = Matrix(m0)     # can't do certain math operations on Matrix3 subclass
        next_m = m0
        for i in range(max_iters):
            m = next_m
            next_m = 2. * m0 * (m.inverse() * m0 + m0.T * m).inverse()
            rms = Qube.rms(next_m * next_m.T - Matrix.IDENTITY3)

            if Matrix._DEBUG:
                sorted_ = np.sort(rms._values.ravel())
                print(i, sorted_[-4:])

            if rms.max() <= Matrix._DELTA:
                break

        new_mask = (rms._values > Matrix._DELTA)
        if not np.any(new_mask):
            new_mask = self._mask
        elif self._mask is not False:
            new_mask |= self._mask

        return Qube._MATRIX3_CLASS(next_m._values, new_mask)

    def solve(self, arg, *, recursive=True, nozeros=False):
        """The Vector X that satisfies A X = B, for this square matrix A.

        Parameters:
            arg (Vector, array-like): The Vector B of right-hand sides. Its item shape
                must match the size of this matrix.
            recursive (bool, optional): True to include the derivatives of the solution,
                which are derived from those of this matrix and of `arg`.
            nozeros (bool, optional): False to mask out any matrices with a zero-valued
                determinant. Set to True only if you know in advance that every
                determinant is nonzero.

        Returns:
            Vector: The solution X, with the leading shape obtained by broadcasting this
            matrix against `arg`. Elements where this matrix is singular are masked. The
            returned object takes the subclass of `arg` where that subclass fits.

        Raises:
            ValueError: If this matrix is not square.
            ValueError: If this matrix or `arg` has a denominator.
            ValueError: If the item shape of `arg` does not match the size of this matrix.
            ValueError: If `nozeros` is True but this matrix is singular.

        Examples:
            >>> a = Matrix([[2., 0.], [0., 4.]])
            >>> a.solve(Vector([2., 4.]))
            Vector(1.0 1.0)
        """

        size = self._numer[0]
        if self._numer[1] != size:
            raise ValueError(f'{type(self).__name__}.solve() requires a square matrix; '
                             f'shape is {self._numer}')

        if self._drank:
            raise ValueError(f'{type(self).__name__}.solve() does not support '
                             'denominators')

        b = Vector.as_vector(arg, recursive=recursive)

        if b._drank:
            raise ValueError(f'{type(self).__name__}.solve() right operand does not '
                             f'support denominators: {b._denom}')

        if b._numer != (size,):
            raise ValueError(f'{type(self).__name__}.solve() operand item shapes are '
                             f'incompatible: {self._numer}, {b._numer}')

        # Broadcast to a common leading shape. The broadcast values are only ever read,
        # so the operands themselves need not become read-only.
        (a, b) = Qube.broadcast(self, b, recursive=recursive, _protected=False)
        new_shape = a._shape

        # Mask out the singular matrices, substituting the identity into a copy so that
        # this object is left alone
        a_vals = a._values
        new_mask = Qube.or_(a._mask, b._mask)
        if not nozeros:
            singular = (np.linalg.det(a_vals) == 0.)
            if np.any(singular):
                a_vals = a_vals.copy()
                a_vals[singular] = np.diag(np.ones(size))
                new_mask = Qube.or_(new_mask, singular)

        def solve_values(values, denom):
            """Solve for one right-hand side, with any denominator axes flattened into
            additional columns.
            """

            columns = values.reshape(new_shape + (size, math.prod(denom)))

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    solution = np.linalg.solve(a_vals, columns)
                except (RuntimeWarning, np.linalg.LinAlgError) as err:
                    raise ValueError(f'{type(self).__name__}.solve() matrix is singular'
                                     ) from err

            return solution.reshape(values.shape)

        obj = Vector(solve_values(b._values, ()), new_mask,
                     unit=Unit.div_units(b._unit, a._unit))

        # Differentiating A X = B gives A dX/dt = dB/dt - (dA/dt) X, so each derivative
        # is the solution of the same system with a new right-hand side
        if recursive and (a._derivs or b._derivs):
            x = obj.wod
            new_derivs = {}
            for key in set(a._derivs) | set(b._derivs):
                if key in a._derivs:
                    term = a._derivs[key] * x
                    rhs = (b._derivs[key] - term) if key in b._derivs else -term
                else:
                    rhs = b._derivs[key]

                new_derivs[key] = Vector(solve_values(rhs._values, rhs._denom),
                                         Qube.or_(new_mask, rhs._mask),
                                         unit=Unit.div_units(rhs._unit, a._unit),
                                         drank=rhs._drank)

            obj.insert_derivs(new_derivs)

        return obj.cast(type(b))

    ######################################################################################
    # Overrides of superclass operators
    ######################################################################################

    def __abs__(self):
        """Raise a TypeError; absolute value is not defined for matrices.

        This is an override of :meth:`Qube.__abs__`.
        """

        Qube._raise_unsupported_op('abs()', self)

    def __floordiv__(self, /, arg):
        """Raise a TypeError; floor division is not defined for matrices.

        This is an override of :meth:`Qube.__floordiv__`.
        """

        Qube._raise_unsupported_op('//', self, arg)

    def __rfloordiv__(self, /, arg):
        """Raise a TypeError; floor division is not defined for matrices.

        This is an override of :meth:`Qube.__rfloordiv__`.
        """

        Qube._raise_unsupported_op('//', arg, self)

    def __ifloordiv__(self, /, arg):
        """Raise a TypeError; floor division is not defined for matrices.

        This is an override of :meth:`Qube.__ifloordiv__`.
        """

        Qube._raise_unsupported_op('//=', self, arg)

    def __mod__(self, /, arg):
        """Raise a TypeError; modulo is not defined for matrices.

        This is an override of :meth:`Qube.__mod__`.
        """

        Qube._raise_unsupported_op('%', self, arg)

    def __rmod__(self, /, arg):
        """Raise a TypeError; modulo is not defined for matrices.

        This is an override of :meth:`Qube.__rmod__`.
        """

        Qube._raise_unsupported_op('%', arg, self)

    def __imod__(self, /, arg):
        """Raise a TypeError; modulo is not defined for matrices.

        This is an override of :meth:`Qube.__imod__`.
        """

        Qube._raise_unsupported_op('%=', self, arg)

    def identity(self):
        """An identity matrix of the same size and subclass as this.

        This method overrides :meth:`Qube.identity`.

        Raises:
            ValueError: If the matrix is not square.
        """

        size = self._numer[0]

        if self._numer[1] != size:
            raise ValueError(f'{type(self).__name__}.identity() requires a square '
                             f'matrix; shape is {self._numer}')

        values = np.zeros((size, size))
        for i in range(size):
            values[i, i] = 1.

        obj = Qube.__new__(type(self))
        obj.__init__(values)

        return obj.as_readonly()

    ######################################################################################
    # Overrides of arithmetic operators
    ######################################################################################

    def reciprocal(self, *, recursive=True, nozeros=False):
        """Return an object equivalent to the reciprocal of this object.

        For a Matrix, the reciprocal is the inverse. This overrides
        :meth:`Qube.reciprocal`.

        Parameters:
            recursive (bool, optional): True to return the derivatives of the reciprocal
                too; otherwise, derivatives are removed.
            nozeros (bool, optional): False to mask out any matrices with zero-valued
                determinants. Set to True only if you know in advance that all
                determinants are nonzero.

        Returns:
            Matrix: The matrix inverse.

        Raises:
            ValueError: If the matrix is not square, has denominators, or has a
                determinant of zero.
        """

        return self.inverse(recursive=recursive, nozeros=nozeros)

##########################################################################################
# Useful class constants
##########################################################################################

Matrix.IDENTITY2 = Matrix([[1, 0], [0, 1]]).as_readonly()
Matrix.IDENTITY3 = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).as_readonly()

Matrix.MASKED2 = Matrix([[1, 1], [1, 1]], True).as_readonly()
Matrix.MASKED3 = Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]], True).as_readonly()

Matrix.ZERO33 = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).as_readonly()
Matrix.UNIT33 = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).as_readonly()

Matrix.ZERO3_ROW = Matrix([[0, 0, 0]]).as_readonly()
Matrix.XAXIS_ROW = Matrix([[1, 0, 0]]).as_readonly()
Matrix.YAXIS_ROW = Matrix([[0, 1, 0]]).as_readonly()
Matrix.ZAXIS_ROW = Matrix([[0, 0, 1]]).as_readonly()

Matrix.ZERO3_COL = Matrix([[0], [0], [0]]).as_readonly()
Matrix.XAXIS_COL = Matrix([[1], [0], [0]]).as_readonly()
Matrix.YAXIS_COL = Matrix([[0], [1], [0]]).as_readonly()
Matrix.ZAXIS_COL = Matrix([[0], [0], [1]]).as_readonly()

##########################################################################################
# Once defined, register with base class
##########################################################################################

Qube._MATRIX_CLASS = Matrix

##########################################################################################
