################################################################################
# polymath/quaternion.py: Quaternion subclass of PolyMath base class
################################################################################

from __future__ import division
import numpy as np

from polymath.qube    import Qube
from polymath.scalar  import Scalar
from polymath.vector  import Vector
from polymath.vector3 import Vector3
from polymath.matrix  import Matrix
from polymath.matrix3 import Matrix3
from polymath.units   import Units

class Quaternion(Vector):
    """A PolyMath subclass containing quaternions and supporting conversions
    between quaternion, rotation matrices and sets of Euler angles."""

    NRANK = 1           # the number of numerator axes.
    NUMER = (4,)        # shape of the numerator.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = False     # True to allow integers.
    BOOLS_OK = False    # True to allow booleans.

    UNITS_OK = False    # True to allow units; False to disallow them.
    DERIVS_OK = True    # True to allow derivatives and to allow this class to
                        # have a denominator; False to disallow them.

    DEFAULT_VALUE = np.array([1.,0.,0.,0.])

    #===========================================================================
    @staticmethod
    def as_quaternion(arg, recursive=True):
        """This object converted to a Quaternion."""

        if isinstance(arg, Quaternion):
            if recursive:
                return arg
            return arg.wod

        if isinstance(arg, Matrix3):
            return Quaternion.from_matrix3(arg, recursive=recursive)

        if isinstance(arg, Qube):

            if arg._numer_ == (3,):
                return Quaternion.from_parts(0., arg, recursive=recursive)

            arg = Quaternion(arg, arg._mask_, example=arg)
            if recursive:
                return arg
            return arg.wod

        return Quaternion(arg)

    #===========================================================================
    @staticmethod
    def from_parts(scalar, vector, recursive=True):
        """Construct a Quaternion from separate scalar and vector components.

        If either argument is None, the associated components are filled with
        zeros.
        """

        # Fill in missing values
        if scalar is None:
            scalar = Scalar(np.zeros(vector._denom_), drank=vector._drank_)

        if vector is None:
            vector = Vector3(np.zeros((3,) + scalar._denom_),
                             drank=scalar._drank_)

        # Broadcast shapes
        scalar = Scalar.as_scalar(scalar)
        vector = Vector3.as_vector3(vector)

        (scalar, vector) = Qube.broadcast(scalar, vector)

        # Validate denominators
        if scalar._denom_ != vector._denom_:
            raise ValueError('Quaternion.from_parts() denominators are '
                             'incompatible: %s, %s'
                             % (scalar._denom_, vector._denom_))

        # Align axes
        drank = scalar._drank_
        before = len(scalar._shape_) * (slice(None),)
        after  = drank * (slice(None),)

        s_slice = before + (0,) + after
        v_slice = before + (slice(1,4),) + after

        values = np.empty(scalar._shape_ + (4,) + scalar._denom_)
        values[s_slice] = scalar._values_
        values[v_slice] = vector._values_

        # Construct object
        obj = Quaternion(values, Qube.or_(scalar._mask_, vector._mask_),
                         drank=drank)

        # Fill in derivatives if necessary
        if recursive:
            new_derivs = {}

            for (key, deriv) in scalar._derivs_.items():
                new_derivs[key] = Quaternion.from_parts(deriv, None, False)

            for (key, deriv) in vector._derivs_.items():
                term = Quaternion.from_parts(None, deriv, False)
                if key in new_derivs:
                    new_derivs[key] = new_derivs[key] + term
                else:
                    new_derivs[key] = term

            obj.insert_derivs(new_derivs)

        return obj

    #===========================================================================
    def to_parts(self, recursive=True):
        """The Scalar and Vector components of a Quaternion."""

        return (self.extract_numer(0, 0, Scalar, recursive=recursive),
                self.slice_numer(0, 1, 4, Vector3, recursive=recursive))

    #===========================================================================
    @staticmethod
    def from_rotation(angle, vector, recursive=True):
        """Construct a Quaternion for an angular rotation about an axis vector.
        """

        angle = Scalar.as_scalar(angle)
        vector = Vector3.as_vector3(vector)

        if not recursive:
            angle = angle.wod
            vector = vector.wod

        (angle, vector) = Qube.broadcast(angle, vector)

        half_angle = 0.5 * angle
        scalar = half_angle.cos()
        vector = (half_angle.sin() / vector.norm()) * vector
        return Quaternion.from_parts(scalar, vector)

    #===========================================================================
    def to_rotation(self, recursive=True):
        """The rotation angle and unit vector defined by a Quaternion.
        """

        (cos_half_angle, vector) = self.to_parts(recursive=recursive)
        sin_half_angle = vector.norm(recursive=recursive)
        angle = 2. * sin_half_angle.arctan2(cos_half_angle, recursive=recursive)

        return (angle, vector/sin_half_angle)

    #===========================================================================
    def conj(self, recursive=True):
        """Complex conjugate of this quaternion."""

        new_values = self._values_.copy()

        if self._drank_ > 0:
            lshape = len(new_values.shape)
            new_values = np.rollaxis(new_values, -self._drank_-1, lshape)

        new_values[..., 1:4] *= -1

        if self._drank_ > 0:
            new_values = np.rollaxis(new_values, -1, -self._drank_-1)

        obj = Quaternion(new_values, self._mask_, example=self)

        if recursive and self._derivs_:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv.conj(recursive=False))

        return obj

    #===========================================================================
    def to_matrix3(self, recursive=True, partials=False):
        """Convert this Quaternion to a Matrix3.

        Input:
            recursive   if True, the returned Matrix3 contains representations
                        of the derivatives of the Quaternion. These are
                        represented as Matrix objects, not Matrix3 objects,
                        because they are not unitary.

            partials    if True, instead of returning the Quaternion, return a
                        tuple containing the Quaternion and its (3x3x4) partial
                        derivatives with respect to the components of the
                        quaternion.
        """

        if self._drank_:
            raise ValueError('Quaternion.to_matrix3() does not support '
                             'denominators')

        # From http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        pvals = self._values_
        pmask = self._mask_
        pnorm = np.sqrt(np.sum(pvals**2, axis=-1))

        zero_mask = (pnorm == 0.)
        if np.any(zero_mask):
            if np.shape(pvals) == ():
                pnorm = 1.
                pmask = True
            else:
                pnorm[zero_mask] = 1.
                pmask = pmask | zero_mask

        # Scale by sqrt(2) to eliminate need to keep multiplying by 2
        qvals = np.sqrt(2) / pnorm[...,np.newaxis] * pvals
        s = qvals[...,0]
        x = qvals[...,1]
        y = qvals[...,2]
        z = qvals[...,3]

        sx = s * x
        sy = s * y
        sz = s * z
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z

        values = np.empty(self._shape_ + (3,3))
        values[...,0,0] = 1. - (yy + zz)
        values[...,0,1] =      (xy - sz)
        values[...,0,2] =      (xz + sy)
        values[...,1,0] =      (xy + sz)
        values[...,1,1] = 1. - (xx + zz)
        values[...,1,2] =      (yz - sx)
        values[...,2,0] =      (xz - sy)
        values[...,2,1] =      (yz + sx)
        values[...,2,2] = 1. - (xx + yy)

        obj = Matrix3(values, pmask)

        if (recursive and self._derivs_) or partials:
#           Before scaling, but assuming a unit quaternion...
#             values[...,0,0] = 1. - 2.*(yy + zz)
#             values[...,0,1] =      2.*(xy - sz)
#             values[...,0,2] =      2.*(xz + sy)
#             values[...,1,0] =      2.*(xy + sz)
#             values[...,1,1] = 1. - 2.*(xx + zz)
#             result[...,1,2] =      2.*(yz - sx)
#             values[...,2,0] =      2.*(xz - sy)
#             values[...,2,1] =      2.*(yz + sx)
#             values[...,2,2] = 1. - 2.*(xx + yy)

#             dm_dq = np.zeros(self._shape_ + (3,3,4))
#             dm_dq[...,0,0,:] = 2*( 0,  0,-2y,-2z)
#             dm_dq[...,0,1,:] = 2*(-z,  y,  x, -s)
#             dm_dq[...,0,2,:] = 2*( y,  z,  s,  x)
#             dm_dq[...,1,0,:] = 2*( z,  y,  x,  s)
#             dm_dq[...,1,1,:] = 2*( 0,-2x,  0,-2z)
#             dm_dq[...,1,2,:] = 2*(-x, -s,  z,  y)
#             dm_dq[...,2,0,:] = 2*(-y,  z, -s,  x)
#             dm_dq[...,2,1,:] = 2*( x,  s,  z,  y)
#             dm_dq[...,2,2,:] = 2*( 0,-2x,-2y,  0)

#           (s,x,y,z) have already been scaled by sqrt(2). Scale by another
#           factor of sqrt(2) when done.

            m = np.zeros(self._shape_ + (3,3,4))
            m[...,1,1,1] = m[...,2,2,1] = -2 * x
            m[...,0,0,2] = m[...,2,2,2] = -2 * y
            m[...,0,0,3] = m[...,1,1,3] = -2 * z

            m[...,0,1,3] = m[...,1,2,1] = m[...,2,0,2] = -s
            m[...,0,2,2] = m[...,1,0,3] = m[...,2,1,1] =  s

            m[...,1,2,0] = -x
            m[...,0,1,2] = m[...,0,2,3] = m[...,1,0,2] = m[...,2,0,3] = \
                                                         m[...,2,1,0] = x

            m[...,2,0,0] = -y
            m[...,0,1,1] = m[...,0,2,0] = m[...,1,0,1] = m[...,1,2,3] = \
                                                         m[...,2,1,3] = y

            m[...,0,1,0] = -z
            m[...,0,2,1] = m[...,1,0,0] = m[...,1,2,2] = m[...,2,0,1] = \
                                                         m[...,2,1,2] = z

            dm_dq = Matrix(m, pmask, drank=1)

#           We also have to deal with the unit() applied to the quaternion at
#           the begininning. Let p be the un-normalized quaternion, q the unit
#           version.
#               q = p / p_norm
#           where
#               qnorm = sqrt(q0**2 + q1**2 + a2**2 + q3**2)
#
#           dq0/dp0 = (p1**2 + p2**2 + p3**2) / pnorm**3
#           dq0/dp0 = -p0*p1 / pnorm**3

            dq_dp = np.zeros(self._shape_ + (4,4))
            for i in range(4):
                temp = pvals.copy()
                temp[...,i] = 0.
                dq_dp[...,i,i] = -np.sum(temp**2, axis=-1)
                for j in range(i+1,4):
                    dq_dp[...,i,j] = dq_dp[...,j,i] = pvals[...,i]*pvals[...,j]

            dm_dp = dm_dq.chain(Quaternion(dq_dp, drank=1)) * (-np.sqrt(2) /
                                                               pnorm**3)

            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, dm_dp.chain(deriv))

        if partials:
            return (obj, dm_dp)

        return obj

    #===========================================================================
    @staticmethod
    def from_matrix3_experimental(matrix, recursive=True):
        """Convert a Matrix3 to a Quaternion.

        Input:
            recursive   if True, the returned Matrix3 contains representations
                        of the derivatives of the Quaternion. These are
                        represented as Matrix objects, not Matrix3 objects,
                        because they are not unitary.
        """

        #### This appears to work, but is notably less accurate than the other
        #### method. Nevertheless, it might be worth exploring further, in part
        #### because it might provide a pathway to supporting derivatives.

        # From https://www.euclideanspace.com/maths/geometry/rotations/-
        # conversions/matrixToQuaternion/
        #
        # Because modern CPUs execute sqrt and __div__ in the same amount of
        # time, the use of four square roots is no big deal, and this avoids all
        # the masks and loops.

        qvals = np.empty(matrix.shape + (4,))

        m00 = matrix.vals[...,0,0]
        m11 = matrix.vals[...,1,1]
        m22 = matrix.vals[...,2,2]

        sign21 = np.sign(matrix.vals[...,2,1] - matrix.vals[...,1,2])
        sign02 = np.sign(matrix.vals[...,0,2] - matrix.vals[...,2,0])
        sign10 = np.sign(matrix.vals[...,1,0] - matrix.vals[...,0,1])

        qvals[...,0] =          np.sqrt(np.maximum(0., 1 + m00 + m11 + m22))
        qvals[...,1] = sign21 * np.sqrt(np.maximum(0., 1 + m00 - m11 - m22))
        qvals[...,2] = sign02 * np.sqrt(np.maximum(0., 1 - m00 + m11 - m22))
        qvals[...,3] = sign10 * np.sqrt(np.maximum(0., 1 - m00 - m11 + m22))

        qvals *= 0.5
        q = Quaternion(qvals, matrix._mask_)

        if recursive and matrix._derivs_:

            # TODO: what to do about divide by zero here?
            # If one of sign21, sign02, and sign10, this will make the
            # associated derivative zero as well; is that correct?
            (q0, q1, q2, q3) = q.to_scalars()
            f0 =  0.5           / q0
            f1 = (0.5 * sign21) / q1
            f2 = (0.5 * sign02) / q2
            f3 = (0.5 * sign10) / q3

            div_by_zero = (q0 == 0.) | (q1 == 0.) | (q2 == 0.) | (q3 == 0.)
            if any(div_by_zero):
                new_mask = Qube.or_(matrix._mask_, div_by_zero)
            else:
                new_mask = matrix._mask_

            for key, deriv in matrix._derivs_.items():
                dm00 = deriv.to_scalar(0, 0, recursive=False)
                dm11 = deriv.to_scalar(1, 1, recursive=False)
                dm22 = deriv.to_scalar(2, 2, recursive=False)

                # Empty buffer with numerator axis first
                new_vals = np.empty((4,) + matrix.shape + matrix.denom)
                new_vals[0] = (f0 * ( dm00 + dm11 + dm22))._values_
                new_vals[1] = (f1 * ( dm00 - dm11 - dm22))._values_
                new_vals[2] = (f2 * (-dm00 + dm11 - dm22))._values_
                new_vals[3] = (f3 * (-dm00 - dm11 + dm22))._values_

                new_mask = Qube.or_(new_mask, deriv._mask_)
                new_deriv = Quaternion(new_vals, new_mask, drank=deriv._drank_)
                q.insert_deriv(key, new_deriv)

        return q

    #===========================================================================
    @staticmethod
    def from_matrix3(matrix, recursive=True):
        """Convert a Matrix3 to a Quaternion.

        Input:
            recursive   if True, the returned Matrix3 contains representations
                        of the derivatives of the Quaternion. These are
                        represented as Matrix objects, not Matrix3 objects,
                        because they are not unitary. THIS FEATURE IS NOT
                        CURRENTLY IMPLEMENTED.
        """

        if recursive and matrix._derivs_:
            raise NotImplementedError('Quaternion.from_matrix3() '      # TODO
                                      'does not implement derivatives')

        # From http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        #
        # Suppose Qxx is the largest diagonal entry in the matrix
        # t = Qxx + Qyy + Qzz
        # r = sqrt(1 + Qxx - Qyy - Qzz)
        # s = 0.5 / r
        # w = (Qzy - Qyz)*s
        # x = 0.5*r
        # y = (Qxy + Qyx)*s
        # z = (Qzx + Qxz)*s
        #
        # Handle the same when Qyy and Qzz are the largest
        #
        # Minor rewrite...
        # trace = Qxx + Qyy + Qzz
        # r_sq = 1 + Qxx - Qyy - Qzz = 1 + 2*Qxx - trace
        # r = sqrt(r_sq)
        # s = 0.5 / r
        # w_over_s = Qzy - Qyz
        # x_over_s = 0.5*r / s = 0.5*r / (0.5/r) = r_sq
        # y_over_s = Qxy + Qyx
        # z_over_s = Qzx + Qxz

        matrix = Matrix3.as_matrix3(matrix)
        Q = matrix._values_[np.newaxis]       # add front axis so indexing works

        # Select the diagonals
        diags = Q.reshape(Q.shape[:-2] + (9,))
        diags = diags[...,::4]              # tricky way to select diagonals

        # Calculate the trace
        trace = np.sum(diags, axis=-1)

        # Designate i as the index of the largest entry on the diagonal
        # j and k follow in sequence
        argmax = np.argmax(diags, axis=-1)
        max_diags = np.max(diags, axis=-1)

        r_sq = 1 + 2*max_diags - trace      # valid regardless of which is max

        r = np.sqrt(r_sq)

        zero_mask = (r == 0.)
        if np.any(zero_mask):
            if np.shape(zero_mask) == ():
                s = 0.
            else:
                r_nozeros = r.copy()
                r_nozeros[zero_mask] = 1.
                s = 0.5 / r_nozeros
        else:
            r_nozeros = r
            s = 0.5 / r

        quat_over_s = np.empty(Q.shape[:-2] + (4,))
        for i in range(3):
            mask = (argmax == i)

            j = (i+1) % 3
            k = (i+2) % 3

            quat_over_s[mask,0]   = Q[mask,k,j] - Q[mask,j,k]
            quat_over_s[mask,i+1] = r_sq[mask]
            quat_over_s[mask,j+1] = Q[mask,i,j] + Q[mask,j,i]
            quat_over_s[mask,k+1] = Q[mask,i,k] + Q[mask,k,i]

        obj = Quaternion((quat_over_s * s[...,np.newaxis])[0], matrix._mask_)

        # The following code does not work, perhaps because of the vague meaning
        # of partial derivatives when the components of a Matrix3 are so closely
        # coupled. When derivatives are requested, a NotImplementedError is
        # raised instead.

        if recursive and matrix._derivs_:

            # Take derivatives using the symmetric (but possibly unstable)
            # algorithm
            # t = Qxx + Qyy + Qzz
            # r = sqrt(1+t)
            # s = 0.5 / r
            # w = 0.5 * r
            # x = (Qzy - Qyz) * s
            # y = (Qxz - Qzx) * s
            # z = (Qyx - Qxy) * s
            #
            # Minor rewrite...
            # t = Qxx + Qyy + Qzz
            # r = sqrt(1+t)
            # s = 0.5 / r
            # w = 0.5 * r
            # x_over_s = (Qzy - Qyz)
            # y_over_s = (Qxz - Qzx)
            # z_over_s = (Qyx - Qxy)
            #
            # dt/dQ = [I]
            # dr/dQ = 0.5/r * dt/dQ = s * [I]
            # ds/dQ = -0.5 / r**2 * dr/dQ = -2s**2 * s * [I] = -2*s**3 * [I]
            #
            # dw/dQ = 0.5 * dr/dQ = s/2 * [I]
            #
            # d(x_over_s)/dQ = [Mzy] == [[ 0, 0, 0],[ 0, 0,-1],[ 0, 1, 0]]
            # d(y_over_s)/dQ = [Mxz] == [[ 0, 0, 1],[ 0, 0, 0],[-1, 0, 0]]
            # d(z_over_s)/dQ = [Myx] == [[ 0,-1, 0],[ 1, 0, 0],[ 0, 0, 0]]
            #
            # dx/dQ = d(x_over_s * s)/dQ = s * [Mzy] + x_over_s * ds/dQ
            #       = s * [Mzy] - 2*s**3 * (Qzy - Qyz) [I]
            # dy/dQ = s * [Mxz] - 2*s**3 * (Qxz - Qzx) [I]
            # dz/dQ = s * [Myx] - 2*s**3 * (Qyx - Qxy) [I]

            neg2_s3 = -2 * s * s * s

            new_values = np.zeros(matrix.shape + (4,3,3))
            new_values[...,0,0,0] = 0.5 * s
            for i in range(3):
                j = (i+1) % 3
                k = (i+2) % 3
                new_values[...,i+1,k,j] =  s
                new_values[...,i+1,j,k] = -s
                new_values[...,i+1,0,0] = neg2_s3 * (Q[...,k,j] - Q[...,j,k])

            new_values[...,2,2] = new_values[...,1,1] = new_values[...,0,0]

            dq_dQ = Quaternion(new_values, matrix._mask_, drank=2)

            for (key, deriv) in matrix._derivs_.items():
                obj.insert_deriv(key, dq_dQ.chain(deriv))

        return obj

    ############################################################################
    # Overrides of arithmetic operators
    ############################################################################

    def __mul__(self, arg, recursive=True):

        # Use default operator for anything but a Qube subclass
        if not isinstance(arg, Qube):
            return Qube.__mul__(self, arg, recursive=recursive)

        # Convert any 3-vector to a Quaternion
        if arg._numer_ == (3,):
            arg = Quaternion.from_parts(0., arg, recursive=recursive)

        # Send any other object to the default operator
        if type(arg) != Quaternion:
            return Qube.__mul__(self, arg, recursive=recursive)

        # Check denominators
        if self._drank_ and arg._drank_:
            Qube._raise_dual_denoms('*', self, arg)

        # Align axes
        a = self
        b = arg
        a_values = a._values_
        b_values = b._values_

        if a._drank_:
            a_values = np.rollaxis(a_values, -a._drank_-1, len(a_values.shape))
            b_values = b_values.reshape(b._shape_ + a._drank_ * (1,) + (4,))

        if b._drank_:
            a_values = a_values.reshape(a._shape_ + b._drank_ * (1,) + (4,))
            b_values = np.rollaxis(b_values, -b._drank_-1, len(b_values.shape))

        new_values = Quaternion.mul_values(a_values, b_values)

        if a._drank_ or b._drank_:
            new_values = np.rollaxis(new_values, -1,
                                                 -(a._drank_ + b._drank_ + 1))

        # Construct object
        obj = Qube.__new__(type(self))
        obj.__init__(new_values, Qube.or_(a._mask_, b._mask_),
                     drank = a._drank_ + b._drank_)

        # Construct the derivatives if necessary
        if recursive:
            new_derivs = {}

            if a._derivs_:
                for (key, a_deriv) in a._derivs_.items():
                    new_derivs[key] = a_deriv * b.wod

            if b._derivs_:
                for (key, b_deriv) in b._derivs_.items():
                    if key in new_derivs:
                        new_derivs[key] += a.wod * b_deriv
                    else:
                        new_derivs[key] = a.wod * b_deriv

            obj.insert_derivs(new_derivs)

        return obj

    #===========================================================================
    @staticmethod
    def mul_values(a, b):
        """Internal method to multiply two quaternions."""

        # Construct the new value array
        (a, b) = np.broadcast_arrays(a, b)
        new_values = np.empty(a.shape)

        new_values[...,0] = (  a[...,0] * b[...,0]
                             - a[...,1] * b[...,1]
                             - a[...,2] * b[...,2]
                             - a[...,3] * b[...,3])

        new_values[...,1] = (  a[...,0] * b[...,1]
                             + a[...,1] * b[...,0]
                             + a[...,2] * b[...,3]
                             - a[...,3] * b[...,2])

        new_values[...,2] = (  a[...,0] * b[...,2]
                             - a[...,1] * b[...,3]
                             + a[...,2] * b[...,0]
                             + a[...,3] * b[...,1])

        new_values[...,3] = (  a[...,0] * b[...,3]
                             + a[...,1] * b[...,2]
                             - a[...,2] * b[...,1]
                             + a[...,3] * b[...,0])

        return new_values

    #===========================================================================
    def __rmul__(self, arg, recursive=True):

        # Convert any 3-vector to a Quaternion and try again
        if isinstance(arg, Qube) and arg._numer_ == (3,):
            arg = Quaternion.from_parts(0., arg, recursive=recursive)
            return arg.__mul__(self, recursive=recursive)

        # Send any other object to the default operator
        return Qube.__mul__(self, arg, recursive=recursive)

    #===========================================================================
    def __truediv__(self, arg, recursive=True):

        # Use default operator for anything but a Qube subclass
        if not isinstance(arg, Qube):
            return Qube.__truediv__(self, arg, recursive=recursive)

        # Convert any 3-vector to a Quaternion
        if arg._numer_ == (3,):
            arg = Quaternion.from_parts(0., arg, recursive=recursive)

        # Send any other subclass to the default operator
        if type(arg) != Quaternion:
            return Qube.__truediv__(self, arg, recursive=recursive)

        # Multiply by the reciprocal
        return self.__mul__(arg.reciprocal(recursive=recursive),
                            recursive=recursive)

    #===========================================================================
    def reciprocal(self, recursive=True):
        """A object equivalent to the reciprocal of this object.

        Input:
            recursive   True to return the derivatives of the reciprocal too;
                        otherwise, derivatives are removed.
            nozeros     False (the default) to mask out any zero-valued items in
                        this object prior to the divide. Set to True only if you
                        know in advance that this object has no zero-valued
                        items.
        """

        return (self.conj(recursive=recursive) /
                self.norm_sq(recursive=recursive))

    #===========================================================================
    def identity(self):
        """Return an identity-valued Quaternion."""

        return Quaternion(np.array([1.,0.,0.,0.])).as_readonly()

    ############################################################################
    # Decomposition into rotations
    #
    # From: http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    #
    # A triple of Euler angles can be applied/interpreted in 24 ways, which can
    # be specified using a 4 character string or encoded 4-tuple:
    #
    #   *Axes 4-string*: e.g. 'sxyz' or 'ryxy'
    #
    #   - first character : rotations are applied to 's'tatic or 'r'otating
    #     frame
    #   - remaining characters : successive rotation axis 'x', 'y', or 'z'
    #
    #   *Axes 4-tuple*: e.g. (0, 0, 0, 0) or (1, 1, 1, 1)
    #
    #   - inner axis: code of axis ('x':0, 'y':1, 'z':2) of rightmost matrix.
    #   - parity : even (0) if inner axis 'x' is followed by 'y', 'y' is
    #     followed by 'z', or 'z' is followed by 'x'. Otherwise odd (1).
    #   - repetition : first and last axis are same (1) or different (0).
    #   - frame : rotations are applied to static (0) or rotating (1) frame.
    ############################################################################

    # axis sequences for Euler angles
    _NEXT_AXIS = [1, 2, 0, 1]

    # map axes strings to/from tuples of inner axis, parity, repetition, frame
    _AXES2TUPLE = {
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

    _TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

    #===========================================================================
    @staticmethod
    def from_euler(ai, aj, ak, axes='rzxz'):
        """Creates a Quaternion from three Scalars of Euler angles plus an axis
        sequence.

        ai, aj, ak : Euler's roll, pitch and yaw angles
        axes : One of 24 axis sequences as string or encoded tuple

        >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
        >>> numpy.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
        True
        """

        ai = Scalar.as_scalar(ai)
        aj = Scalar.as_scalar(aj)
        ak = Scalar.as_scalar(ak)
        Units.require_angle(ai._units_)
        Units.require_angle(aj._units_)
        Units.require_angle(ak._units_)

        (ai,aj,ak) = Qube.broadcast(ai,aj,ak)

        axes = axes.lower()
        try:
          (firstaxis, parity, repetition, frame) = Quaternion._AXES2TUPLE[axes]
        except (AttributeError, KeyError):
          Quaternion._TUPLE2AXES[axes]  # validation
          firstaxis, parity, repetition, frame = axes

        i = firstaxis + 1
        j = Quaternion._NEXT_AXIS[i+parity-1] + 1
        k = Quaternion._NEXT_AXIS[i-parity] + 1

        if frame:
            (ai, ak) = (ak, ai)
        if parity:
            aj = -aj

        half_ai = 0.5 * ai
        half_aj = 0.5 * aj
        half_ak = 0.5 * ak

        ci = half_ai.cos()._values_
        si = half_ai.sin()._values_
        cj = half_aj.cos()._values_
        sj = half_aj.sin()._values_
        ck = half_ak.cos()._values_
        sk = half_ak.sin()._values_

        cc = ci * ck
        cs = ci * sk
        sc = si * ck
        ss = si * sk

        q = np.empty(ai._shape_ + (4,))
        if repetition:
            q[...,0] = cj*(cc - ss)
            q[...,i] = cj*(cs + sc)
            q[...,j] = sj*(cc + ss)
            q[...,k] = sj*(cs - sc)
        else:
            q[...,0] = cj*cc + sj*ss
            q[...,i] = cj*sc - sj*cs
            q[...,j] = cj*ss + sj*cc
            q[...,k] = cj*cs - sj*sc

        if parity:
            q[...,j] *= -1.

        q *= np.sign(q[...,0])[...,np.newaxis]

        return Quaternion(q, Qube.or_(ai._mask_, aj._mask_, ak._mask_))

    #===========================================================================
    def to_euler(self, axes='rzxz'):
        return self.to_matrix3().to_euler(axes)

    #===========================================================================
    @staticmethod
    def from_euler_via_matrix(ai, aj, ak, axes='rzxz'):
        """Just for testing and validation."""

        m = Matrix3.from_euler(ai, aj, ak, axes)
        return Quaternion.from_matrixs(m)

################################################################################
# Useful class constants
################################################################################

Quaternion.ZERO     = Quaternion((0,0,0,0)).as_readonly()
Quaternion.XAXIS    = Quaternion((0,1,0,0)).as_readonly()
Quaternion.YAXIS    = Quaternion((0,0,1,0)).as_readonly()
Quaternion.ZAXIS    = Quaternion((0,0,0,1)).as_readonly()
Quaternion.IDENTITY = Quaternion((1,0,0,0)).as_readonly()
Quaternion.MASKED   = Quaternion((1,0,0,0), True).as_readonly()

################################################################################
# Fill in a reference to the Quaternion class inside the Qube object.
################################################################################

Qube.QUATERNION_CLASS = Quaternion

################################################################################
