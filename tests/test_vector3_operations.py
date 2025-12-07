##########################################################################################
# tests/test_vector3_operations.py
# Vector3 instance methods: coordinate conversions, transformations, and vector operations
##########################################################################################

import numpy as np
import unittest

from polymath import Scalar, Vector3, Matrix


class Test_Vector3_Operations(unittest.TestCase):

    def runTest(self):

        np.random.seed(2599)

        # Test to_ra_dec_length method
        v24 = Vector3([1., 0., 0.])
        ra24, dec24, length24 = v24.to_ra_dec_length()
        self.assertEqual(type(ra24), Scalar)
        self.assertEqual(type(dec24), Scalar)
        self.assertEqual(type(length24), Scalar)
        self.assertTrue(np.allclose(ra24.vals, 0., atol=1e-10))
        self.assertTrue(np.allclose(dec24.vals, 0., atol=1e-10))
        self.assertTrue(np.allclose(length24.vals, 1., atol=1e-10))

        # Test to_cylindrical method
        v29 = Vector3([1., 0., 0.])
        radius29, longitude29, z29 = v29.to_cylindrical()
        self.assertEqual(type(radius29), Scalar)
        self.assertEqual(type(longitude29), Scalar)
        self.assertEqual(type(z29), Scalar)
        self.assertTrue(np.allclose(radius29.vals, 1., atol=1e-10))
        self.assertTrue(np.allclose(longitude29.vals, 0., atol=1e-10))
        self.assertTrue(np.allclose(z29.vals, 0., atol=1e-10))

        # Test longitude method
        v31 = Vector3([1., 0., 0.])
        lon31 = v31.longitude()
        self.assertEqual(type(lon31), Scalar)
        self.assertTrue(np.allclose(lon31.vals, 0., atol=1e-10))

        v32 = Vector3([0., 1., 0.])
        lon32 = v32.longitude()
        self.assertTrue(np.allclose(lon32.vals, np.pi/2, atol=1e-10))

        # Test latitude method
        v34 = Vector3([1., 0., 0.])
        lat34 = v34.latitude()
        self.assertEqual(type(lat34), Scalar)
        self.assertTrue(np.allclose(lat34.vals, 0., atol=1e-10))

        v35 = Vector3([0., 0., 1.])
        lat35 = v35.latitude()
        self.assertTrue(np.allclose(lat35.vals, np.pi/2, atol=1e-10))

        # Test spin method
        v37 = Vector3([1., 0., 0.])
        pole = Vector3([0., 0., 1.])  # z-axis
        angle = Scalar(np.pi/2)
        v37_spun = v37.spin(pole, angle)
        self.assertEqual(type(v37_spun), Vector3)
        # Rotating (1,0,0) about z-axis by pi/2 should give (0,1,0)
        self.assertTrue(np.allclose(v37_spun.vals, [0., 1., 0.], atol=1e-10))

        # Test spin with angle=None (uses pole magnitude via arcsin)
        v38 = Vector3([1., 0., 0.])
        # Use pole with magnitude 1.0 so arcsin(1.0) = pi/2
        pole38 = Vector3([0., 0., 1.])  # magnitude is 1.0, arcsin(1.0) = pi/2
        v38_spun = v38.spin(pole38)
        self.assertEqual(type(v38_spun), Vector3)
        # For v38 = (1,0,0) and pole38 with magnitude 1.0 (arcsin gives pi/2), the spun vector should be (0,1,0)
        self.assertTrue(np.allclose(v38_spun.vals, [0., 1., 0.], atol=1e-10))

        # Test offset_angles method
        v40 = Vector3([1., 0., 0.])
        v41 = Vector3([0., 1., 0.])
        lon_off, lat_off = v40.offset_angles(v41)
        self.assertEqual(type(lon_off), Scalar)
        self.assertEqual(type(lat_off), Scalar)
        # Should have some angular offset
        self.assertTrue(np.isfinite(lon_off.vals))
        self.assertTrue(np.isfinite(lat_off.vals))

        # Test inherited methods from Vector - to_scalar
        v44 = Vector3(np.random.randn(4, 1, 5, 3))
        s44 = v44.to_scalar(0)
        self.assertEqual(type(s44), Scalar)
        self.assertEqual(s44.shape, v44.shape)

        # Test to_scalars
        scalars44 = v44.to_scalars()
        self.assertEqual(len(scalars44), 3)
        self.assertEqual(type(scalars44[0]), Scalar)
        self.assertEqual(scalars44[0].shape, v44.shape)

        # Test as_column
        v45 = Vector3([1., 2., 3.])
        m45 = v45.as_column()
        self.assertEqual(type(m45), Matrix)
        self.assertEqual(m45.numer, (3, 1))
        self.assertTrue(np.allclose(m45.vals[..., 0], [1., 2., 3.]))

        # Test as_row
        v46 = Vector3([1., 2., 3.])
        m46 = v46.as_row()
        self.assertEqual(type(m46), Matrix)
        self.assertEqual(m46.numer, (1, 3))
        self.assertTrue(np.allclose(m46.vals[0, :], [1., 2., 3.]))

        # Test as_diagonal
        v47 = Vector3([1., 2., 3.])
        m47 = v47.as_diagonal()
        self.assertEqual(type(m47), Matrix)
        self.assertEqual(m47.numer, (3, 3))
        self.assertTrue(np.allclose(m47.vals[0, 0], 1.))
        self.assertTrue(np.allclose(m47.vals[1, 1], 2.))
        self.assertTrue(np.allclose(m47.vals[2, 2], 3.))

        # Test dot
        v48 = Vector3([1., 2., 3.])
        v49 = Vector3([4., 5., 6.])
        dot48 = v48.dot(v49)
        self.assertEqual(type(dot48), Scalar)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        self.assertTrue(np.allclose(dot48.vals, 32.))

        # Test norm
        v52 = Vector3([3., 4., 0.])
        norm52 = v52.norm()
        self.assertEqual(type(norm52), Scalar)
        # sqrt(3^2 + 4^2 + 0^2) = 5
        self.assertTrue(np.allclose(norm52.vals, 5.))

        # Test unit
        v54 = Vector3([3., 4., 0.])
        unit54 = v54.unit()
        self.assertEqual(type(unit54), Vector3)
        # Should be normalized: (3/5, 4/5, 0)
        self.assertTrue(np.allclose(unit54.vals, [0.6, 0.8, 0.], atol=1e-10))
        self.assertTrue(np.allclose(unit54.norm().vals, 1., atol=1e-10))

        # Test cross
        v56 = Vector3([1., 0., 0.])
        v57 = Vector3([0., 1., 0.])
        cross56 = v56.cross(v57)
        self.assertEqual(type(cross56), Vector3)
        # Should be (0, 0, 1)
        self.assertTrue(np.allclose(cross56.vals, [0., 0., 1.], atol=1e-10))

        # Test ucross
        v60 = Vector3([1., 0., 0.])
        v61 = Vector3([0., 1., 0.])
        ucross60 = v60.ucross(v61)
        self.assertEqual(type(ucross60), Vector3)
        # Should be unit vector (0, 0, 1)
        self.assertTrue(np.allclose(ucross60.vals, [0., 0., 1.], atol=1e-10))
        self.assertTrue(np.allclose(ucross60.norm().vals, 1., atol=1e-10))

        # Test outer
        v62 = Vector3([1., 2., 3.])
        v63 = Vector3([4., 5., 6.])
        outer62 = v62.outer(v63)
        self.assertEqual(type(outer62), Matrix)
        # Outer product should be 3x3 matrix
        self.assertEqual(outer62.numer, (3, 3))

        # Test perp
        v64 = Vector3([1., 1., 0.])
        v65 = Vector3([1., 0., 0.])
        perp64 = v64.perp(v65)
        self.assertEqual(type(perp64), Vector3)
        # Component of (1,1,0) perpendicular to (1,0,0) should be (0,1,0)
        self.assertTrue(np.allclose(perp64.vals, [0., 1., 0.], atol=1e-10))

        # Test proj
        v66 = Vector3([1., 1., 0.])
        v67 = Vector3([1., 0., 0.])
        proj66 = v66.proj(v67)
        self.assertEqual(type(proj66), Vector3)
        # Projection of (1,1,0) onto (1,0,0) should be (1,0,0)
        self.assertTrue(np.allclose(proj66.vals, [1., 0., 0.], atol=1e-10))

        # Test sep
        v68 = Vector3([1., 0., 0.])
        v69 = Vector3([0., 1., 0.])
        sep68 = v68.sep(v69)
        self.assertEqual(type(sep68), Scalar)
        # Separation angle between (1,0,0) and (0,1,0) should be pi/2
        self.assertTrue(np.allclose(sep68.vals, np.pi/2, atol=1e-10))

        # Test cross_product_as_matrix
        v72 = Vector3([1., 2., 3.])
        m72 = v72.cross_product_as_matrix()
        self.assertEqual(type(m72), Matrix)
        self.assertEqual(m72.numer, (3, 3))
        # Test that matrix * vector equals cross product
        v73 = Vector3([4., 5., 6.])
        cross72 = v72.cross(v73)
        m72_v73 = m72 * v73
        self.assertTrue(np.allclose(m72_v73.vals, cross72.vals, atol=1e-10))

        # Test element_mul
        v75 = Vector3([1., 2., 3.])
        v76 = Vector3([4., 5., 6.])
        elem_mul75 = v75.element_mul(v76)
        self.assertEqual(type(elem_mul75), Vector3)
        # Should be (4, 10, 18)
        self.assertTrue(np.allclose(elem_mul75.vals, [4., 10., 18.]))

        # Test element_div
        v79 = Vector3([4., 10., 18.])
        v80 = Vector3([4., 5., 6.])
        elem_div79 = v79.element_div(v80)
        self.assertEqual(type(elem_div79), Vector3)
        # Should be (1, 2, 3)
        self.assertTrue(np.allclose(elem_div79.vals, [1., 2., 3.], atol=1e-10))

        # Test __abs__ (norm)
        v83 = Vector3([3., 4., 0.])
        abs83 = abs(v83)
        self.assertEqual(type(abs83), Scalar)
        self.assertTrue(np.allclose(abs83.vals, 5.))

##########################################################################################
