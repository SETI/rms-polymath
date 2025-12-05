##########################################################################################
# tests/test_vector3_advanced.py
# Vector3 advanced tests: n-D arrays, round-trips, type preservation
##########################################################################################

import numpy as np
import unittest

from polymath import Scalar, Vector3


class Test_Vector3_Advanced(unittest.TestCase):

    def runTest(self):

        np.random.seed(2599)

        # Test n-D arrays
        v5 = Vector3(np.random.randn(2, 3, 3))
        self.assertEqual(v5.shape, (2, 3))
        self.assertEqual(v5.item, (3,))
        self.assertEqual(v5.vals.shape, (2, 3, 3))

        # Test higher-dimensional arrays
        v6 = Vector3(np.random.randn(4, 5, 6, 3))
        self.assertEqual(v6.shape, (4, 5, 6))
        self.assertEqual(v6.item, (3,))
        self.assertEqual(v6.vals.shape, (4, 5, 6, 3))

        # Test from_ra_dec_length with n-D inputs
        ra_2d = Scalar([[0., np.pi/2], [np.pi, 3*np.pi/2]])
        dec_2d = Scalar([[0., 0.], [0., 0.]])
        v23 = Vector3.from_ra_dec_length(ra_2d, dec_2d, 2.)
        self.assertEqual(v23.shape, (2, 2))
        # First should be along x, second along y, etc.
        self.assertTrue(np.allclose(v23.vals[0, 0], [2., 0., 0.], atol=1e-10))

        # Test to_ra_dec_length with n-D
        v25 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
        ra25, dec25, length25 = v25.to_ra_dec_length()
        self.assertEqual(ra25.shape, (2, 2))
        self.assertEqual(dec25.shape, (2, 2))
        self.assertEqual(length25.shape, (2, 2))

        # Test from_cylindrical with n-D inputs
        radius_2d = Scalar([[1., 2.], [3., 4.]])
        longitude_2d = Scalar([[0., np.pi/2], [np.pi, 3*np.pi/2]])
        v28 = Vector3.from_cylindrical(radius_2d, longitude_2d, 0.)
        self.assertEqual(v28.shape, (2, 2))

        # Test to_cylindrical with n-D
        v30 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
        radius30, longitude30, z30 = v30.to_cylindrical()
        self.assertEqual(radius30.shape, (2, 2))
        self.assertEqual(longitude30.shape, (2, 2))
        self.assertEqual(z30.shape, (2, 2))

        # Test longitude with n-D
        v33 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[-1., 0., 0.], [0., -1., 0.]]]))
        lon33 = v33.longitude()
        self.assertEqual(lon33.shape, (2, 2))

        # Test latitude with n-D
        v36 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
        lat36 = v36.latitude()
        self.assertEqual(lat36.shape, (2, 2))

        # Test spin with n-D
        v39 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
        pole39 = Vector3([0., 0., 1.])
        angle39 = Scalar(np.pi/2)
        v39_spun = v39.spin(pole39, angle39)
        self.assertEqual(v39_spun.shape, (2, 2))

        # Test offset_angles with n-D
        v42 = Vector3(np.array([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 0.]]]))
        v43 = Vector3([1., 0., 0.])
        lon_off2, lat_off2 = v42.offset_angles(v43)
        self.assertEqual(lon_off2.shape, (2, 2))
        self.assertEqual(lat_off2.shape, (2, 2))

        # Test dot with n-D
        v50 = Vector3(np.random.randn(4, 1, 5, 3))
        v51 = Vector3(np.random.randn(8, 5, 3))
        dot50 = v50.dot(v51)
        # Broadcasting: (4, 1, 5) and (8, 5) -> (4, 8, 5)
        self.assertEqual(dot50.shape, (4, 8, 5))

        # Test norm with n-D
        v53 = Vector3(np.random.randn(2, 3, 3))
        norm53 = v53.norm()
        self.assertEqual(norm53.shape, (2, 3))

        # Test unit with n-D
        v55 = Vector3(np.random.randn(2, 3, 3))
        unit55 = v55.unit()
        self.assertEqual(unit55.shape, (2, 3))

        # Test cross with n-D
        v58 = Vector3(np.random.randn(4, 1, 5, 3))
        v59 = Vector3(np.random.randn(8, 5, 3))
        cross58 = v58.cross(v59)
        # Broadcasting: (4, 1, 5) and (8, 5) -> (4, 8, 5)
        self.assertEqual(cross58.shape, (4, 8, 5))

        # Test cross_product_as_matrix with n-D
        v74 = Vector3(np.random.randn(2, 3, 3))
        m74 = v74.cross_product_as_matrix()
        self.assertEqual(m74.shape, (2, 3))
        self.assertEqual(m74.numer, (3, 3))

        # Test element_mul with n-D
        v77 = Vector3(np.random.randn(2, 3, 3))
        v78 = Vector3(np.random.randn(2, 3, 3))
        elem_mul77 = v77.element_mul(v78)
        self.assertEqual(elem_mul77.shape, (2, 3))

        # Test element_div with n-D
        v81 = Vector3(np.random.randn(2, 3, 3))
        v82 = Vector3(np.random.randn(2, 3, 3))
        elem_div81 = v81.element_div(v82)
        self.assertEqual(elem_div81.shape, (2, 3))

        # Test sep with n-D
        v70 = Vector3(np.random.randn(2, 3, 3))
        v71 = Vector3(np.random.randn(2, 3, 3))
        sep70 = v70.sep(v71)
        self.assertEqual(sep70.shape, (2, 3))

        # Test complex n-D case
        v87 = Vector3(np.random.randn(3, 4, 5, 6, 3))
        self.assertEqual(v87.shape, (3, 4, 5, 6))
        self.assertEqual(v87.item, (3,))
        self.assertEqual(v87.vals.shape, (3, 4, 5, 6, 3))

        # Test that operations preserve type
        v88 = Vector3([1., 2., 3.])
        v89 = Vector3([4., 5., 6.])
        v_result = v88 + v89
        self.assertEqual(type(v_result), Vector3)

        v_result2 = v88 * 2.
        self.assertEqual(type(v_result2), Vector3)

        # Test round-trip conversions
        v90 = Vector3([1., 2., 3.])
        ra90, dec90, length90 = v90.to_ra_dec_length()
        v90_recon = Vector3.from_ra_dec_length(ra90, dec90, length90)
        self.assertTrue(np.allclose(v90.vals, v90_recon.vals, atol=1e-10))

        v91 = Vector3([1., 2., 3.])
        radius91, longitude91, z91 = v91.to_cylindrical()
        v91_recon = Vector3.from_cylindrical(radius91, longitude91, z91)
        self.assertTrue(np.allclose(v91.vals, v91_recon.vals, atol=1e-10))

        # Test n-D round-trip
        v92 = Vector3(np.random.randn(2, 3, 3))
        ra92, dec92, length92 = v92.to_ra_dec_length()
        v92_recon = Vector3.from_ra_dec_length(ra92, dec92, length92)
        self.assertEqual(v92_recon.shape, (2, 3))
        self.assertTrue(np.allclose(v92.vals, v92_recon.vals, atol=1e-10))

##########################################################################################
