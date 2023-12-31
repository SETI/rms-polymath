################################################################################
# Scalar.max() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Units

class Test_Scalar_max(unittest.TestCase):

  # setUp
  def setUp(self):
    Qube.PREFER_BUILTIN_TYPES = True

  # tearDown
  def tearDown(self):
    Qube.PREFER_BUILTIN_TYPES = False

  # runTest
  def runTest(self):

    np.random.seed(7250)

    # Individual values
    self.assertEqual(Scalar(0.3).max(), 0.3)
    self.assertEqual(type(Scalar(0.3).max()), float)

    self.assertEqual(Scalar(4).max(), 4)
    self.assertEqual(type(Scalar(4).max()), int)

    self.assertTrue(Scalar(4, mask=True).max().mask)
    self.assertEqual(type(Scalar(4, mask=True).max()), Scalar)

    # Multiple values
    self.assertTrue(Scalar((1,2,3)).max() == 3)
    self.assertEqual(type(Scalar((1,2,3)).max()), int)

    self.assertTrue(Scalar((1,2,3)).argmax() == 2)

    self.assertTrue(Scalar((1.,2.,3.)).max() == 3.)
    self.assertEqual(type(Scalar((1.,2,3)).max()), float)

    self.assertTrue(Scalar((1,2,3)).argmax() == 2)

    # Arrays
    N = 400
    x = Scalar(np.random.randn(N).reshape((2,4,5,10)))
    self.assertEqual(x.max(), np.max(x.values))

    argmax = x.argmax()
    self.assertEqual(x.flatten()[argmax], x.max())

    # Test units
    values = np.random.randn(10)
    random = Scalar(values, units=Units.KM)
    self.assertEqual(random.max().units, Units.KM)

    values = np.random.randn(10)
    random = Scalar(values, units=Units.DEG)
    self.assertEqual(random.max().units, Units.DEG)

    values = np.random.randn(10)
    random = Scalar(values, units=None)
    self.assertEqual(type(random.max()), float)

    # Masks
    N = 1000
    x = Scalar(np.random.randn(N), mask=(np.random.randn(N) < -1.))

    maxval = -np.inf
    for i in range(N):
        if (not x.mask[i]) and (x.values[i] > maxval):
            maxval = x.values[i]

    self.assertEqual(maxval, x.max())

    argmax = x.argmax()
    self.assertEqual(x[argmax], x.max())

    # If we mask the maximum value(s), the maximum should decrease
    x = x.mask_where_eq(maxval)
    self.assertTrue(x.max() < maxval)

    argmax = x.argmax()
    self.assertEqual(x.flatten()[argmax], x.max())

    masked = Scalar(x, mask=True)
    self.assertTrue(masked.max().mask)
    self.assertTrue(type(masked.max()), Scalar)

    argmax = x.argmax()
    self.assertEqual(x[argmax], x.max())

    # Denominators
    a = Scalar([1.,2.], drank=1)
    self.assertRaises(ValueError, a.max)

    # Maxes over axes
    x = -Scalar(np.arange(30).reshape(2,3,5))
    m0 = x.max(axis=0)
    m01 = x.max(axis=(0,1))
    m012 = x.max(axis=(-1,1,0))

    self.assertEqual(m0.shape, (3,5))
    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], np.max(x.values[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    for k in range(5):
        self.assertEqual(m01[k], np.max(x.values[:,:,k]))

    self.assertEqual(np.shape(m012), ())
    self.assertEqual(type(m012), int)
    self.assertEqual(m012, 0)

    argmax = x.argmax(axis=0)
    for j in range(3):
      for k in range(5):
        self.assertEqual(x[argmax[j,k],j,k], m0[j,k])

    # Maxes with masks
    values = -np.arange(30).reshape(2,3,5)
    mask = (values > -5)
    x = Scalar(values, mask)
    m0 = x.max(axis=0)
    m01 = x.max(axis=(0,1))
    m012 = x.max(axis=(-1,1,0))

    self.assertEqual(m0.shape, (3,5))
    xx = x.values.copy()
    xx[xx > -5] -= 100
    for j in range(3):
      for k in range(5):
        self.assertEqual(m0[j,k], np.max(xx[:,j,k]))

    self.assertEqual(m01.shape, (5,))
    self.assertEqual(m01, [-5,-6,-7,-8,-9])

    self.assertEqual(m012, -5)

    argmax = x.argmax(axis=0)
    for j in range(3):
      for k in range(5):
        self.assertEqual(x[argmax[j,k],j,k], m0[j,k])

    values = -np.arange(30).reshape(2,3,5)
    mask = (values > -5)
    mask[:,1] = True
    x = Scalar(values, mask)
    m0 = x.max(axis=0)

    for j in (0,2):
      for k in range(5):
        self.assertEqual(m0[j,k], np.max(xx[:,j,k]))

    j = 1
    for k in range(5):
        self.assertEqual(m0[j,k], Scalar.MASKED)
        self.assertTrue(np.all(m0[j,k].values == np.max(x.values[:,j,k])))

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
