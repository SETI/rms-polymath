##########################################################################################
# tests/test_qube_shrink.py
##########################################################################################

import numpy as np

from polymath import Qube, Scalar, Vector3, Boolean


def test_qube_shrink_corners() -> None:
    """corners."""

    np.random.seed(1207)

    values = np.ones((100,200))
    a = Scalar(values)
    assert a.corners == ((0,0), (100,200))
    a = Scalar(values, mask=True)
    assert a.corners == ((0,0), (0,0))
    mask = np.ones((100,200), dtype='bool')
    a = Scalar(values, mask)
    assert a.corners == ((0,0), (0,0))
    mask.fill(False)
    a = Scalar(values, mask)
    assert a.corners == ((0,0), (100,200))
    mask[0] = True
    a = Scalar(values, mask)
    assert a.corners == ((1,0), (100,200))
    mask[:,0] = True
    a = Scalar(values, mask)
    assert a.corners == ((1,1), (100,200))
    values = np.ones((100,200,3))
    a = Vector3(values)
    assert a.corners == ((0,0), (100,200))
    a = Vector3(values, mask=True)
    assert a.corners == ((0,0), (0,0))
    mask = np.ones((100,200), dtype='bool')
    a = Vector3(values, mask)
    assert a.corners == ((0,0), (0,0))
    mask.fill(False)
    a = Vector3(values, mask)
    assert a.corners == ((0,0), (100,200))
    mask[0] = True
    a = Vector3(values, mask)
    assert a.corners == ((1,0), (100,200))
    mask[:,0] = True
    a = Vector3(values, mask)
    assert a.corners == ((1,1), (100,200))


def test_qube_shrink_slicer() -> None:
    """_slicer."""

    np.random.seed(1207)

    values = np.ones((100,200))
    mask = np.zeros((100,200), dtype='bool')
    mask[0] = True
    mask[:,0] = True
    a = Scalar(values, mask)
    assert a._slicer == (slice(1, 100, None), slice(1, 200, None))
    assert a[a._slicer].shape == (99,199)
    a = Scalar(values, mask=True)
    assert a._slicer == (slice(0, 0, None), slice(0, 0, None))
    assert a[a._slicer].shape == (0,0)
    a = Scalar(values, mask=False)
    assert a._slicer == (slice(0, 100, None), slice(0, 200, None))
    assert a[a._slicer].shape == (100,200)
    values = np.ones((100,200,3))
    mask = np.zeros((100,200), dtype='bool')
    mask[0] = True
    mask[:,0] = True
    a = Vector3(values, mask)
    assert a._slicer == (slice(1, 100, None), slice(1, 200, None))
    assert a[a._slicer].shape == (99,199)
    a = Vector3(values, mask=True)
    assert a._slicer == (slice(0, 0, None), slice(0, 0, None))
    assert a[a._slicer].shape == (0,0)
    a = Vector3(values, mask=False)
    assert a._slicer == (slice(0, 100, None), slice(0, 200, None))
    assert a[a._slicer].shape == (100,200)


def test_qube_shrink_antimask() -> None:
    """antimask."""

    np.random.seed(1207)

    values = np.ones((100,200))
    mask = np.zeros((100,200), dtype='bool')
    mask[0] = True
    mask[:,0] = True
    a = Scalar(values, mask)
    assert np.all(a.mask ^ a.antimask)
    a = Scalar(values, False)
    assert np.all(a.mask ^ a.antimask)
    assert a[a.antimask] == a
    a = Scalar(values, True)
    assert np.all(a.mask ^ a.antimask)
    assert a[a.antimask].shape == (np.sum(a.antimask),200)
    assert a[np.newaxis][:0].shape == (0,100,200)
    values = np.ones((100,200,3))
    mask = np.zeros((100,200), dtype='bool')
    mask[0] = True
    mask[:,0] = True
    a = Vector3(values, mask)
    assert np.all(a.mask ^ a.antimask)
    a = Vector3(values, False)
    assert np.all(a.mask ^ a.antimask)
    assert a[a.antimask] == a
    a = Vector3(values, True)
    assert np.all(a.mask ^ a.antimask)
    assert a[a.antimask].shape == (np.sum(a.antimask),200)


def test_qube_shrink_test_unshrink_with_and_without_ignore_unshrunk_as_cached() -> None:
    """Test unshrink with and without _IGNORE_UNSHRUNK_AS_CACHED."""

    np.random.seed(1207)

    for ignore in (False, True):

        Qube._IGNORE_UNSHRUNK_AS_CACHED = ignore

        # shrink and unshrink, unmasked
        values = np.arange(100*200).reshape(100,200)
        a = Scalar(values)

        b = a.shrink(True)
        assert a == b

        b = a.shrink(False)
        assert b == Scalar.MASKED

        antimask = np.zeros((100,200), dtype='bool')
        antimask[0] = True
        b = a.shrink(antimask)
        assert b.shape == (200,)
        assert np.all(b.values == np.arange(200))

        c = b.unshrink(antimask)
        assert a.shape == c.shape
        assert a[0] == c[0]
        assert np.all(c.mask[1:])

        # shrink and unshrink, masked
        values = np.arange(100*200).reshape(100,200)
        a = Scalar(values, mask=(np.random.randn(100,200) < 0))

        b = a.shrink(True)
        assert a == b

        b = a.shrink(False)
        assert b == Scalar.MASKED

        antimask = np.zeros((100,200), dtype='bool')
        antimask[0] = True
        b = a.shrink(antimask)
        assert b.shape == (200,)
        assert np.all(b.values == np.arange(200))

        c = b.unshrink(antimask)
        assert a.shape == c.shape
        assert a[0] == c[0]
        assert np.all(c.mask[1:])

        dist = Scalar(np.arange(-50,50)[:,np.newaxis]**2 +
                      np.arange(-100,100)**2).sqrt()
        mask = (dist > 40)
        a = Scalar(dist, mask)
        assert a.corners == ((10,60),(91,141))

        b = a.shrink(a.antimask)
        c = b.unshrink(a.antimask)
        assert a == c

        antimask = a.antimask
        v = Vector3(np.random.randn(100,200,3),
                    mask=np.random.randn(100,200) < 0.)
        v2 = v.shrink(antimask)
        v3 = v2.unshrink(antimask)
        assert v[antimask] == v3[antimask]

        v = v.mask_where(~antimask)
        v2 = v.shrink(antimask)
        v3 = v2.unshrink(antimask)
        assert v == v3

        v3 = v2.unshrink(antimask)
        assert v == v3

        # Shape control
        a = Scalar(np.arange(900).reshape(100,3,3), drank=1, mask=True)
        b = a.shrink(False)
        aa = b.unshrink(False, shape=a.shape)
        assert aa == a
        aa = b.unshrink(False)
        assert aa.shape == ()

        a = Boolean(np.arange(900).reshape(100,3,3) % 2 == 0, mask=True)
        b = a.shrink(False)
        aa = b.unshrink(False, shape=a.shape)
        assert aa == a
        aa = b.unshrink(False)
        assert aa.shape == ()

        a = Vector3(np.random.randn(100,3,3), drank=1, mask=True)
        b = a.shrink(False)
        aa = b.unshrink(False, shape=a.shape)
        assert aa == a
        aa = b.unshrink(False)
        assert aa.shape == ()

        # Zero-sized objects

        a = a[:0]
        assert a.shape == (0,)
        b = a.shrink(True)
        assert b.shape == (0,)
        aa = b.unshrink(True, (0,))
        assert aa.shape == (0,)

        aa = b.unshrink(True)
        assert aa.shape == (0,)

        aa = b.unshrink(False)
        assert aa.shape == ()

        # Unshaped, unmasked objects
        a = Scalar(8.)

        antimask = (np.random.randn(7,5) < 0.)
        b = a.shrink(antimask)
        c = b.unshrink(antimask)
        assert a == b
        assert a == c
#         self.assertTrue((a == b).all())
#         self.assertEqual(len(b), np.sum(antimask))
#         cc = c.copy()
#         cc[c == Scalar.MASKED] = 0.
#         self.assertEqual(8. * np.asfarray(antimask), cc)

        antimask[...] = False
        b = a.shrink(antimask)
        assert b == Scalar.MASKED
        c = b.unshrink(antimask)
        assert c == Scalar.MASKED

        antimask = True
        b = a.shrink(antimask)
        assert a == b
        c = b.unshrink(antimask)
        assert a == c

        antimask = False
        b = a.shrink(antimask)
        assert b == Scalar.MASKED
        c = b.unshrink(antimask)
        assert c == Scalar.MASKED

        # Unshaped, masked objects
        a = Scalar(0., mask=True)

        antimask = (np.random.randn(7,5) < 0.)
        b = a.shrink(antimask)
        assert a == b
        c = b.unshrink(antimask)
        assert a == c

        antimask[...] = False
        b = a.shrink(antimask)
        assert b == a
        c = b.unshrink(antimask)
        assert c == a

        antimask = True
        b = a.shrink(antimask)
        assert a == b
        c = b.unshrink(antimask)
        assert a == c

        antimask = False
        b = a.shrink(antimask)
        assert a == b
        c = b.unshrink(antimask)
        assert a == c

        # Shaped object, unshaped mask
        a = Scalar(np.random.randn(7,5), mask=False)

        antimask = True
        b = a.shrink(antimask)
        assert a == b
        c = b.unshrink(antimask)
        assert a == c

        antimask = False
        b = a.shrink(antimask)
        assert b == Scalar.MASKED
        c = b.unshrink(antimask)
        assert c == Scalar.MASKED

        # Object becomes totally masked only upon shrinking
        antimask = (np.random.randn(7,5) < 0.)
        a = Scalar(np.random.randn(7,5), mask=antimask)
        b = a.shrink(antimask)
        assert b == Scalar.MASKED
        c = b.unshrink(antimask)
        assert c == Scalar.MASKED

        # Calculations
        b = Vector3(np.random.randn(100,3), mask=np.random.randn(100) > 1.)
        c = Scalar(np.random.randn(3,1,100), mask=np.random.randn(3,1,100) > 1.)
        d = Vector3(np.random.randn(100,3), mask=np.random.randn(100) > 1.)

        for value in [1., Scalar(np.random.randn(2,100))]:
            for mask in [True, False,
                         np.ones((2,100), dtype='bool'),
                         np.zeros((2,100), dtype='bool'),
                         np.random.randn(2,100) > 1.]:

                if np.shape(value) == () and np.shape(mask) != ():
                    continue

                a = Scalar(value, mask)

                value1 = a * b + c * d

                for antishape in (value1.shape, value1.shape[1:], value1.shape[2:]):
                    for antimask in [True, False,
                                     np.ones(antishape, dtype='bool'),
                                     np.zeros(antishape, dtype='bool'),
                                     np.random.randn(*antishape) > 1]:

                        aa = a.shrink(antimask)
                        bb = b.shrink(antimask)
                        cc = c.shrink(antimask)
                        dd = d.shrink(antimask)

                        value2 = aa * bb + cc * dd

                        test1 = value1.shrink(antimask) == value2
                        if isinstance(test1, bool):
                            assert test1
                        else:
                            assert test1.all()

                        if np.shape(antimask) == ():
                            test_mask = antimask
                        else:
                            pad = len(value1.shape) - len(np.shape(antimask))
                            test_mask = pad * (slice(None),) + (antimask,)

                        assert (value1[test_mask] == value2).all()

                        value3 = value2.unshrink(antimask)
                        assert (value3.shape in ((), value1.shape))

                        if value3.shape == ():
                            assert (value1[test_mask] == value3).all()
                        else:
                            assert ((value1[test_mask] ==
                                             value3[test_mask]).all())

    # Fully masked after shrink


def test_qube_shrink_qube_antimask() -> None:
    """(qube, antimask)."""

    np.random.seed(1207)

    values = [
        (Boolean(True, False), False),
        (Boolean(True, True ), False),
        (Scalar([1,2], False), False),
        (Scalar([1,2], True ), False),
        (Scalar([1,2], np.array([False, True])), False),
        (Scalar([1,2], np.array([False, True])), np.array([False, True])),
        (Scalar([1.,2.], False), False),
        (Scalar([1.,2.], np.array([False, True])), np.array([False, True])),
        (Scalar(np.arange(100), False), False),
    ]
    for (a, antimask) in values:
        aa = a.shrink(antimask)
        assert aa.shape == ()
        b = aa.unshrink(antimask, a.shape)
        assert a.shape == b.shape
        assert a.dtype() == b.dtype()


##########################################################################################
