##########################################################################################
# tests/test_vector_as_index.py
#
# vector.as_index()
# vector.as_index_and_mask()
##########################################################################################

import numpy as np
import pytest

from polymath import Vector, Qube


def test_vector_as_index_array_to_test_for_indexing() -> None:
    """Array to test for indexing."""

    array = np.arange(1000).reshape(10,10,10)

    index1 = np.where(array % 13 == 0)

    index2 = (index1[0].reshape((7,11)),
              index1[1].reshape((7,11)),
              index1[2].reshape((7,11)))

    values = np.empty((7,11,3), dtype='int')
    values[...,0] = index2[0]
    values[...,1] = index2[1]
    values[...,2] = index2[2]
    vec = Vector(values)

    index13 = vec.as_index()

    indexed = array[index13]
    assert indexed.shape == (7,11)
    assert indexed.shape == vec.shape
    assert (np.all(indexed % 13) == 0)
    assert np.all(indexed.ravel() // 13 == np.arange(77))

    qube = Qube(array)
    assert qube[index13].shape == (7,11)
    assert qube[index13] % 13 == 0
    assert qube[index13].flatten() // 13 == np.arange(77)

    qube = Qube(array)
    assert qube[index13].shape == (7,11)
    assert qube[index13] % 13 == 0
    assert qube[index13].flatten() // 13 == np.arange(77)

    mask = np.zeros(vec.shape, dtype='bool')
    mask[0,0] = True
    mask[0,1] = True
    vec_one_masked = Vector(vec, mask)

    new_index = vec_one_masked.as_index(masked=None)
    assert qube[new_index].shape == (7*11-2,)
    assert qube[new_index] // 13 == np.arange(2,77)

    new_index = vec_one_masked.as_index(masked=(9,9,9))
    assert qube[new_index].shape == (7,11)
    assert qube[new_index][0,0] == 999
    assert qube[new_index][0,1] == 999
    flattened = qube[new_index].flatten()
    assert flattened[2:] == 13 * np.arange(2,77)

    vec = Vector([1.,2.,3.])
    with pytest.raises(TypeError) as cm:
        vec.as_index_and_mask()
    assert str(cm.value) == 'floating-point indexing is not permitted'
    vec = Vector(np.arange(12).reshape(6,2), drank=1)
    with pytest.raises(ValueError) as cm:
        vec.as_index_and_mask()
    assert str(cm.value) == ('Vector.as_index_and_mask() does not support '
                                        'denominators')
    vec = Vector([1,2,3], True)
    assert vec.as_index_and_mask(purge=True) == ((), False)
    indx, mask = vec.as_index_and_mask(purge=False)
    assert indx == (1,2,3)
    assert mask == True
    indx, mask = vec.as_index_and_mask(purge=False, masked=0)
    assert indx == (0,0,0)
    assert mask == True
    vals = np.arange(9).reshape(3,3)
    vec = Vector(vals, [False, False, True])
    indx, mask = vec.as_index_and_mask(purge=True)
    assert np.all(indx[0] == (0,3))
    assert np.all(indx[1] == (1,4))
    assert np.all(indx[2] == (2,5))
    assert mask == False
    vec = Vector(vals, [False, False, True])
    indx, mask = vec.as_index_and_mask(purge=False)
    assert np.all(indx[0] == (0,3,6))
    assert np.all(indx[1] == (1,4,7))
    assert np.all(indx[2] == (2,5,8))
    assert np.all(mask == [False, False, True])
    vec = Vector(vals, [False, False, True])
    indx, mask = vec.as_index_and_mask(purge=False, masked=0)
    assert np.all(indx[0] == (0,3,0))
    assert np.all(indx[1] == (1,4,0))
    assert np.all(indx[2] == (2,5,0))
    assert np.all(mask == [False, False, True])


##########################################################################################
