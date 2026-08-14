##########################################################################################
# tests/test_pair_swapxy.py
##########################################################################################

import numpy as np

from polymath import Pair


def test_pair_clip2d() -> None:
    """Exercise pair clip2d."""

    a = Pair([[1,2],[3,4],[5,6]])
    assert a.clip2d([2,3],[4,5], remask=False) == [[2,3],[3,4],[4,5]]
    assert (np.all(a.clip2d([2,3],[4,5], remask=True).mask ==
                                    [True,False,True]))
    assert a.clip2d(None,[4,5], remask=False) == [[1,2],[3,4],[4,5]]
    assert (np.all(a.clip2d(None,[4,5], remask=True).mask ==
                                    [False,False,True]))
    lower = Pair([2,3], True)
    assert a.clip2d(lower,[4,5], remask=False) == [[1,2],[3,4],[4,5]]
    assert (np.all(a.clip2d(lower,[4,5], remask=True).mask ==
                                    [False,False,True]))


##########################################################################################
