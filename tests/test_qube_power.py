##########################################################################################
# tests/test_qube_power.py
##########################################################################################

import numpy as np

from polymath import Matrix


def test_qube_power() -> None:
    """Exercise qube power."""

    np.random.seed(9947)
    a = Matrix(np.random.randint(-100, 101, (10,5,2,2)))
    assert a**0 == a.identity()
    assert a**1 == a
    assert a**2 == a*a
    assert a**3 == a*a*a
    assert a**4 == a*a*a*a
    assert a**5 == a*a*a*a*a
    assert a**6 == a*a*a*a*a*a
    assert (np.all(abs((a**7 ).vals - (a*a*a*a*a*a*a).vals)) < 1.e-13)
    assert (np.all(abs((a**8 ).vals - (a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    assert (np.all(abs((a**9 ).vals - (a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    assert (np.all(abs((a**10).vals - (a*a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    assert (np.all(abs((a**11).vals - (a*a*a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    assert (np.all(abs((a**12).vals - (a*a*a*a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    assert (np.all(abs((a**13).vals - (a*a*a*a*a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    assert (np.all(abs((a**14).vals - (a*a*a*a*a*a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    assert (np.all(abs((a**15).vals - (a*a*a*a*a*a*a*a*a*a*a*a*a*a*a).vals)) < 1.e-13)
    b = a.inverse()
    assert a**-1 == b
    assert (np.all(abs((a**-2 ).vals - (b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-3 ).vals - (b*b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-4 ).vals - (b*b*b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-5 ).vals - (b*b*b*b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-6 ).vals - (b*b*b*b*b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-7 ).vals - (b*b*b*b*b*b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-8 ).vals - (b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-9 ).vals - (b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-10).vals - (b*b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-11).vals - (b*b*b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-12).vals - (b*b*b*b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-13).vals - (b*b*b*b*b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-14).vals - (b*b*b*b*b*b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    assert (np.all(abs((a**-15).vals - (b*b*b*b*b*b*b*b*b*b*b*b*b*b*b).vals)) < 1.e-13)
    a.insert_deriv('t', Matrix(np.random.randn(10,5,2,2)))
    assert np.all((a**0).d_dt.vals == 0.)
    assert (a**1).d_dt == a.d_dt
    assert (np.all(abs((a**2 ).d_dt.vals - (a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**3 ).d_dt.vals - (a*a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**4 ).d_dt.vals - (a*a*a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**5 ).d_dt.vals - (a*a*a*a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**6 ).d_dt.vals - (a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**7 ).d_dt.vals - (a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**8 ).d_dt.vals - (a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**9 ).d_dt.vals - (a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**10).d_dt.vals - (a*a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**11).d_dt.vals - (a*a*a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**12).d_dt.vals - (a*a*a*a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**13).d_dt.vals - (a*a*a*a*a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**14).d_dt.vals - (a*a*a*a*a*a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**15).d_dt.vals - (a*a*a*a*a*a*a*a*a*a*a*a*a*a*a).d_dt.vals)) < 1.e-13)
    b = a.inverse()
    assert (np.all(abs((a**-1 ).d_dt.vals - (b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-2 ).d_dt.vals - (b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-3 ).d_dt.vals - (b*b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-4 ).d_dt.vals - (b*b*b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-5 ).d_dt.vals - (b*b*b*b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-6 ).d_dt.vals - (b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-7 ).d_dt.vals - (b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-8 ).d_dt.vals - (b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-9 ).d_dt.vals - (b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-10).d_dt.vals - (b*b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-11).d_dt.vals - (b*b*b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-12).d_dt.vals - (b*b*b*b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-13).d_dt.vals - (b*b*b*b*b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-14).d_dt.vals - (b*b*b*b*b*b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)
    assert (np.all(abs((a**-15).d_dt.vals - (b*b*b*b*b*b*b*b*b*b*b*b*b*b*b).d_dt.vals)) < 1.e-13)


##########################################################################################
