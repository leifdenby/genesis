from scipy.constants import pi
import numpy as np


@np.vectorize
def calc_minkowski_scales(a, b):
    # a = lm*r
    # b = r
    # => lm = a/b, r = b
    lm = a / b
    r = b

    @np.vectorize
    def f(x):
        # http://www.suitcaseofdreams.net/inverse_functions.htm#P1
        d = np.sqrt(1 + 0j - x ** 2.0)

        if d.imag > 0.0:
            assert np.all(d.real == 0.0)
            return np.log(x + np.sqrt(x ** 2.0 - 1.0)) / d.imag
        else:
            return np.arccos(x) / d.real

    V0 = 4.0 * pi / 3.0 * r ** 3.0 * lm
    V1 = pi / 3.0 * r ** 2.0 * (1 + f(1.0 / lm))
    V2 = 2.0 / 3.0 * r * (lm + f(lm))
    V3 = 1.0

    return V0, V1, V2, V3
