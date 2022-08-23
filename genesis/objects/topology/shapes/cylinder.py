import numpy as np
from scipy.constants import pi


@np.vectorize
def calc_minkowski_functionals(h, r):
    # h = lm*r
    # => lm = h/r
    lm = h / r

    V0 = pi * r**3.0 * lm
    V1 = pi / 3.0 * r**2.0 * (1 + lm)
    V2 = 1.0 / 3.0 * r * (pi + lm)
    V3 = 1.0

    return V0, V1, V2, V3
