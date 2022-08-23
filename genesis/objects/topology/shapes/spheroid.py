import numpy as np
from scipy.constants import pi


@np.vectorize
def calc_minkowski_functionals(a, c):
    """
    Calculate Minkowski functionals for a spheroid as given in equation 3 of
    Schmalzing et al 1999 (https://arxiv.org/abs/astro-ph/9904384v2)

    NOTE: the expression for V1 is missing a factor of lambda

    Spheroid with two axis length a=r and one with length c=lambda*r
    """
    # c = lm*r
    # a = r
    # => lm = c/a, r = a
    lm = c / a
    r = a

    @np.vectorize
    def f(x):
        if x > 1.0:
            # in this case
            #  i*arccos(x) = ln(x + sqrt(x**2 - 1))
            # and so
            #  arccos(x) / sqrt(1-x**2.) = ln(x + sqrt(x**2. - 1))/sqrt(x**2. - 1)
            return np.log(x + np.sqrt(x**2.0 - 1)) / np.sqrt(x**2.0 - 1)
        else:
            return np.arccos(x) / np.sqrt(1 - x**2.0)

    V = 4.0 * pi / 3.0 * r**3.0 * lm
    V0 = V

    # V1 = A / 6.
    # NOTE: there is a factor of lambda missing in Schmalzing et al in front of f(...)
    V1 = pi / 3.0 * r**2.0 * (1 + lm * f(1.0 / lm))

    V2 = 2.0 / 3.0 * r * (lm + f(lm))
    V3 = 1.0

    return V0, V1, V2, V3
