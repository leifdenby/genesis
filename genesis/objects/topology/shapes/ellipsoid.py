import numpy as np
from scipy.constants import pi
from scipy.special import ellipkinc, ellipeinc


@np.vectorize
def spheroid_minkowski_functionals(a, b, c):
    """https://arxiv.org/abs/1104.5145
    """
    a, b, c = np.sort([a, b, c])[::-1]
    print(a, b, c)

    V0 = 4.0 * pi / 3.0 * a * b * c

    # surface area, eqn. 9 in Poelaert et al 2011
    # sqrt = np.sqrt
    # e = sqrt(1.0 - c ** 2.0 / a ** 2.0)
    m = (a ** 2 * (b ** 2 - c ** 2)) / (b ** 2 * (a ** 2 - c ** 2))

    # F = scipy.special.ellipkinc
    # E = scipy.special.ellipeinc
    # asin = np.arcsin

    # S = 2*pi*(c**2.
    #           + 0*b*c**2./sqrt(a**2. - c**2.)*E(a*asin(e), m)
    #           + 0*b*sqrt(a**2. - c**2.)*F(a*asin(e), m)
    #         )

    # the equations above from Poelaert don't work as is, I think because E and F call
    # parameters are defined differently
    # using https://www.johndcook.com/blog/2014/07/06/ellipsoid-surface-area/ for now

    arccos = np.arccos
    sin = np.sin
    cos = np.cos

    phi = arccos(c / a)
    temp = ellipeinc(phi, m) * sin(phi) ** 2 + ellipkinc(phi, m) * cos(phi) ** 2
    S = 2 * pi * (c ** 2 + a * b * temp / sin(phi))
    V1 = S

    # mean curvature, eqn. 15 in Poelaert et al 2011
    H = a * c / b
    V2 = H ** 4.0 / (a ** 2.0 * b ** 2.0 * c ** 2.0)

    # no holes,
    V3 = 1.0

    return V0, V1, V2, V3
