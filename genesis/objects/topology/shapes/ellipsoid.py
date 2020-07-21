import numpy as np
from scipy.constants import pi
from scipy.special import ellipkinc, ellipeinc
from scipy import integrate

import math


@np.vectorize
def calc_minkowski_functionals(a, b, c):
    """
    https://arxiv.org/abs/1104.5145
    """
    a, b, c = np.sort([a, b, c])[::-1]

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
    V1 = S / 6.0

    V2 = _calc_mean_curvature_integral(a=a, b=b, c=c) / (3.0 * pi)

    # no holes,
    V3 = 1.0

    return V0, V1, V2, V3


try:
    from numba import njit

    cos = math.cos
    sin = math.sin

    @njit
    def _integral(phi, theta, a, b, c):
        cos_t = cos(theta)
        sin_t = sin(theta)
        cos_p = cos(phi)
        sin_p = sin(phi)

        Rsq = (
            a * a * cos_t * cos_t
            + b * b * sin_t * sin_t * cos_p * cos_p
            + c * c * sin_t * sin_t * sin_p * sin_p
        )
        Hsq_less_abc = (a * b * c) / (
            b * b * c * c * cos_t * cos_t
            + c * c * a * a * sin_t * sin_t * cos_p * cos_p
            + a * a * b * b * sin_t * sin_t * sin_p * sin_p
        )

        return 0.5 * Hsq_less_abc * ((a * a + b * b + c * c - Rsq)) * sin_t

    def _calc_mean_curvature_integral(a, b, c):
        return integrate.dblquad(
            _integral, 0.0, pi, lambda x: 0.0, lambda x: 2.0 * pi, args=(a, b, c)
        )[0]


except ImportError:
    import warnings

    warnings.warn(
        "numba not available to using slow fall-back for ellipsoid mean curvature calculation"
    )

    def _calc_mean_curvature_integral(a, b, c):
        # I couldn't find an integral expression for the mean curvature so we will
        # do a numerical integral here
        sin = np.sin
        cos = np.cos
        sqrt = np.sqrt

        # Poelaert eqn 2
        def R(phi, theta):
            return sqrt(
                a ** 2.0 * cos(theta) ** 2.0
                + b ** 2.0 * sin(theta) ** 2.0 * cos(phi) ** 2.0
                + c ** 2.0 * sin(theta) ** 2.0 * sin(phi) ** 2.0
            )

        # Poelaert eqn 3
        def H(phi, theta):
            return (a * b * c) / sqrt(
                b ** 2.0 * c ** 2.0 * cos(theta) ** 2.0
                + c ** 2.0 * a ** 2.0 * sin(theta) ** 2.0 * cos(phi) ** 2.0
                + a ** 2.0 * b ** 2.0 * sin(theta) ** 2.0 * sin(phi) ** 2.0
            )

        def integral(phi, theta):
            # area element from Poelaert eqn 5, cancels with terms in mean
            # curvature equation 15, cancelling terms to reduce computation
            #
            # dS = sin(theta)*sqrt(
            #     (b*c)**2.0*cos(theta)**2.
            #     + (a*c)**2.0*sin(theta)**2.*cos(phi)**2.
            #     + (a*b)**2.0*sin(theta)**2.*sin(phi)**2.
            # )

            return (
                0.5
                * H(theta=theta, phi=phi) ** 2.0
                * (
                    (a ** 2.0 + b ** 2.0 + c ** 2.0 - R(theta=theta, phi=phi) ** 2.0)
                    / (a * b * c)
                )
                * sin(theta)
            )

        #        integrate.dblquad(f(y, x) , x0, x1, lambda x: y0, lambda x: y1)
        #        integrate.dblquad(f(phi, theta), theta0, theta1, lambda theta: phi0, lambda theta: phi1)
        return integrate.dblquad(integral, 0.0, pi, lambda x: 0.0, lambda x: 2.0 * pi)[
            0
        ]
