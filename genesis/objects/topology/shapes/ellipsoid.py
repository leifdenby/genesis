import numpy as np
from scipy.constants import pi
from scipy.special import ellipkinc, ellipeinc
from scipy import integrate


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
    sqrt = np.sqrt

    phi = arccos(c / a)
    temp = ellipeinc(phi, m) * sin(phi) ** 2 + ellipkinc(phi, m) * cos(phi) ** 2
    S = 2 * pi * (c ** 2 + a * b * temp / sin(phi))
    V1 = S / 6.0

    # I couldn't find an integral expression for the mean curvature so we will
    # do a numerical integral here
    H = a * c / b
    C = H ** 4.0 / (a ** 2.0 * b ** 2.0 * c ** 2.0)

    def R(phi, theta):
        return sqrt(
            a**2.0 * cos(theta)**2.0
            + b**2.0 * sin(theta)**2.0 * cos(phi)**2.0
            + c**2.0 * sin(theta)**2.0 * sin(phi)**2.0
        )

    def H(phi, theta):
        return (a*b*c)/sqrt(
            b**2.0 * c**2.0 * cos(theta)**2.0
            + c**2.0 * a**2.0 * sin(theta)**2.0 * cos(phi)**2.0
            + a**2.0 * b**2.0 * sin(theta)**2.0 * sin(phi)**2.0
        )

    def mean_curvature(phi, theta):
        return H(theta=theta, phi=phi) ** 3.0 * (
            (a ** 2.0 + b**2.0 + c**2.0 - R(theta=theta, phi=phi)**2.0)
            / (a**2 * b**2.0 * c**2.0)
        )

    # def mean_curvature(u, v):
        # return a*b*c*(
                # cos(2*v)*(a**2 + b**2 - 2*c**2)
                # + 2*(b**2 - a**2)*cos(2*u)*sin(v)**2
                # + 3*(a**2 + b**2) + 2*c**2
            # )/(8*(
                # c**2*sin(v)**2*(a**2*sin(u)**2
                # + b**2*cos(u)**2)
                # + a**2*b**2*cos(v)**2)**(3/2)
            # )


    # v = theta, u = phi
    def integral(phi, theta):
        x = a*cos(phi)*sin(theta)
        y = b*sin(phi)*sin(theta)
        z = c*cos(theta)
        r_squared = x**2.0 + y**2.0 + z**2.0
        # dS = r_squared*sin(theta)

        dS = sin(theta)*sqrt(
            (b*c)**2.0*cos(theta)**2.
            + (a*c)**2.0*sin(theta)**2.*cos(phi)**2.
            + (a*b)**2.0*sin(theta)**2.*sin(phi)**2.
        )

        return mean_curvature(phi=phi, theta=theta)*dS/2.0
        # return mean_curvature(u=phi, v=theta)*dS

    # mc_int = integrate.dblquad(integral, 0, 2*pi, lambda x: -pi/2.0, lambda x: pi/2.0)[0]
    #        integrate.dblquad(f(y, x) , x0, x1, lambda x: y0, lambda x: y1)
    #        integrate.dblquad(f(phi, theta), theta0, theta1, lambda theta: phi0, lambda theta: phi1)
    mc_int, int_err = integrate.dblquad(integral, 0.0, pi, lambda x: 0.0, lambda x: 2.0*pi)
    print("%%", mc_int/ (4*pi*a))
    print("$$", int_err)
    V2 = mc_int / (3.0*pi)

    # no holes,
    V3 = 1.0

    return V0, V1, V2, V3
