# coding: utf-8

import numpy as np
from scipy.constants import pi
import xarray as xr

from .shapes import ellipsoid, cylinder, spheroid


def length_scales(V0, V1, V2, V3):
    """
    Calculate length-scales from Minkowski functionals
    """
    T = V0 / (2.0 * V1)
    W = 2 * V1 / (pi * V2)
    L = 3 * V2 / (4 * V3)

    return L, W, T


def filamentarity_planarity(L, W, T):
    """
    Calculate filamentarity and planarity from Minkowski length-scales length
    (L), width (W) and thickness (T)
    """
    P = (W - T) / (W + T)
    F = (L - W) / (L + W)

    return F, P


def _calc_ellipsoid_scales(alpha, r0=600.0):
    """
    so that
      a*b*c*4/3*pi = v0 = 4/3*pi*r0**3
    and
      a = alpha*r0
    and
      a >= b >= c

    The second and third constraint mean that
      b_min = c_max
    =>
      alpha*r0*b_min**2 = r0**3
      b_min = sqrt(r0**2/alpha)
    And
      a_min = b_max = r0 (a sphere)
    """
    a = alpha * r0
    b = np.linspace(np.sqrt(r0 ** 2.0 / alpha), r0, 100)
    c = r0 ** 3.0 / (a * b)

    lm = b / c

    LWT = length_scales(*ellipsoid.spheroid_minkowski_functionals(a=a, b=b, c=c))
    # recompute FP with the W/T values we know are within the range we want to
    # plot
    FP = filamentarity_planarity(*LWT)
    ds = xr.Dataset(coords=dict(i=np.arange(len(b))))
    for n, label in enumerate("length width thickness".split(" ")):
        ds[label] = (("i",), LWT[n])
        ds[label].attrs["units"] = "1"

    ds.attrs["alpha"] = alpha
    ds["filamentarity"] = (("i",), FP[0])
    ds["planarity"] = (("i",), FP[1])
    ds["lambda"] = (("i"), lm)
    ds.attrs["r0"] = r0

    return ds


def _calc_lambda_parameterised_scale(shape):
    r_ = 100.0
    m_min, m_max = -6, 6
    m = np.linspace(-6.0, 6.0, (m_max - m_min) * 10 + 1)
    i = np.arange(len(m))
    lm = 2.0 ** m
    r = np.array([r_] * len(lm))

    if shape == "cylinder":
        kwargs = dict(h=r * lm, r=r)
        fn_mink = cylinder.calc_minkowski_scales
    elif shape == "spheroid":
        kwargs = dict(a=r * lm, b=r)
        fn_mink = spheroid.calc_minkowski_scales
    else:
        raise NotImplementedError(shape)

    LWT = length_scales(*fn_mink(**kwargs))
    FP = filamentarity_planarity(*LWT)

    ds = xr.Dataset(coords=dict(i=i))
    for n, label in enumerate("length width thinckness".split(" ")):
        ds[label] = (("i",), LWT[n])
        ds[label].attrs["units"] = "1"

    ds["filamentarity"] = (("i",), FP[0])
    ds["planarity"] = (("i",), FP[1])
    ds["lm"] = (("i",), lm)

    return ds


def calc_analytical_scales(shape, alpha=None):
    if shape in ["cylinder", "spheroid"]:
        return _calc_lambda_parameterised_scale(shape=shape)
    elif shape == "ellipsoid":
        if alpha is None:
            raise Exception(
                f"To use {shape} method the variable `alpha` must be provided"
            )
        return _calc_ellipsoid_scales(alpha=alpha)
    else:
        raise NotImplementedError(shape)
