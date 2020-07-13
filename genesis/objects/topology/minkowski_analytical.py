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


def calc_analytical_scales(shape):
    r_ = 100.0
    m_min, m_max = -6, 6
    m = np.linspace(-6.0, 6.0, (m_max - m_min) * 10 + 1)
    i = np.arange(len(m))
    lm = 2.0 ** m
    r = np.array([r_,] * len(lm))

    if shape == "cylinder":
        kwargs = dict(h=r*lm, r=r)
        fn_mink = cylinder.calc_minkowski_scales
    elif shape == "spheroid":
        kwargs = dict(a=r*lm, b=r)
        fn_mink = spheroid.calc_minkowski_scales
    else:
        raise NotImplementedError

    LWT = length_scales(fn_mink=_cylinder_minkowski, r=r, lm=lm)
    FP = filamentarity_planarity(
        **length_scales(**fn_mink(**kwargs)
        )
    )

    ds = xr.Dataset(coords=dict(i=i))
    for n, label in enumerate("length width thinckness".split(" ")):
        ds[label] = (("i",), LWT[n])
        ds[label].attrs["units"] = "1"

    ds["filamentarity"] = (("i",), FP[0])
    ds["planarity"] = (("i",), FP[1])
    ds["lm"] = (("i",), lm)

    return ds
