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


def _calc_ellipsoid_scales(lm, r0=300.0, N_points=100):
    """
    Calculates Minkowski scales for an ellipsoid (axis lengths a, b and c)
    parameterised by a reference radius r0 (specifying the volume as that of a
    sphere with radius r0) and a major axis (a) fraction a=alpha*r0, so that
      4/3*pi*a*b*c = v0 = 4/3*pi*r0**3
    and
      c = lm*a
    and
      c >= b >= a

    The second and third constraints mean that
      b_min = a_max
    =>
      4/3*pi*a*b*c = 4/3*pi*r0**3
      lm*a_max*a_max*b_min = r0**3
      lm*b_min**3. = r0**3
      b_min = r0/(lm)^(1/3)

      b_max = c_min
    =>
      a*b_max*c_min = r0**3.0
      c_min/lm*b_max*c_min = r0**3
      b_max**3. = r0 * (lm)^1/3


    Once c and b are given, then a can be calculated from
      4/3*pi*a*b*c = 4/3*pi*r0**3
      a = r0**3/(b*c)
    """
    # c = lm*r0
    r1 = r0 * (1.0 / lm) ** (1.0 / 3.0)

    # make a non-linear spacing for the points, we need more a lower radius
    # values, so instead of
    #   b = np.linspace(r1, r1 * lm, N_points)
    # we use the expression below
    def scaling_fn(x):
        c = 5.0
        return (np.exp(x * c) - 1.0) / (np.exp(c) - 1.0)

    b = r1 + (r1 * lm - r1) * scaling_fn(np.linspace(0.0, 1.0, N_points))

    c = lm * r1
    a = r0 ** 3.0 / (b * c)

    alpha = b / a

    mink_scales = ellipsoid.calc_minkowski_functionals(a=a, b=b, c=c)
    LWT = length_scales(*mink_scales)
    # recompute FP with the W/T values we know are within the range we want to
    # plot
    FP = filamentarity_planarity(*LWT)
    ds = xr.Dataset(coords=dict(i=np.arange(len(b))))
    for n, label in enumerate("length width thickness".split(" ")):
        ds[label] = (("i",), LWT[n])
        ds[label].attrs["units"] = "1"
    for n, label in enumerate("v0 v1 v2 v3".split(" ")):
        ds[label] = (("i",), mink_scales[n])
        ds[label].attrs["units"] = "1"

    ds.attrs["lm"] = lm
    ds["a"] = (("i"), a)
    ds["b"] = (("i"), b)
    ds["c"] = (("i"), c * np.ones_like(a))
    ds["filamentarity"] = (("i",), FP[0])
    ds["planarity"] = (("i",), FP[1])
    ds["alpha"] = (("i"), alpha)
    ds.attrs["r0"] = r0
    ds.attrs["r1"] = r1

    return ds


def _calc_lambda_parameterised_scale(shape, r0=300.0, N_points=100):
    """
    Calculates Minkowski scales for spheroid or cylindrical shapes with a fixed
    volume (same a sphere of radius r0) with axis r, and lambda*r
    """
    # V = 4/3*pi*r^3
    v0 = 4.0 / 3.0 * pi * r0 ** 3.0
    m = np.linspace(-6.0, 6.0, N_points)
    i = np.arange(len(m))
    lm = 2.0 ** m

    if shape == "cylinder":
        # V = h*pi*r**2 and h=lambda*r, r=r
        # => V = pi*r**3*lambda
        # => r = [V/(pi*lambda)]^(1/3)
        r = (v0 / (pi * lm)) ** (1.0 / 3.0)
        kwargs = dict(h=r * lm, r=r)
        fn_mink = cylinder.calc_minkowski_functionals
    elif shape == "spheroid":
        # V = 4/3*pi*c*a^2 and a=r, c=lambda*r
        # => 4/3*pi*r0^3 = 4/3*pi*r^3*lambda
        # => r0^3 = r^3*lambda
        # => r = r0*(lambda)^(1/3)
        r1 = r0 * (1.0 / lm) ** (1.0 / 3.0)

        kwargs = dict(a=r1, c=r1 * lm)
        fn_mink = spheroid.calc_minkowski_functionals
    else:
        raise NotImplementedError(shape)

    mink_scales = fn_mink(**kwargs)
    LWT = length_scales(*mink_scales)
    FP = filamentarity_planarity(*LWT)

    ds = xr.Dataset(coords=dict(i=i))
    for n, label in enumerate("length width thickness".split(" ")):
        ds[label] = (("i",), LWT[n])
        ds[label].attrs["units"] = "1"
    for n, label in enumerate("v0 v1 v2 v3".split(" ")):
        ds[label] = (("i",), mink_scales[n])
        ds[label].attrs["units"] = "1"
    for k, v in kwargs.items():
        ds[k] = ("i"), v

    ds["filamentarity"] = (("i",), FP[0])
    ds["planarity"] = (("i",), FP[1])
    ds["lm"] = (("i",), lm)
    ds.attrs["r0"] = r0

    return ds


def calc_analytical_scales(shape, lm=None, N_points=100, r0=300):
    if shape in ["cylinder", "spheroid"]:
        ds = _calc_lambda_parameterised_scale(shape=shape, N_points=N_points, r0=r0)
    elif shape == "ellipsoid":
        if lm is None:
            raise Exception(f"To use {shape} method the variable `lm` must be provided")
        ds = _calc_ellipsoid_scales(lm=lm, N_points=N_points, r0=r0)
    else:
        raise NotImplementedError(shape)
    ds["shape"] = shape
    return ds
