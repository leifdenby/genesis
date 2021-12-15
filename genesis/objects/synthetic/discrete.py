# coding: utf-8
"""
Routines for creating masks of parameterised synthetic 3D shapes
"""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.optimize
import xarray as xr


def make_grid(dx, lx=3e3, lz=2e3):
    ly = lx
    nx, ny, nz = int(lx / dx) + 1, int(ly / dx) + 1, int(lz / dx) + 1
    x = np.linspace(-lx / 2.0, lx / 2.0, nx)
    y = np.linspace(-ly / 2.0, ly / 2.0, ny)
    z = np.linspace(0, lz, nz)

    grid = xr.Dataset(coords=OrderedDict(x=x, y=y, z=z))

    return grid


# shearing function
def f_shear(ls, h0):
    return lambda h: ls / h0 ** 2.0 * h ** 2


def len_fn_approx(h, ls, h0):
    "approximate distance along length"
    return np.sqrt(h ** 2.0 + f_shear(ls, h0)(h) ** 2.0)


def len_fn(h, ls, h0):
    "numerically integrated distance along length"
    # ax^2 + bc + c
    def dldh(h_):
        return np.sqrt(1.0 + (ls / h0 ** 2.0 * 2.0 * h_) ** 2.0)

    return scipy.integrate.quad(dldh, 0, h)[0]


def find_scaling(ls, h0):
    """find height fraction `alpha` at which the top of the shape
    is sheared a horizontal distance `ls` while keeping the length
    of the shape constant"""

    def fn(alpha):
        return h0 - len_fn(alpha * h0, ls=ls, h0=h0)

    return scipy.optimize.brentq(fn, 0.5, 1.0)


def make_plume_mask(grid, r0, h, shear_distance=0.0):
    """
    Return a dataset with a synthetic plume mask, `r0` denotes the characteristic radius,
    `h` the height and `shear_distance` the vertical sheared distance of the plume at
    height `h`
    """
    if h > grid.z.max():
        raise Exception(
            "Grid too small to contain plume, please increase z coordinate max"
        )

    ds = grid.copy()

    a = shear_distance / h ** 2
    ds["x_c"] = 0.0 * ds.z + a * ds.z ** 2.0
    ds["y_c"] = 0.0 * ds.z
    ds["xy_dist"] = np.sqrt((ds.x - ds.x_c) ** 2.0 + (ds.y - ds.y_c) ** 2.0)
    ds["r"] = r0 + 0.0 * ds.z
    ds["mask"] = ds.xy_dist < ds.r

    s = find_scaling(ls=shear_distance, h0=h)
    ds["mask"] = ds.mask.where(ds.z < h * s, False)

    ds["type"] = "thermal"

    # ensure coordinates are in correct order
    ds = ds.transpose("x", "y", "z")

    # cloud identification code isn't so good with objects that touch domain edge...
    ds["mask"].values[:, :, :1] = False
    ds["mask"].values[:, :, -1:] = False

    return ds


def make_thermal_mask(grid, r0, h, z_offset=0.0, shear_distance=0.0):
    """
    Return a dataset with a synthetic thermal mask, `r0` denotes the characteristic radius,
    `h` the height and `shear_distance` the vertical sheared distance of the thermal at
    height `h`
    """
    ds = grid.copy()

    s = find_scaling(ls=shear_distance, h0=h)

    z_c = s * h / 2.0 + z_offset

    a = shear_distance / h ** 2
    ds["x_c"] = 0.0 * ds.z + a * ds.z ** 2.0
    ds["y_c"] = 0.0 * ds.z
    ds["xy_dist"] = np.sqrt((ds.x - ds.x_c) ** 2.0 + (ds.y - ds.y_c) ** 2.0)
    ds["z_dist"] = np.abs(ds.z - z_c)

    ds["mask"] = (ds.xy_dist / r0) ** 2.0 + (ds.z_dist / (h * s / 2.0)) ** 2.0 < 1.0
    ds["type"] = "plume"

    # ensure coordinates are in correct order
    ds = ds.transpose("x", "y", "z")

    return ds


def make_mask(h, length, dx, shape, l_shear):
    r0 = h / 2.0 / length

    # ensure the domain contains the full object
    lz = 2.5 * r0 * length
    lx_shear = 2 * l_shear
    lx_noshear = 2.5 * r0
    grid = make_grid(dx=dx, lx=lx_noshear + lx_shear, lz=lz)

    if l_shear != np.inf:
        grid = grid.sel(x=slice(-lx_noshear, None))

    if shape == "plume":
        ds = make_plume_mask(
            grid,
            r0=r0,
            h=h,
            shear_distance=l_shear,
        )
    elif shape == "thermal":
        ds = make_thermal_mask(grid, r0=r0, h=h, shear_distance=l_shear)

    ds["r0"] = r0
    ds.attrs["dx"] = dx

    return ds


def plot_mask(ds):
    fig, ax = plt.subplots()
    m = ds.sel(y=0, method="nearest").mask
    m.where(m, other=np.nan).plot(y="z", ax=ax, add_colorbar=False, cmap="Greys_r")
    plt.gca().set_aspect(1)
    plt.axis("off")
    plt.title("")
