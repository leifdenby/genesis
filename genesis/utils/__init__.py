import numpy as np
from scipy.constants import pi
import xarray as xr

from collections import OrderedDict
import warnings

REQUIRED_DX_PRECISION = 4


def find_grid_spacing(mask):
    # NB: should also checked for stretched grids..
    try:
        x, y, z = mask.xt, mask.yt, mask.zt
        z_var = "zt"
    except AttributeError:
        x, y, z = mask.x, mask.y, mask.z
        z_var = "z"

    dx_all = np.diff(x.values)
    dy_all = np.diff(y.values)
    dx, dy = np.max(dx_all), np.max(dy_all)

    dx = np.round(dx, REQUIRED_DX_PRECISION)
    dy = np.round(dx, REQUIRED_DX_PRECISION)

    if z_var not in mask.coords:
        warnings.warn("z hasn't got any coordinates defined, assuming dz=dx")
        dz = np.max(dx)
    else:
        dz_all = np.diff(z.values)
        dz = np.max(dz_all)
        dz = np.round(dz, REQUIRED_DX_PRECISION)

        if not dx == dy or not dx == dz:
            raise NotImplementedError(
                "Only isotropic grids are supported"
                "(dx,dy,dz)=({},{},{})".format(dx, dy, dz)
            )

    return dx


def find_vertical_grid_spacing(da):
    assert "zt" in da.coords

    zt = da.zt

    dz_all = np.diff(zt.values)

    if np.min(dz_all) != np.max(dz_all):
        print(dz_all)
        raise Exception("Non-uniform vertical grid")

    return np.min(dz_all)


def angle_mean(theta):
    x = np.mean(np.cos(theta))
    y = np.mean(np.sin(theta))

    return np.arctan2(y, x)


def wrap_angles(theta):
    theta_mean = angle_mean(theta)

    quartile = int(theta_mean / (pi / 2.0))

    @np.vectorize
    def _wrap(v):
        if v > (quartile + 1) * pi / 2.0:
            return v - pi
        else:
            return v

    return _wrap(theta)


def center_staggered_field(phi_da):
    """
    Create cell-centered values for staggered (velocity) fields
    """
    dim = [d for d in phi_da.dims if d.endswith("m")][0]
    newdim = dim.replace("m", "t")

    s_left, s_right = slice(0, -1), slice(1, None)

    # average vertical velocity to cell centers
    coord_vals = 0.5 * (
        phi_da[dim].isel(**{dim: s_left}).values
        + phi_da[dim].isel(**{dim: s_right}).values
    )
    coord = xr.DataArray(
        coord_vals, coords={newdim: coord_vals}, attrs=dict(units="m"), dims=(newdim,)
    )

    # create new coordinates for cell-centered vertical velocity
    coords = OrderedDict(phi_da.coords)
    del coords[dim]
    coords[newdim] = coord

    phi_cc_vals = 0.5 * (
        phi_da.isel(**{dim: s_left}).values + phi_da.isel(**{dim: s_right}).values
    )

    dims = list(phi_da.dims)
    dims[dims.index(dim)] = newdim

    phi_cc = xr.DataArray(
        phi_cc_vals,
        coords=coords,
        dims=dims,
        attrs=dict(units=phi_da.units, long_name=phi_da.long_name),
    )

    phi_cc.name = phi_da.name

    return phi_cc
