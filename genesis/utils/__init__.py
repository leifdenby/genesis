import numpy as np
from scipy.constants import pi

import warnings

REQUIRED_DX_PRECISION = 4

def find_grid_spacing(mask):
    # NB: should also checked for stretched grids..
    try:
        x, y, z = mask.xt, mask.yt, mask.zt
        z_var = 'zt'
    except AttributeError:
        x, y, z = mask.x, mask.y, mask.z
        z_var = 'z'

    dx_all = np.diff(x.values)
    dy_all = np.diff(y.values)
    dx, dy = np.max(dx_all), np.max(dy_all)

    dx = np.round(dx, REQUIRED_DX_PRECISION)
    dy = np.round(dx, REQUIRED_DX_PRECISION)

    if not z_var in mask.coords:
        warnings.warn("z hasn't got any coordinates defined, assuming dz=dx")
        dz = np.max(dx)
    else:
        dz_all = np.diff(z.values)
        dz = np.max(dz_all)
        dz = np.round(dz, REQUIRED_DX_PRECISION)

        if not dx == dy or not dx == dz:
            raise NotImplementedError("Only isotropic grids are supported"
                                      "(dx,dy,dz)=({},{},{})".format(
                                          dx, dy, dz
                                      ))

    return dx

def find_vertical_grid_spacing(da):
    assert 'zt' in da.coords

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

    quartile = int(theta_mean / (pi/2.))

    @np.vectorize
    def _wrap(v):
        if v > (quartile+1)*pi/2.:
            return v - pi
        else:
            return v

    return _wrap(theta)
