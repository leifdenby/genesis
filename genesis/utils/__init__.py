import numpy as np

import warnings


def find_grid_spacing(mask):
    # NB: should also checked for stretched grids..
    xt, yt, zt = mask.xt, mask.yt, mask.zt

    dx_all = np.diff(xt.values)
    dy_all = np.diff(yt.values)
    dx, dy = np.max(dx_all), np.max(dy_all)

    if not 'zt' in mask.coords:
        warnings.warn("zt hasn't got any coordinates defined, assuming dz=dx")
        dz = np.max(dx)
    else:
        dz_all = np.diff(zt.values)
        dz = np.max(dz_all)
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

    print(dz_all)

    assert np.min(dz_all) == np.max(dz_all)

    return np.min(dz_all)
