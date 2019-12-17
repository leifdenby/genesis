"""
Utility to compute vertical fluxes of scalars from 3D output in UCLALES
"""
import os

import numpy as np

from . import center_staggered_field

xr_chunks = dict(zt=20)

def get_horz_devition(da):
    dv = da - da.mean(dim=(da.dims[1], da.dims[2]), dtype=np.float64)
    dv = dv.sel(zt=slice(0, None)) # remove sub-surface values
    dv.attrs['units'] = da.units
    dv.attrs['long_name'] = "{} horz deviation".format(da.long_name)
    dv.name = 'd_{}'.format(dv.name)
    return dv

def z_center_field(da):
    return center_staggered_field(da)

def compute_vertical_flux(da, w):
    """
    Compute vertical flux of `da`
    """
    dphi = get_horz_devition(da=da)
    if w.dims != da.dims:
        w = z_center_field(da=w)

    assert dphi.time == w.time
    # if dims aren't identical xarray ends up allocating huge arrays for
    # dealing with the missing overlap
    assert w.dims == dphi.dims

    # old routines using new array
    # phi_flux = dphi*w
    # phi_flux.attrs['units'] = "{} {}".format(w.units, dphi.units)
    # dphi_long_name = dphi.long_name.replace('horz deviation', '').strip()
    # phi_flux.attrs['long_name'] = "{} vertical flux".format(dphi_long_name)

    dphi_long_name = dphi.long_name.replace('horz deviation', '').strip()

    # to inplace update to conserve memory
    phi_flux = dphi
    phi_flux *= w

    phi_flux.attrs['units'] = "{} {}".format(w.units, dphi.units)
    phi_flux.attrs['long_name'] = "{} vertical flux".format(dphi_long_name)
    phi_flux.name = '{}_flux'.format(da.name)

    return phi_flux
