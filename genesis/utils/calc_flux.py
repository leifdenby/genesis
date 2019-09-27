"""
Utility to compute vertical fluxes of scalars from 3D output in UCLALES
"""
import os
from collections import OrderedDict

import xarray as xr
import numpy as np

xr_chunks = dict(zt=20)

def get_horz_devition(da):
    dv = da - da.mean(dim=(da.dims[1], da.dims[2]), dtype=np.float64)
    dv = dv[:,:,:,1:] # remove sub-surface values
    dv.attrs['units'] = da.units
    dv.attrs['long_name'] = "{} horz deviation".format(da.long_name)
    dv.name = 'd_{}'.format(dv.name)
    return dv

def center_staggered_field(phi_da):
    """
    Create cell-centered values for staggered (velocity) fields
    """
    dim = [d for d in phi_da.dims if d.endswith('m')][0]
    newdim = dim.replace('m', 't')

    s_left, s_right = slice(0, -1), slice(1, None)

    # average vertical velocity to cell centers
    coord_vals = 0.5*(
        phi_da[dim].isel(**{dim:s_left}).values
      + phi_da[dim].isel(**{dim:s_right}).values
    )
    coord = xr.DataArray(coord_vals, coords={newdim: coord_vals},
                      attrs=dict(units='m'),dims=(newdim,))

    # create new coordinates for cell-centered vertical velocity
    coords=OrderedDict(phi_da.coords)
    del(coords[dim])
    coords[newdim] = coord

    phi_cc_vals = 0.5*(
        phi_da.isel(**{dim:s_left}).values
      + phi_da.isel(**{dim:s_right}).values
    )

    dims = list(phi_da.dims)
    dims[dims.index(dim)] = newdim

    phi_cc = xr.DataArray(
        phi_cc_vals, coords=coords, dims=dims,
        attrs=dict(units=phi_da.units, long_name=phi_da.long_name)
    )

    phi_cc.name = phi_da.name

    return phi_cc

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
