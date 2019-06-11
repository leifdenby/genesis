"""
Utility to compute vertical fluxes of scalars from 3D output in UCLALES
"""
import os
from collections import OrderedDict

import xarray as xr

xr_chunks = dict(zt=20)

def get_horz_devition(var_name):
    fn_dv = FN_BASE.format("d_{}".format(var_name))

    if not os.path.exists(fn_dv):
        print("calculating horizontal deviation of {}".format(var_name))
        fn = FN_BASE.format(var_name)
        da = xr.open_dataarray(fn, decode_times=False, chunks=xr_chunks)
        dv = da - da.mean(dim=da.dims[1]).mean(dim=da.dims[2])
        dv = dv[:,:,:,1:] # remove sub-surface values
        dv.attrs['units'] = da.units
        dv.attrs['longname'] = "{} horz deviation".format(da.longname)
        dv.name = 'd_{}'.format(dv.name)
        dv.to_netcdf(fn_dv)
        del(da)

    return xr.open_dataarray(fn_dv, decode_times=False, chunks=xr_chunks)

def z_center_field(phi_da):
    assert phi_da.dims[-1] == 'zm'

    # average vertical velocity to cell centers
    zt_vals = 0.5*(phi_da.zm[1:].values + phi_da.zm[:-1].values)
    zt = xr.DataArray(zt_vals, coords=dict(zt=zt_vals),
                      attrs=dict(units='m'),dims=('zt',))

    # create new coordinates for cell-centered vertical velocity
    coords=OrderedDict(phi_da.coords)
    del(coords['zm'])
    coords['zt'] = zt

    phi_cc_vals = 0.5*(phi_da[...,1:].values + phi_da[...,:-1].values)

    dims = list(phi_da.dims)
    dims[dims.index('zm')] = 'zt'

    phi_cc = xr.DataArray(
        phi_cc_vals, coords=coords, dims=dims,
        attrs=dict(units=phi_da.units, longname=phi_da.longname)
    )

    phi_cc.name = phi_da.name

    return phi_cc

def get_cell_centered_vertical_velocity():

    fn_w_zt = FN_BASE.format("w_zt")

    if not os.path.exists(fn_w_zt):
        print("cell-centering vertical velocity")
        fn = FN_BASE.format('w')
        da = xr.open_dataarray(fn, decode_times=False)
        da_zt = z_center_field(da)
        da_zt.to_netcdf(fn_w_zt)
        del(da)

    return xr.open_dataarray(fn_w_zt, decode_times=False, chunks=xr_chunks)


def compute_vertical_flux(var_name):
    """
    Compute vertical flux of `var_name`
    """
    dphi = get_horz_devition(var_name=var_name)
    w = get_cell_centered_vertical_velocity()

    assert dphi.time == w.time
    # if dims aren't identical xarray ends up allocating huge arrays for
    # dealing with the missing overlap
    assert w.dims == dphi.dims

    # old routines using new array
    # phi_flux = dphi*w
    # phi_flux.attrs['units'] = "{} {}".format(w.units, dphi.units)
    # dphi_longname = dphi.longname.replace('horz deviation', '').strip()
    # phi_flux.attrs['longname'] = "{} vertical flux".format(dphi_longname)

    dphi_longname = dphi.longname.replace('horz deviation', '').strip()

    # to inplace update to conserve memory
    phi_flux = dphi
    phi_flux *= w

    phi_flux.attrs['units'] = "{} {}".format(w.units, dphi.units)
    phi_flux.attrs['longname'] = "{} vertical flux".format(dphi_longname)
    phi_flux.name = '{}_flux'.format(var_name)

    return phi_flux


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument('var_name', type=str)
    argparser.add_argument('base_name', type=str)

    args = argparser.parse_args()

    global FN_BASE
    FN_BASE = args.base_name + ".{}.nc"

    var_name = args.var_name
    phi_flux = compute_vertical_flux(var_name=var_name)

    out_filename = FN_BASE.format('{}_flux'.format(var_name))
    phi_flux.to_netcdf(out_filename)

    print("Wrote output to {}".format(out_filename))
