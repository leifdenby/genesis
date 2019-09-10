"""
Utilities for labelling 3D objects from a mask and calculating their
characteristics scales using Minkowski functionals
"""
import os
import warnings

import xarray as xr
import numpy as np

import cloud_identification


def calc_scales(object_labels, dx):
    """
    Compute characteristic scales from Minkowski functionals in 3D using
    labelled objects in `object_labels`
    """
    mf = cloud_identification.topological_scales(labels=object_labels, dx=dx)
    V0 = cloud_identification.N3(object_labels)

    n_objects = mf.shape[1]
    # the cloud_identification code doesn't return properties for the zeroth
    # object, since that is empty space
    object_ids = np.arange(1, n_objects+1)

    nn = ~np.isnan(mf[0,:])
    mf = mf[:,nn]
    V0 = V0[nn]
    object_ids = object_ids[nn]

    planarity = cloud_identification.planarity(mf=mf)
    filamentarity = cloud_identification.filamentarity(mf=mf)

    ds = xr.Dataset(coords=dict(object_id=object_ids))

    mf_variables = "length_m width_m thickness_m genus_m".split(" ")
    for n, v in enumerate(mf_variables):
        units = 'm' if v != "genus_m" else "1"
        ds[v] = xr.DataArray(data=mf[n,:], coords=dict(object_id=object_ids,),
                             dims=('object_id',), attrs=dict(units=units))

    ds['planarity'] = xr.DataArray(data=planarity,
                                   coords=dict(object_id=object_ids,),
                                   dims=('object_id',),
                                   attrs=dict(units="1"))

    ds['filamentarity'] = xr.DataArray(data=filamentarity,
                                       coords=dict(object_id=object_ids,),
                                       dims=('object_id',),
                                       attrs=dict(units="1"))

    volume = V0*dx**3.
    ds['volume'] = xr.DataArray(data=volume,
                                       coords=dict(object_id=object_ids,),
                                       dims=('object_id',),
                                       attrs=dict(units="m^3"))

    ds['num_cells'] = xr.DataArray(data=V0,
                                       coords=dict(object_id=object_ids,),
                                       dims=('object_id',),
                                       attrs=dict(units="1"))


    return ds


def label_objects(mask, splitting_scalar=None, remove_at_edge=True):
    def _remove_at_edge(object_labels):
        mask_edge = np.zeros_like(mask)
        mask_edge[:,:,1] = True  # has to be k==1 because object identification codes treats actual edges as ghost cells
        mask_edge[:,:,-2] = True
        cloud_identification.remove_intersecting(object_labels, mask_edge)

    if splitting_scalar is None:
        splitting_scalar = np.ones_like(mask)
    else:
        assert mask.shape == splitting_scalar.shape
        assert mask.dims == splitting_scalar.dims
        # NB: should check coord values too

    object_labels = cloud_identification.number_objects(
        splitting_scalar, mask=mask
    )

    if remove_at_edge:
        _remove_at_edge(object_labels)

    return object_labels


def process(mask, splitting_scalar=None, remove_at_edge=True):
    dx = find_grid_spacing(mask)

    object_labels = label_objects(mask=mask, splitting_scalar=splitting_scalar,
                                  remove_at_edge=remove_at_edge)

    return calc_scales(object_labels=object_labels, dx=dx)


def find_grid_spacing(mask):
    # NB: should also checked for stretched grids..
    xt, yt, zt = mask.xt, mask.yt, mask.zt

    dx_all = np.diff(xt.values)
    dy_all = np.diff(yt.values)
    dx, dy = np.max(dx_all), np.max(dy_all)

    if not 'zt' in mask.coords:
        warnings.warn("zt hasn't got any coordinates defined, assuming dz=dx")
        dz_all = np.diff(zt.values)
        dz = np.max(dx)

    if not dx == dy:
        raise NotImplementedError("Only isotropic grids are supported")

    return dx


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(__doc__)

    argparser.add_argument('base_name', type=str)
    argparser.add_argument('mask_name', type=str)
    argparser.add_argument('tn', type=int)
    argparser.add_argument('--splitting-scalar', type=str, default=None)
    argparser.add_argument('--z_max', type=float, default=None)
    argparser.add_argument('--keep-edge-objects', default=False,
                           action="store_true")

    args = argparser.parse_args()

    input_name = '{}.tn{}'.format(args.base_name, args.tn)

    out_filename = "{}.minkowski_scales.{}.nc".format(
        input_name.replace('/', '__'), args.mask_name
    )

    fn_mask = "{}.{}.mask.nc".format(input_name, args.mask_name)
    if not os.path.exists(fn_mask):
        raise Exception("Couldn't find mask file `{}`".format(fn_mask))
    mask = xr.open_dataarray(fn_mask, decode_times=False)

    if args.splitting_scalar is not None:
        fn_ss = "{}.{}.nc".format(input_name, args.splitting_scalar)
        if not os.path.exists(fn_ss):
            raise Exception("Couldn't find splitting scalar file `{}`"
                            "".format(fn_ss))
        splitting_scalar = xr.open_dataarray(fn_ss, decode_times=False)
    else:
        splitting_scalar = None

    if args.z_max is not None:
        mask = mask.sel(zt=slice(0.0, args.z_max), drop=True).squeeze()
        if splitting_scalar is not None:
            splitting_scalar = splitting_scalar.sel(
                zt=slice(0.0, args.z_max), drop=True
            ).squeeze()

    ds = process(mask=mask, splitting_scalar=splitting_scalar,
                 remove_at_edge=not args.keep_edge_objects)

    ds.attrs['input_name'] = input_name
    ds.attrs['mask_name'] = args.mask_name

    ds.to_netcdf(out_filename)
    print("Wrote output to `{}`".format(out_filename))
