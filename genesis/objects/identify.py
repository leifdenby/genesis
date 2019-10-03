"""
Utilities for labelling 3D objects from a mask
"""
import os
import warnings

import xarray as xr
import numpy as np

import cloud_identification
from ..utils import find_grid_spacing

OUT_FILENAME_FORMAT = "{base_name}.objects.{objects_name}.nc"

def make_objects_name(mask_name, splitting_var):
    return "{mask_name}.split_on.{splitting_var}".format(**locals())

def label_objects(mask, splitting_scalar, remove_at_edge=False):
    if remove_at_edge:
        raise NotImplementedError
        def _remove_at_edge(object_labels):
            mask_edge = np.zeros_like(mask)
            mask_edge[:,:,1] = True  # has to be k==1 because object identification codes treats actual edges as ghost cells
            mask_edge[:,:,-2] = True
            cloud_identification.remove_intersecting(object_labels, mask_edge)

    if mask.shape != splitting_scalar.shape:
        raise Exception("Incompatible shapes of splitting scalar ({}) and "
                        "mask ({})".format(splitting_scalar.shape, mask.shape))
    assert mask.dims == splitting_scalar.dims
    for d in mask.dims:
        assert np.all(mask[d].values == splitting_scalar[d].values)


    object_labels = cloud_identification.number_objects(
        splitting_scalar.values, mask=mask.values
    )

    # if remove_at_edge:
        # _remove_at_edge(object_labels)

    return object_labels

def process(mask, splitting_scalar):
    dx = find_grid_spacing(mask)

    if splitting_scalar is not None:
        mask = mask.sel(zt=splitting_scalar.zt).squeeze()

    object_labels = label_objects(mask=mask, splitting_scalar=splitting_scalar)

    da = xr.DataArray(data=object_labels, coords=mask.coords, dims=mask.dims,
                      name="object_labels")

    da.name = make_objects_name(
        mask_name=mask.name, splitting_var=splitting_scalar.name
    )
    da.attrs['mask_name'] = mask.name
    da.attrs['splitting_scalar'] = splitting_scalar.name

    return da

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(__doc__)

    argparser.add_argument('base_name', type=str)
    argparser.add_argument('mask_name', type=str)
    argparser.add_argument('splitting_scalar')
    argparser.add_argument('--z_max', type=float, default=np.inf)
    argparser.add_argument('--remove-edge-objects', default=False,
                           action="store_true")

    args = argparser.parse_args()

    input_name = args.base_name

    fn_mask = "{}.mask_3d.{}.nc".format(input_name, args.mask_name)
    fn_mask_2d = "{}.mask.{}.nc".format(input_name, args.mask_name)
    if os.path.exists(fn_mask):
        pass
    elif os.path.exists(fn_mask_2d):
        fn_mask = fn_mask_2d
        print("Using 2D xy mask")
    else:
        raise Exception("Couldn't find mask file `{}` or `{}`"
                        "".format(fn_mask, fn_mask_2d))
    mask = xr.open_dataarray(fn_mask, decode_times=False)

    if args.splitting_scalar is not None:
        fn_ss = "{}.{}.nc".format(input_name, args.splitting_scalar)
        if not os.path.exists(fn_ss):
            raise Exception("Couldn't find splitting scalar file `{}`"
                            "".format(fn_ss))
        splitting_scalar = xr.open_dataarray(
            fn_ss, decode_times=False
        ).squeeze()
    else:
        splitting_scalar = None

    if args.z_max is not np.inf:
        mask = mask.sel(zt=slice(0.0, args.z_max)).squeeze()
        if splitting_scalar is not None:
            splitting_scalar = splitting_scalar.sel(
                zt=slice(0.0, args.z_max)
            ).squeeze()

    ds = process(mask=mask, splitting_scalar=splitting_scalar,
                 remove_at_edge=args.remove_edge_objects)

    ds.attrs['input_name'] = input_name
    ds.attrs['mask_name'] = args.mask_name
    ds.attrs['z_max'] = args.z_max
    ds.attrs['splitting_scalar'] = args.splitting_scalar

    out_filename = OUT_FILENAME_FORMAT.format(
        base_name=input_name.replace('/', '__'),
        objects_name=make_objects_name(
            mask_name=args.mask_name, splitting_var=args.splitting_scalar
        )
    ).replace('__masks', '')

    ds.to_netcdf(out_filename)
    print("Wrote output to `{}`".format(out_filename))
