"""
Using 2D mask filter out objects and create a new objects file
"""
import os
import warnings

import xarray as xr
import numpy as np

import cloud_identification


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
    argparser.add_argument('--objects', type=str)
    argparser.add_argument('--mask', type=str)

    args = argparser.parse_args()

    input_name = args.base_name

    out_filename = "{}.objects.{}.nc".format(
        input_name.replace('/', '__'), args.mask_name
    )

    fn_mask = "{}.{}.mask.nc".format(input_name, args.mask)
    if not os.path.exists(fn_mask):
        raise Exception("Couldn't find mask file `{}`".format(fn_mask))
    mask = xr.open_dataarray(fn_mask, decode_times=False)

    fn_objects = "{}.{}.objects.nc".format(input_name, args.objects)
    if not os.path.exists(fn_objects):
        raise Exception("Couldn't find objects file `{}`".format(fn_objects))
    objects = xr.open_dataarray(fn_objects, decode_times=False)

    ds = process(mask=mask, splitting_scalar=splitting_scalar,
                 remove_at_edge=not args.keep_edge_objects)

    ds.attrs['input_name'] = input_name
    ds.attrs['mask_name'] = "{} && {}".format(ds.mask_name, args.mask)

    ds.to_netcdf(out_filename)
    print("Wrote output to `{}`".format(out_filename))
