import os

import xarray as xr
import numpy as np
from scipy import ndimage


def _estimate_dx(da):
    dx = np.max(np.diff(da.xt))
    dy = np.max(np.diff(da.yt))
    dz = np.max(np.diff(da.zt))

    assert dx == dy == dz

    return dx


def integrate(objects, da):
    if 'object_ids' in da:
        object_ids = da.object_ids
    else:
        object_ids = np.unique(objects)
        # ensure object 0 (outside objects) is excluded
        if object_ids[0] == 0:
            object_ids = object_ids[1:]

    assert objects.dims == da.dims
    assert objects.shape == da.shape

    dx = _estimate_dx(da=da)
    vals = ndimage.sum(da, labels=objects, index=object_ids)*dx**3.

    longname = "per-object integral of {}".format(da.name)
    units = "{} m^3".format(da.units)
    da = xr.DataArray(vals, coords=dict(object_id=object_ids),
                      dims=('object_id',),
                      attrs=dict(longname=longname, units=units))

    return da


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument('object_file')
    argparser.add_argument('scalar_field')

    args = argparser.parse_args()
    object_file = args.object_file

    if not 'objects' in object_file:
        raise Exception()

    base_name, objects_mask = object_file.split('.objects.')

    fn_scalar = "{}.{}.nc".format(base_name, args.scalar_field)
    if not os.path.exists(fn_scalar):
        raise Exception("Couldn't find scalar file `{}`".format(fn_scalar))

    scalar_field = args.scalar_field

    da_scalar = xr.open_dataarray(fn_scalar, decode_times=False).squeeze()

    fn_objects = "{}.nc".format(object_file)
    if not os.path.exists(fn_objects):
        raise Exception("Couldn't find objects file `{}`".format(fn_objects))
    objects = xr.open_dataarray(fn_objects, decode_times=False).squeeze()

    out_filename = "{}.objects.{}.{}.nc".format(
        base_name.replace('/', '__'), 'integral', scalar_field
    )

    if objects.zt.max() < da_scalar.zt.max():
        zt_ = da_scalar.zt.values
        da_scalar = da_scalar.sel(zt=slice(None, zt_[25]))

    da = integrate(objects=objects, da=da_scalar)

    da.to_netcdf(out_filename)
    print("Wrote output to `{}`".format(out_filename))
