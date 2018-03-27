"""
Create mask files which can be used elsewhere
"""
import inspect
import os

import numpy as np
import xarray as xr

import mask_functions


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument('base_name', type=str)
    argparser.add_argument('fn', choices=mask_functions.__dict__.keys())

    args = argparser.parse_args()

    fn = getattr(mask_functions, args.fn)

    needed_vars, _, _, _ = inspect.getargspec(fn)

    kwargs = {}

    for v in needed_vars:
        filename = "{}.{}.nc".format(args.base_name, v)
        if not os.path.exists(filename):
            raise Exception("Can't find required var `{}` for mask "
                            "function `{}`".format(v, args.fn))

        kwargs[v] = xr.open_dataarray(filename, decode_times=False,
                                      chunks=dict(zt=10))

    mask = fn(**kwargs)
    mask.name = "mask"

    if hasattr(fn, "description"):
        mask.attrs['longname'] = fn.description

    out_filename = "{}.{}.mask.nc".format(args.base_name, args.fn)

    mask.to_netcdf(out_filename)

    print("Wrote mask to `{}`".format(out_filename))
