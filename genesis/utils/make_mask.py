"""
Create mask files which can be used elsewhere
"""
import inspect
import os

import numpy as np
import xarray as xr

from . import mask_functions

# register a progressbar so we can see progress of dask'ed operations with xarray
from dask.diagnostics import ProgressBar
ProgressBar().register()


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument('base_name', type=str)
    mask_function_names = [
        o[0] for o in inspect.getmembers(mask_functions)
        if inspect.isfunction(o[1]) and not o[0].startswith('_')
    ]

    argparser.add_argument('fn', choices=list(mask_function_names))

    args = argparser.parse_args()

    fn = getattr(mask_functions, args.fn)

    fn_argspec = inspect.getargspec(fn)
    needed_vars = fn_argspec.args
    default_values = dict(
        zip(fn_argspec.args[-len(fn_argspec.defaults):],fn_argspec.defaults)
    )

    kwargs = {}

    for v in needed_vars:
        if v in default_values:
            print("Using default value `{}` for argument `{}`".format(
                default_values.get(v), v
            ))
            kwargs[v] = default_values.get(v)
        elif v == "aux_filename_base":
            kwargs[v] = "{}.{}.mask.aux.nc".format(args.base_name, "{}")
        elif v == "ds_profile":
            case_name = args.base_name.split('.')[0]
            filename = "{}.ps.nc".format(case_name)
            if not os.path.exists(filename):
                raise Exception("Could not find profile file, looked in "
                                "`{}`".format(filename))
            kwargs[v] = xr.open_dataset(filename, decode_times=False,
                                        chunks=dict(time=1))
        else:
            filename = "{}.{}.nc".format(args.base_name, v)
            if not os.path.exists(filename):
                raise Exception("Can't find required var `{}` for mask "
                                "function `{}`, `{}`".format(v, args.fn, filename))

            try:
                kwargs[v] = xr.open_dataarray(filename, decode_times=False,
                                              chunks=dict(zt=1))
            except ValueError:
                kwargs[v] = xr.open_dataarray(filename, decode_times=False)

    mask = fn(**kwargs)
    mask.name = args.fn

    if hasattr(fn, "description"):
        mask.attrs['longname'] = fn.description

    out_filename = "{}.{}.mask.nc".format(args.base_name, args.fn)

    mask.to_netcdf(out_filename)

    print("Wrote mask to `{}`".format(out_filename))
