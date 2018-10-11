"""
Create mask files which can be used elsewhere
"""
import inspect
import os
import argparse

import numpy as np
import xarray as xr

from . import mask_functions

# register a progressbar so we can see progress of dask'ed operations with xarray
from dask.diagnostics import ProgressBar
ProgressBar().register()

class StoreDictKeyPair(argparse.Action):
    """
    Custom parser so that we can provide extra values to mask functions
    https://stackoverflow.com/a/42355279
    """
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
            setattr(namespace, self.dest, my_dict)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument('base_name', type=str)
    mask_function_names = [
        o[0] for o in inspect.getmembers(mask_functions)
        if inspect.isfunction(o[1]) and not o[0].startswith('_')
    ]

    argparser.add_argument('fn', choices=list(mask_function_names))
    argparser.add_argument('--extra', action=StoreDictKeyPair, default={})

    args = argparser.parse_args()

    fn = getattr(mask_functions, args.fn)

    fn_argspec = inspect.getargspec(fn)
    needed_vars = fn_argspec.args
    if not fn_argspec.defaults is None:
        default_values = dict(
            zip(fn_argspec.args[-len(fn_argspec.defaults):],fn_argspec.defaults)
        )
    else:
        default_values = {}

    kwargs = {}

    extra_args_str = ''

    for v in needed_vars:
        if v in default_values:
            if v in args.extra:
                # attempt to type cast to the correct type
                val = type(default_values.get(v))(args.extra[v])
                kwargs[v] = val
                extra_args_str += "{v}{val}".format(v=v, val=val)
            else:
                print("Using default value `{}` for argument `{}`".format(
                    default_values.get(v), v
                ))
                kwargs[v] = default_values.get(v)
        elif v == "ds_profile":
            case_name = args.base_name.split('.')[0]
            filename = "{}.ps.nc".format(case_name)
            if not os.path.exists(filename):
                raise Exception("Could not find profile file, looked in "
                                "`{}`".format(filename))
            kwargs[v] = xr.open_dataset(filename, decode_times=False,
                                        chunks=dict(time=1))
        elif v == "base_name":
            kwargs["base_name"] = args.base_name
        else:
            filename = "{}.{}.nc".format(args.base_name, v)
            if not os.path.exists(filename):
                raise Exception("Can't find required var `{}` for mask "
                                "function `{}`, `{}`".format(v, args.fn, filename))

            try:
                kwargs[v] = xr.open_dataarray(filename, decode_times=False,
                                              chunks=dict(zt=10))
            except ValueError:
                kwargs[v] = xr.open_dataarray(filename, decode_times=False)

    mask = fn(**kwargs).squeeze()
    if isinstance(mask, xr.DataArray):
        mask.name = args.fn

    if hasattr(fn, "description"):
        mask.attrs['longname'] = fn.description.format(**kwargs)

    out_filename = "{}.mask.{}.nc".format(args.base_name, args.fn)
    if len(extra_args_str) > 0:
        out_filename = out_filename.replace('.nc', '.{}.nc'.format(extra_args_str))
        mask.name = '{}.{}'.format(mask.name, extra_args_str)

    if len(filter(lambda d: d != "time", mask.dims)) == 3:
        out_filename = out_filename.replace('.mask.', '.mask_3d.')

    mask.to_netcdf(out_filename)

    print("Wrote mask to `{}`".format(out_filename))
