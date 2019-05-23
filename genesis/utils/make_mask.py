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

OUT_FILENAME_FORMAT = "{base_name}.mask.{mask_name}.nc"

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


class MissingInputException(Exception):
    def __init__(self, missing_kwargs, *args, **kwargs):
        self.missing_kwargs = missing_kwargs

def make_mask_name(method, method_kwargs):
    method_kwargs = build_method_kwargs(method=method, kwargs=method_kwargs)

    def include_kw_in_name(v):
        if v == 'base_name':
            return False
        else:
            return type(method_kwargs[v]) in [float, int, str]

    name = ".".join(
        [method,] + 
        ["{v}{val}".format(v=v, val=method_kwargs[v]) 
            for v in method_kwargs.keys() if include_kw_in_name(v)
        ]
    )

    return name

def build_method_kwargs(method, kwargs):
    """
    Use the provided arguments together with default ones and check which are
    still missing to satisfy the function signature
    """
    fn = getattr(mask_functions, method)
    fn_argspec = inspect.getargspec(fn)
    needed_vars = fn_argspec.args
    if not fn_argspec.defaults is None:
        default_values = dict(
            zip(fn_argspec.args[-len(fn_argspec.defaults):],fn_argspec.defaults)
        )
    else:
        default_values = {}

    missing_kwargs = []

    # we iterator over the functions required arguments and check if they've
    # been passed in
    for v in needed_vars:
        if v in default_values:
            if v in kwargs:
                # attempt to type cast to the correct type
                val = type(default_values.get(v))(kwargs[v])
                kwargs[v] = val
            else:
                print("Using default value `{}` for argument `{}`".format(
                    default_values.get(v), v
                ))
                kwargs[v] = default_values.get(v)
        else:
            if not v in kwargs:
                missing_kwargs.append(v)

    if len(missing_kwargs) > 0:
        raise MissingInputException(missing_kwargs)
    else:
        return kwargs

def main(method, method_kwargs):
    fn = getattr(mask_functions, method)

    method_kwargs = build_method_kwargs(method=method, kwargs=method_kwargs)

    mask = fn(**method_kwargs).squeeze()

    if hasattr(fn, "description"):
        mask.attrs['long_name'] = fn.description.format(**method_kwargs)

    mask.name = make_mask_name(method, method_kwargs)

    return mask


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

    kwargs = dict(args.extra)

    try:
        kwargs = build_method_kwargs(method=args.fn, kwargs=kwargs)
    except MissingInputException as e:
        missing_kwargs = e.missing_kwargs

        for v in missing_kwargs:
            if v == "ds_profile":
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

    mask = main(method=args.fn, method_kwargs=kwargs)


    out_filename = OUT_FILENAME_FORMAT.format(base_name=args.base_name,
                                              mask_name=mask.name)

    mask.to_netcdf(out_filename)

    print("Wrote mask to `{}`".format(out_filename))
