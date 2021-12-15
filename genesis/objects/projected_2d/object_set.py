import inspect
import operator
import warnings

import numpy as np
import xarray as xr

from . import aggregation, operations


def _find_module_operations(module):
    from inspect import getargspec, getmembers, isfunction

    f = lambda o: isfunction(o[1]) and "ds" in getargspec(o[1]).args  # noqa
    return [o[0] for o in getmembers(module) if f(o)]


def _find_objectset_operations():
    return _find_module_operations(operations)


def _find_objectset_aggregations():
    return _find_module_operations(aggregation)


class ObjectSet:
    def __init__(self, ds, object_type="cloud", parent=None):
        ds_local = ds.copy()
        self.parent = parent
        self.object_type = object_type

        if parent is None:
            vars_to_keep = list(
                filter(lambda v: v.startswith(f"sm{object_type}"), list(ds.data_vars))
            )
            self.ds = ds_local[vars_to_keep]
        else:
            self.ds = ds

        if f"sm{object_type}" in self.ds.coords:
            self.ds = self.ds.swap_dims({f"sm{object_type}": f"sm{object_type}id"})

        # rename the variables to something more useful
        for v in list(self.ds.data_vars) + list(self.ds.coords):
            new_name = v.replace(f"sm{object_type}", "")
            if new_name == "id":
                new_name = "object_id"
            elif new_name == "t":
                new_name = "object_time"
            elif new_name == "type":
                new_name = "object_type"
            elif new_name == "dur":
                new_name = "duration"
            elif v == f"sm{object_type}":
                del self.ds[v]
                continue
            self.ds = self.ds.rename({v: new_name})

        # fix units. The cloud-tracking code doesn't check that the units are
        # consistent between the input files and so we can end up with the
        # units string referring to fractions of days but actually the values
        # are in seconds...
        for v in list(self.ds.data_vars) + list(self.ds.coords):
            if "units" in self.ds[v].attrs:
                units = self.ds[v].units
                val0 = self.ds[v].values[0]
                if units == "day as %Y%m%d.%f" and np.allclose(val0, int(val0)):
                    warnings.warn(
                        "The the time units are given as `day as %Y%m%d.%f`,"
                        " but the first value is an integer value so I'm going"
                        " to assume the units are actually in seconds"
                    )
                    self.ds[v].attrs["units"] = "seconds since 2000-01-01T00:00:00"
                    self.ds[v].attrs["calendar"] = "proleptic_gregorian"
                    self.ds = xr.decode_cf(self.ds)

    def get_value(self, function_name, as_xarray=False, only_present=False, **kwargs):
        """
        Using function `function_name` compute a value for all clouds in the
        cloudset and return these values
        """
        fn = getattr(operations, function_name, None)
        if fn is None:
            available_operations = _find_objectset_operations()
            raise NotImplementedError(
                "Couldn't find cloud operation `{}`,"
                " available operations: {}"
                "".format(function_name, ", ".join(available_operations))
            )

        fn_argspec = inspect.getargspec(fn)
        needed_vars = fn_argspec.args

        return fn(ds=self.ds, **kwargs).squeeze()

    def filter(self, kwargs={}, exclude=False, **filter_fns):
        """
        .filter(area=100)
        .filter(area__lt=200)
        .filter(is_active=True)
        """
        kwargs = dict(kwargs)

        if len(filter_fns) > 1:
            m = 0
            for k, v in list(filter_fns.items()):
                if k.startswith("_"):
                    kwargs[k[1:]] = v
                else:
                    m += 1
                    if m > 1:
                        raise NotImplementedError()
                    else:
                        function_name = k
                        ref_value = v
        else:
            function_name = list(filter_fns.keys())[0]
            ref_value = list(filter_fns.values())[0]

        comp_op = None
        if "__" in function_name:
            function_name, comp_op = function_name.split("__")
        else:
            comp_op = "eq"

        value = self.get_value(function_name, **kwargs)
        object_ids = self.ds.object_id
        if hasattr(operator, comp_op):
            op = getattr(operator, comp_op)
            r = op(value, ref_value)
        elif comp_op == "isnan":
            r = np.isnan(value) == ref_value
        elif comp_op == "in":
            r = np.zeros(object_ids.shape)
            for r_value in ref_value:
                r = np.logical_or(r, r_value == value)
        elif comp_op == "between":
            r = np.zeros(object_ids.shape)
            if not type(ref_value) == list:
                ref_value = [
                    ref_value,
                ]
            for r_value_lower, r_value_upper in ref_value:
                r = np.logical_or(
                    r, np.logical_and(r_value_lower < value, value < r_value_upper)
                )
        else:
            raise NotImplementedError()

        # XXX: xarray messes up indexing by boolean array, recast to bare
        # np.array
        if exclude:
            object_ids = object_ids[np.logical_not(np.array(r))]
        else:
            object_ids = object_ids[np.array(r)]

        ds_new = self.ds.sel(object_id=object_ids)

        return ObjectSet(ds=ds_new, parent=self, object_type=self.object_type)

    def __len__(self):
        return int(self.ds.object_id.count())
