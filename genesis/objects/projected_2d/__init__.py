import xarray as xr
import operator
import numpy as np
import inspect


from . import operations
from .operations import CloudType


def _find_objectset_operations():
    from inspect import getmembers, isfunction, getargspec
    f = lambda o: isfunction(o[1]) and 'ds' in getargspec(o[1]).args
    return [o[0] for o in getmembers(operations) if f(o)]

class ObjectSet():
    def __init__(self, ds, parent=None, ds_masks=None):
        ds_local = ds.copy()
        self.parent = parent

        vars_to_keep = list(filter(
            lambda v: v.startswith('smcloud'),
            list(ds.data_vars)
        ))
        self.ds_props = ds_local[vars_to_keep]

        if ds_masks is None:
            vars_to_keep = list(filter(
                lambda v: v.startswith('nr'),
                list(ds.data_vars)
            ))
            self.ds_masks = ds_local[vars_to_keep]
        else:
            self.ds_masks = ds_masks


    def get_value(self, function_name, as_xarray=False, only_present=False, **kwargs):
        """
        Using function `function_name` compute a value for all clouds in the
        cloudset and return these values
        """
        fn = getattr(operations, function_name, None)
        if fn is None:
            available_operations = _find_objectset_operations()
            raise NotImplementedError("Couldn't find cloud operation `{}`,"
                    " available operations: {}"
                    "".format(function_name, ", ".join(available_operations)))

        fn_argspec = inspect.getargspec(fn)
        needed_vars = fn_argspec.args

        if "da_nrcloud" in needed_vars:
            kwargs['da_nrcloud'] = self.ds_masks.nrcloud

        import ipdb
        with ipdb.launch_ipdb_on_exception():
            return fn(ds=self.ds_props, **kwargs).squeeze()

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
                if k.startswith('_'):
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
        if '__' in function_name:
            function_name, comp_op = function_name.split('__')
        else:
            comp_op = 'eq'

        value = self.get_value(function_name, **kwargs)
        cloud_ids = self.get_value('cloud_id')
        if hasattr(operator, comp_op):
            op = getattr(operator, comp_op)
            r = op(value, ref_value)
        elif comp_op == 'isnan':
            r = np.isnan(value) == ref_value
        elif comp_op == 'in':
            r = np.zeros(cloud_ids.shape)
            for r_value in ref_value:
                r = np.logical_or(r, r_value == value)
        elif comp_op == 'between':
            r = np.zeros(cloud_ids.shape)
            if not type(ref_value) == list:
                ref_value = [ref_value,]
            for r_value_lower, r_value_upper in ref_value:
                r = np.logical_or(r, np.logical_and(r_value_lower < value, value < r_value_upper))
        else:
            raise NotImplementedError()

        # XXX: xarray messes up indexing by boolean array, recast to bare
        # np.array
        if exclude:
            new_cloud_ids = cloud_ids[np.logical_not(np.array(r))]
        else:
            new_cloud_ids = cloud_ids[np.array(r)]

        ds_new_props = self.ds_props.sel(smcloudid=cloud_ids)

        return ObjectSet(
            ds=ds_new_props,
            ds_masks=self.ds_masks,
            parent=self,
        )
