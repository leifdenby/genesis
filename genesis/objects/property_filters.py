import functools

import numpy as np

def _filter_by_percentile(ds, var_name, frac=90., part="upper"):
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        if part == "lower":
            q = frac
        elif part == "upper":
            q = 100. - frac
        else:
            raise Exception("Invalid percentile part `{}`".format(part))

        v = ds.dropna(dim='object_id')[var_name]
        vlim = np.percentile(v, q=q)

        if part == "lower":
            return ds.where(ds[var_name] < vlim, drop=True)
        else:
            return ds.where(ds[var_name] > vlim, drop=True)

def _property_filter(f_cond):
    s_prop_and_op, s_value = f_cond.split("=")
    prop_name, op_name = s_prop_and_op.split('__')

    if op_name in ["upper_percentile", "lower_percentile"]:
        value = float(s_value)
        part, _ = op_name.split('_')
        fn = functools.partial(_filter_by_percentile, frac=value, part=part,
                               var_name=prop_name)
    else:
        op = dict(lt="less_than", gt="greater_than", eq="equals")[op_name]
        op_fn = getattr(np, op.replace('_than', ''))
        value = float(s_value)
        fn = lambda da: da.where(op_fn(getattr(da, prop_name), value))

    return prop_name, fn

def parse_defs(filter_defs):
    filters = dict(reqd_props=[], fns=[])
    s_filters = filter_defs.split(',')

    for s_filter in s_filters:
        try:
            f_type, f_cond = s_filter.split(':')
            if f_type == 'prop':
                prop_name, fn = _property_filter(f_cond)
                filters['reqd_props'].append(prop_name)
                filters['fns'].append(fn)
            else:
                raise NotImplementedError("Filter type `{}` not recognised"
                                          "".format(f_type))
        except (IndexError, ValueError) as e:
            raise Exception("Malformed filter definition: `{}`".format(
                            s_filter))
    return filters
