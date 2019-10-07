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

def _make_filter_fn(prop_name, op_name, s_value):

    if op_name in ["upper_percentile", "lower_percentile"]:
        value = float(s_value)
        part, _ = op_name.split('_')
        fn = functools.partial(_filter_by_percentile, frac=value, part=part,
                               var_name=prop_name)
    else:
        op = dict(lt="less_than", gt="greater_than", eq="equal")[op_name]
        op_fn = getattr(np, op.replace('_than', ''))
        value = float(s_value)
        fn = lambda da: da.where(op_fn(getattr(da, prop_name), value))

    return fn

def _defs_iterator(filter_defs):
    s_filters = filter_defs.split(',')
    for s_filter in s_filters:
        try:
            f_type, f_cond = s_filter.split(':')
            s_prop_and_op, s_value = f_cond.split("=")

            if f_type == 'prop':
                i = s_prop_and_op.rfind('__')
                prop_name, op_name = s_prop_and_op[:i], s_prop_and_op[i+2:]
                yield f_type, (prop_name, op_name, s_value)
            else:
                raise NotImplementedError("Filter type `{}` not recognised"
                                          "".format(f_type))
        except (IndexError, ValueError) as e:
            raise Exception("Malformed filter definition: `{}`".format(
                            s_filter))

def parse_defs(filter_defs):
    filters = dict(reqd_props=[], fns=[])

    for (f_type, f_def) in _defs_iterator(filter_defs):
        if f_type == 'prop':
            prop_name, op_name, s_value = f_def
            fn = _make_filter_fn(prop_name, op_name, s_value)
            filters['reqd_props'].append(prop_name)
            filters['fns'].append(fn)
        else:
            raise NotImplementedError

    return filters

PROP_NAME_MAPPING = dict(
    num_cells="N_c",
    z_max="z_{max}",
    z_min="z_{min}",
    qv_flux="w'q'",
    cvrxp_p_stddivs="\sigma(\phi)",
    qc='q_c',
)

def _get_prop_name_in_latex(s):
    latex = PROP_NAME_MAPPING.get(s)
    if latex is None:
        if '__' in s:
            prop, op = s.split('__')
            if op == 'volume_integral':
                latex = r"\int_V {}".format(_get_prop_name_in_latex(prop))
            elif op == 'maximum_pos_z':
                latex = r"z(MAX_{{{}}})".format(_get_prop_name_in_latex(prop))
            elif op == 'maximum':
                latex = r"{}^{{max}}".format(_get_prop_name_in_latex(prop))
            else:
                raise NotImplementedError(prop, op)
        else:
            latex = s
    return latex


OP_NAME_MAPPING=dict(
    lt="<",eq="=",gt=">"
)

def _format_op_for_latex(prop_latex, op_name, s_value):
    op_latex = OP_NAME_MAPPING.get(op_name)
    if op_latex is not None:
        # TODO: implement passing in units
        s = r"${prop}{op}{val}$".format(
            prop=prop_latex,
            op=OP_NAME_MAPPING.get(op_name, op_name),
            val=s_value
        )
    elif op_name.endswith('_percentile'):
        part, _ = op_name.split('_')
        s = r"{part} {val}% of ${prop}$ dist.".format(
            prop=prop_latex, part=part, val=s_value
        )
    else:
        raise NotImplementedError(op_name)

    return s

def latex_format(filter_defs):
    if filter_defs is None:
        return ""
    s_latex = []
    s_filters = filter_defs.split(',')

    for (f_type, f_def) in _defs_iterator(filter_defs):
        if f_type == 'prop':
            prop_name, op_name, s_value = f_def
            prop_latex=_get_prop_name_in_latex(prop_name)
            s_latex.append(_format_op_for_latex(prop_latex, op_name, s_value))
        else:
            raise NotImplementedError(f_type, f_def)
    return " and ".join(s_latex)
