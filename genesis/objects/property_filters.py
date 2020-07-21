import functools

import numpy as np


def _filter_by_percentile(ds, var_name, frac=90.0, part="upper"):
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        if part == "lower":
            q = frac
        elif part == "upper":
            q = 100.0 - frac
        else:
            raise Exception("Invalid percentile part `{}`".format(part))

        v = ds.dropna(dim="object_id")[var_name]
        vlim = np.percentile(v, q=q)

        if part == "lower":
            return ds.where(ds[var_name] < vlim, drop=True)
        else:
            return ds.where(ds[var_name] > vlim, drop=True)


def make_filter_fn(prop_name, op_name, s_value):

    if op_name in ["upper_percentile", "lower_percentile"]:
        value = float(s_value)
        part, _ = op_name.split("_")
        fn = functools.partial(
            _filter_by_percentile, frac=value, part=part, var_name=prop_name
        )
    if op_name == "isnan":
        if s_value == "True":
            op_fn = np.isnan
        elif s_value == "False":
            op_fn = lambda v: np.logical_not(np.isnan(v))  # noqa
        else:
            raise NotImplementedError(s_value)

        fn = lambda da: da.where(op_fn(da[prop_name]), drop=True)  # noqa
    else:
        op = dict(
            lt="less_than",
            gt="greater_than",
            eq="equal",
            lte="less_equal",
            gte="greater_equal",
            isnan="isnan",
        )[op_name]
        op_fn = getattr(np, op.replace("_than", ""))
        value = float(s_value)

        fn = lambda da: da.where(op_fn(getattr(da, prop_name), value), drop=True)  # noqa

    return fn


PROP_NAME_MAPPING = dict(
    num_cells="N_c",
    z_max="z_{max}",
    z_min="z_{min}",
    qv_flux="w'q'",
    cvrxp_p_stddivs=r"\sigma(\phi)",
    qc="q_c",
    r_equiv="r_{equiv}",
)


def _get_prop_name_in_latex(s):
    latex = PROP_NAME_MAPPING.get(s)
    if latex is None:
        if "__" in s:
            prop, op = s.split("__")
            if op == "volume_integral":
                latex = r"\int_V {}".format(_get_prop_name_in_latex(prop))
            elif op == "maximum_pos_z":
                latex = r"z(MAX_{{{}}})".format(_get_prop_name_in_latex(prop))
            elif op == "maximum":
                latex = r"{}^{{max}}".format(_get_prop_name_in_latex(prop))
            else:
                raise NotImplementedError(prop, op)
        else:
            latex = s
    return latex


OP_NAME_MAPPING = dict(lt="<", eq="=", gt=">", lte=r"\leq", gte=r"\geq")

PROP_NAME_UNITS_MAPPING = dict(z_min="m", z_max="m", r_equiv="m",)


def _format_op_for_latex(prop_latex, op_name, s_value, units=""):
    op_latex = OP_NAME_MAPPING.get(op_name)
    if op_latex is not None:
        s = r"${prop}{op}{val}{units}$".format(
            prop=prop_latex,
            op=OP_NAME_MAPPING.get(op_name, op_name),
            val=s_value,
            units=units,
        )
    elif op_name.endswith("_percentile"):
        part, _ = op_name.split("_")
        s = r"{part} {val}% of ${prop}$ dist.".format(
            prop=prop_latex, part=part, val=s_value
        )
    else:
        raise NotImplementedError(op_name)

    return s


def latex_format(f_type, f_def):
    if f_type == "prop":
        prop_name, op_name, s_value = f_def
        prop_latex = _get_prop_name_in_latex(prop_name)

        units = PROP_NAME_UNITS_MAPPING.get(prop_name, "")
        if units == "m":
            s_value = int(float(s_value))
        else:
            units = ""
        return _format_op_for_latex(prop_latex, op_name, s_value, units)
