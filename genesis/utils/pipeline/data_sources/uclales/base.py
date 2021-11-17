import xarray as xr
import numpy as np
import scipy.optimize

from pathlib import Path
import os
from .... import center_staggered_field

from .common import _fix_time_units
from .extraction import Extract3D


FIELD_NAME_MAPPING = dict(
    w="w_zt",
    u="u_xt",
    v="v_yt",
    xt="xt",
    yt="yt",
    zt="zt",
    qt="q",
    qc="l",
    qr="r",
    theta_l="t",
    cvrxp="cvrxp",
    p="p",
    theta_l_v="theta_l_v",
    theta_l_v_hack="theta_l_v_hack",
    qv="qv",
    qv__norain="qv",
    abstemp="abstemp",
)

FIELD_DESCRIPTIONS = dict(
    w="vertical velocity",
    qv="water vapour",
    qc="cloud liquid water",
    theta="potential temperature",
    cvrxp="radioactive tracer",
    theta_l="liquid potential temperature",
)

UNITS_FORMAT = {
    "METERS/SECOND": "m/s",
    "KELVIN": "K",
    "KG/KG": "kg/kg",
}

DERIVED_FIELDS = dict(
    abstemp=("qc", "qr", "theta_l", "p"),
    theta_l_v=("theta_l", "qv", "qc", "qr"),
    theta_l_v_hack=(
        "theta_l",
        "qv",
        "qc",
    ),
    qv=("qt", "qc", "qr"),
    qv__norain=("qt", "qc"),
)

FN_FORMAT_3D = "3d_blocks/full_domain/{experiment_name}.tn{timestep}.{field_name}.nc"


def _get_uclales_field(field_name):
    field_name_src = FIELD_NAME_MAPPING.get(field_name)

    if field_name_src is None:
        raise NotImplementedError(
            "please define a mapping for the field `{}`"
            " in {}".format(field_name, __file__)
        )

    return field_name_src


def _get_uclales_field_description(field_name):
    field_description = FIELD_DESCRIPTIONS.get(field_name)

    if field_description is None:
        raise NotImplementedError(
            "please define a description for the field "
            "`{}` in {}".format(field_name, __file__)
        )

    return field_description


def _scale_field(da):
    modified = False
    if da.units == "km":
        da.values *= 1000.0
        da.attrs["units"] = "m"
        modified = True
    # XXX: there is a bug in UCLALES where the units reported (g/kg) are
    # actually wrong (they are kg/kg), this messes up the density calculation
    # later if we don't fix it
    # qv is already scaled because it has been calculated from qt and qr
    # which were scaled
    elif da.name in ["q", "l", "r"]:
        assert da.max() < 1.0 and da.units == "g/kg"
        da.values *= 1000.0
        da.attrs["scaling"] = "correcting for wrong units in UCLALES output"
        modified = True
    return da, modified


def _cleanup_units(units):
    return UNITS_FORMAT.get(units, units)


def _fix_long_name(da):
    modified = False
    # the CF conventions stipulate that the long name attribute should be named
    # `long_name` not `longname`
    if "longname" in da.attrs and "long_name" not in da.attrs:
        da.attrs["long_name"] = da.longname
        modified = True
    return da, modified


class RawDataPathDoesNotExist(Exception):
    pass


def _build_block_extraction_task(dataset_meta, field_name):
    if field_name in ["u", "v", "w"]:
        var_name = field_name
    else:
        var_name = FIELD_NAME_MAPPING[field_name]

    raw_data_path = Path(dataset_meta["path"]) / "raw_data"

    if not raw_data_path.exists():
        raise RawDataPathDoesNotExist

    task = Extract3D(
        source_path=raw_data_path,
        file_prefix=dataset_meta["experiment_name"],
        var_name=var_name,
        tn=dataset_meta["timestep"],
    )
    return task


def extract_field_to_filename(dataset_meta, path_out, field_name, **kwargs):  # noqa
    field_name_src = _get_uclales_field(field_name)

    fn_format = dataset_meta.get("fn_format", FN_FORMAT_3D)
    path_in = Path(dataset_meta["path"]) / fn_format.format(
        field_name=field_name_src, **dataset_meta
    )

    can_symlink = True

    if field_name == "theta_l_v":
        da = _calc_theta_l_v(**kwargs)
        can_symlink = False
    elif field_name == "theta_l_v_hack":
        da = _calc_theta_l_v_hack(**kwargs)
        can_symlink = False
    elif field_name == "qv":
        da = _calc_qv(**kwargs)
        can_symlink = False
    elif field_name == "qv__norain":
        da = _calc_qv__norain(**kwargs)
        can_symlink = False
    elif field_name == "abstemp":
        da = _calc_temperature(**kwargs)
        can_symlink = False
    else:
        center_field = False
        if field_name_src == "w_zt":
            path_in = path_in.parent / path_in.name.replace(".w_zt.", ".w.")
            center_field = True
        elif field_name_src == "u_xt":
            path_in = path_in.parent / path_in.name.replace(".u_xt.", ".u.")
            center_field = True
        elif field_name_src == "v_yt":
            path_in = path_in.parent / path_in.name.replace(".v_yt.", ".v.")
            center_field = True

        if not path_in.exists():
            try:
                task = _build_block_extraction_task(
                    dataset_meta=dataset_meta,
                    field_name=field_name,
                )
                # if the source file doesn't exist we return a task to create
                # it, next time we pass here the file should exist and we can
                # just open it
                if not task.output().exists():
                    return task

                da = task.output().open(decode_times=False)
                can_symlink = False

            except RawDataPathDoesNotExist:
                raise Exception(
                    f"Can't open `{path_in}` because it doesn't exist. If you have"
                    f" the raw model output you can put it in `{dataset_meta['path']}/raw_data`"
                    " and extraction from blocks will be done automatically"
                )
        else:
            da = xr.open_dataarray(path_in, decode_times=False)

        if center_field:
            da, _ = _fix_long_name(da)
            da = center_staggered_field(da)

    da, modified = _scale_field(da)
    if modified:
        can_symlink = False

    da, modified = _fix_long_name(da)
    if modified:
        can_symlink = False

    if "time" in da.coords:
        da.coords["time"], modified = _fix_time_units(da["time"])
        if modified:
            can_symlink = False

    for c in "xt yt zt".split(" "):
        if "fixes" in dataset_meta:
            if "missing_{}_coordinate".format(c[0]) in dataset_meta["fixes"]:
                can_symlink = False

                dx = dataset_meta["dx"]
                da.coords[c] = -0.5 * dx + dx * np.arange(0, len(da[c]))
                da.coords[c].attrs["units"] = "m"

    if field_name_src != field_name:
        can_symlink = False
        da.name = field_name

    if can_symlink and path_in.exists():
        os.symlink(str(path_in.absolute()), str(path_out))
    else:
        da.to_netcdf(path_out)


def _calc_theta_l_v(theta_l, qv, qc, qr):
    assert qv.units.lower() == "g/kg" and qv.max() > 1.0
    assert theta_l.units == "K"

    # qc here refers to q "cloud", the total condensate is qc+qr (here
    # forgetting about ice...)
    eps = 0.608
    theta_l_v = theta_l * (1.0 + 1.0e-3 * (eps * qv - (qc + qr)))

    theta_l_v.attrs["units"] = "K"
    theta_l_v.attrs["long_name"] = "virtual liquid potential temperature"
    theta_l_v.name = "theta_l_v"
    return theta_l_v


def _calc_theta_l_v_hack(theta_l, qv, qc):
    assert qv.units.lower() == "g/kg" and qv.max() > 1.0
    assert theta_l.units == "K"

    # qc here refers to q "cloud", the total condensate is qc+qr (here
    # forgetting about ice...)
    eps = 0.608
    theta_l_v = theta_l * (1.0 + 1.0e-3 * (eps * qv - (qc)))

    theta_l_v.attrs["units"] = "K"
    theta_l_v.attrs["long_name"] = "virtual liquid potential temperature"
    theta_l_v.name = "theta_l_v_hack"
    return theta_l_v


def _calc_qv(qt, qc, qr):
    assert qt.units.lower() == "g/kg" and qt.max() > 1.0

    qv = qt - qc - qr
    qv.attrs["units"] = "g/kg"
    qv.attrs["long_name"] = "water vapour mixing ratio"
    qv.name = "qv"
    return qv


def _calc_qv__norain(qt, qc):
    assert qt.units.lower() == "g/kg" and qt.max() > 1.0

    qv = qt - qc
    qv.attrs["units"] = "g/kg"
    qv.attrs["long_name"] = "water vapour mixing ratio (assumed zero rain)"
    qv.name = "qv"
    return qv


@np.vectorize
def _calc_temperature_single(q_l, p, theta_l):
    # constants from UCLALES
    cp_d = 1.004 * 1.0e3  # [J/kg/K]
    R_d = 287.04  # [J/kg/K]
    L_v = 2.5 * 1.0e6  # [J/kg]
    p_theta = 1.0e5

    # XXX: this is *not* the *actual* liquid potential temperature (as
    # given in B. Steven's notes on moist thermodynamics), but instead
    # reflects the form used in UCLALES where in place of the mixture
    # heat-capacity the dry-air heat capacity is used
    def temp_func(T):
        return theta_l - T * (p_theta / p) ** (R_d / cp_d) * np.exp(
            -L_v * q_l / (cp_d * T)
        )

    if np.all(q_l == 0.0):
        # no need for root finding
        return theta_l / ((p_theta / p) ** (R_d / cp_d))

    # XXX: brentq solver requires bounds, I don't expect we'll get below -100C
    T_min = -100.0 + 273.0
    T_max = 50.0 + 273.0
    T = scipy.optimize.brentq(f=temp_func, a=T_min, b=T_max)

    # check that we're within 1.0e-4
    assert np.all(np.abs(temp_func(T)) < 1.0e-4)

    return T


def _calc_temperature(qc, qr, p, theta_l):
    q_l = qc + qr
    arr_temperature = np.vectorize(_calc_temperature_single)(
        q_l=q_l, p=p, theta_l=theta_l
    )

    da_temperature = xr.DataArray(
        arr_temperature,
        dims=p.dims,
        attrs=dict(longname="temperature", units="K"),
    )

    return da_temperature
