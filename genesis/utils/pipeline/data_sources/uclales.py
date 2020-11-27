import xarray as xr
import numpy as np

from pathlib import Path
import os
from ...calc_flux import z_center_field

FIELD_NAME_MAPPING = dict(
    w="w_zt",
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
    T=("qc", "theta_l", "p"),
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
    elif da.name in ['q', 'l', 'r']:
        assert da.max() < 1.0 and da.units == "g/kg"
        da.values *= 1000.0
        da.attrs['scaling'] = "correcting for wrong units in UCLALES output"
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


def _fix_time_units(da):
    modified = False
    if np.issubdtype(da.dtype, np.datetime64):
        # already converted since xarray has managed to parse the time in
        # CF-format
        pass
    elif da.attrs["units"].startswith("seconds since 2000-01-01"):
        # I fixed UCLALES to CF valid output, this is output from a fixed
        # version
        pass
    elif da.attrs["units"].startswith("seconds since 2000-00-00"):
        da.attrs["units"] = da.attrs["units"].replace(
            "seconds since 2000-00-00",
            "seconds since 2000-01-01",
        )
        modified = True
    elif da.attrs["units"].startswith("seconds since 0-00-00"):
        # 2D fields have strange time units...
        da.attrs["units"] = da.attrs["units"].replace(
            "seconds since 0-00-00",
            "seconds since 2000-01-01",
        )
        modified = True
    elif da.attrs["units"].startswith("seconds since 0-0-0"):
        # 2D fields have strange time units...
        da.attrs["units"] = da.attrs["units"].replace(
            "seconds since 0-0-0",
            "seconds since 2000-01-01",
        )
        modified = True
    elif da.attrs["units"] == "day as %Y%m%d.%f":
        da = (da * 24 * 60 * 60).astype(int)
        da.attrs["units"] = "seconds since 2000-01-01 00:00:00"
        modified = True
    else:
        raise NotImplementedError(da.attrs["units"])
    return da, modified


def extract_field_to_filename(dataset_meta, path_out, field_name, **kwargs):  # noqa
    field_name_src = _get_uclales_field(field_name)

    fn_format = dataset_meta.get("fn_format", FN_FORMAT_3D)
    path_in = Path(dataset_meta["path"]) / fn_format.format(
        field_name=field_name_src, **dataset_meta
    )

    can_symlink = True

    if field_name_src == "w_zt":
        path_in = path_in.parent / path_in.name.replace(".w_zt.", ".w.")
        da_w_orig = xr.open_dataarray(path_in, decode_times=False)
        da_w_orig, _ = _fix_long_name(da_w_orig)
        da = z_center_field(da_w_orig)
        can_symlink = False
    elif field_name == "theta_l_v":
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
    else:
        if not path_in.exists():
            raise Exception(
                "Can't open `{}` because it doesn't exist" "".format(path_in)
            )
        da = xr.open_dataarray(path_in, decode_times=False)

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
