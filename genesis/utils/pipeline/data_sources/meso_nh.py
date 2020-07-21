from pathlib import Path
import warnings
from collections import OrderedDict

import ipdb
import xarray as xr
import numpy as np

FIELD_NAME_MAPPING = dict(
    w="WT",
    w_zt="WT",
    xt="W_E_direction",
    yt="S_N_direction",
    zt="vertical_levels",
    qv="RVT",
    qc="RCT",
    theta="THT",
    cvrxp="SVT001",
)

FIELD_DESCRIPTIONS = dict(
    w="vertical velocity",
    qv="water vapour",
    qc="cloud liquid water",
    t="potential temperature",
    cvrxp="radioactive tracer",
    theta="potential temperature",
)

UNITS_FORMAT = {
    "METERS/SECOND": "m/s",
    "KELVIN": "K",
    "KG/KG": "kg/kg",
    "kg kg-1": "kg/kg",
}

DERIVED_FIELDS = {
    "theta_v": ("qv", "theta"),
    "d_theta_v": ("theta_v",),
    "ddz_w": ("w"),
    "ddt_w": ("w", "ddz_w"),
    "ww": ("w",),
}


def _get_meso_nh_field(field_name):
    field_name_src = FIELD_NAME_MAPPING.get(field_name)

    if field_name_src is None:
        raise NotImplementedError(
            "please define a mapping for the field `{}`"
            " in {}".format(field_name, __file__)
        )

    return field_name_src


def _get_meso_nh_field_description(field_name):
    field_description = FIELD_DESCRIPTIONS.get(field_name)

    if field_description is None:
        raise NotImplementedError(
            "please define a description for the field "
            "`{}` in {}".format(field_name, __file__)
        )

    return field_description


def _scale_field(da):
    if da.units == "km":
        da.values *= 1000.0
        da.attrs["units"] = "m"
    elif "q" in da.name and da.units == "kg/kg":
        da.values *= 1000.0
        da.attrs["units"] = "g/kg"

    return da


def _cleanup_units(da):
    if hasattr(da, "units"):
        return UNITS_FORMAT.get(da.units, da.units)
    # one datafile didn't have an actually units attribute, but had a mention
    # in the COMMENT attribute...
    elif hasattr(da, "COMMENT"):
        if da.COMMENT.strip().endswith("(KG/KG)"):
            return "kg/kg"
        elif da.COMMENT.strip().endswith("(K)"):
            return "K"
        elif da.COMMENT.strip().endswith("(m/s)"):
            return "m/s"
        raise NotImplementedError(da.COMMENT)
    # sometimes the distance coordinates don't have units it seems..
    elif da.name in ["xt", "yt"]:
        warnings.warn("Assuming units for `{}` are in km".format(da.name))
        return "km"
    else:
        raise NotImplementedError


def _calculate_theta_v(theta, qv):
    assert qv.units == "g/kg"
    assert theta.units == "K"

    return theta * (1.0 + 0.61 * qv / 1000.0)


def _get_height_coordinate(ds, horz_coords):
    if "VLEV" in ds.data_vars:
        # all height levels should be the same, we do a quick check and then
        # use those values
        z__min = ds.VLEV.min(dim=horz_coords)
        z__max = ds.VLEV.max(dim=horz_coords)
        assert np.all(z__max - z__min < 1.0e-10)
        da_zt = np.round(z__min, decimals=3)
        da_zt.attrs["units"] = ds[horz_coords[0]].units
        return da_zt
    elif "vertical_levels" in ds.coords:
        da_zt = np.round(ds.vertical_levels * 1000.0, decimals=3)
        da_zt.attrs["units"] = "m"
        da_zt.attrs["long_name"] = "height"
        return da_zt
    else:
        raise NotImplementedError


def _center_vertical_velocity_field(w_old):
    w_bottom = w_old.isel(level_w=slice(0, -1))
    w_top = w_old.isel(level_w=slice(1, None))

    w_center = 0.5 * (w_bottom.values + w_top.values)
    zt = 0.5 * (w_bottom.level_w.values + w_top.level_w.values)

    dims = list(w_old.dims)
    dims[dims.index("level_w")] = "zt"

    # create new coordinates for cell-centered vertical velocity
    coords = OrderedDict(w_old.coords)
    del coords["level_w"]
    coords["zt"] = zt

    w_cc = xr.DataArray(
        w_center,
        coords=coords,
        dims=dims,
        attrs=dict(units=w_old.units, long_name="vertical velocity"),
    )
    w_cc.zt.attrs["units"] = "m"
    w_cc.zt.attrs["long_name"] = "height"
    w_cc.name = "w"

    return w_cc


def extract_field_to_filename(dataset_meta, path_out, field_name, **kwargs):
    if field_name == "theta_v":
        assert "qv" in kwargs and "theta" in kwargs

        da_theta = kwargs["theta"].open()
        da_qv = kwargs["qv"].open()

        da = _calculate_theta_v(theta=da_theta, qv=da_qv)
        da.name = "theta_v"
        da.attrs["units"] = "K"
        da.attrs["long_name"] = "virtual potential temperature"
        da.name = field_name
    elif field_name.startswith("d_"):
        da_v = kwargs[field_name[2:]].open()

        v_mean = da_v.mean(dim=("xt", "yt"), dtype=np.float64, keep_attrs=True)
        dv = da_v - v_mean
        dv.attrs["long_name"] = "{} horz. dev.".format(da_v.long_name)
        dv.attrs["units"] = da_v.units
        da = dv
    elif field_name.startswith("ddz_"):
        da_v = kwargs[field_name[4:]].open()

        z_axis = list(da_v.dims).index("zt")

        dv = np.gradient(da_v, axis=z_axis)
        dz = np.gradient(da_v.zt)

        da_dv = xr.DataArray(dv, coords=da_v.coords, dims=da_v.dims)
        da_dz = xr.DataArray(dz, coords=da_v.zt.coords, dims=da_v.zt.dims)

        ddz_v = da_dv / da_dz
        ddz_v.attrs["units"] = da_v.units + "/" + da_v.zt.units
        ddz_v.attrs["long_name"] = "vertical gradient of {}".format(da_v.long_name)
        ddz_v.name = field_name

        da = ddz_v
    elif field_name.startswith("ddt_"):
        ddz_v = kwargs[field_name.replace("ddt_", "ddz_")].open()
        da_w = kwargs["w"].open()

        ddt_v = da_w * ddz_v
        long_name = ddz_v.long_name.replace(
            "vertical gradient of", "time rate of change of"
        )
        ddt_v.attrs["long_name"] = long_name
        ddt_v.attrs["units"] = ddz_v.units.replace("/m", "/s")
        ddt_v.name = field_name

        da = ddt_v
    elif field_name == "ww":
        da_w = kwargs["w"].open()
        da_ww = da_w * da_w
        long_name = "form-drag from vertical velocity"
        da_ww.attrs["long_name"] = long_name
        da_ww.attrs["units"] = "m^2/s^2"
        da_ww.name = field_name

        da = da_ww
    else:
        field_name_src = _get_meso_nh_field(field_name)
        fn_format = dataset_meta["fn_format"]

        path_in = Path(dataset_meta["path"]) / fn_format.format(
            field_name=field_name_src, **dataset_meta
        )

        ds = xr.open_dataset(path_in)

        field_name_src = _get_meso_nh_field(field_name)

        da = ds[field_name_src]
        if field_name == "w_zt":
            field_name = "w"
        da.name = field_name

        new_coords = "xt yt".split(" ")
        old_coords = [_get_meso_nh_field(c) for c in new_coords]
        coord_map = dict(zip(old_coords, new_coords))
        da = da.rename(coord_map)
        da.attrs["long_name"] = _get_meso_nh_field_description(field_name)
        da.attrs["units"] = _cleanup_units(da)

        if field_name == "w" and "level_w" in da.coords:
            da = _center_vertical_velocity_field(w_old=da)
        else:
            da["zt"] = _get_height_coordinate(ds=ds, horz_coords=old_coords)
            da = da.swap_dims(dict(vertical_levels="zt"))

        for c in "xt yt zt".split(" "):
            if "fixes" in dataset_meta:
                if "{}_scale_is_km".format(c[0]) in dataset_meta["fixes"]:
                    da.coords[c].attrs["units"] = "km"

            da.coords[c].attrs["units"] = _cleanup_units(da[c])
            da.coords[c] = _scale_field(da[c])

        _scale_field(da)

        da = da.squeeze()

    da.to_netcdf(path_out)
