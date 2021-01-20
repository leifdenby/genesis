"""
Utility script to process all 3D dataset and calculate cumulant length-scales
in 2D domain-wide cross-sections
"""
import os
import warnings

import xarray as xr
import numpy as np
from tqdm import tqdm

from collections import OrderedDict

from .. import calc as cumulant_analysis


def z_center_field(phi_da):
    assert phi_da.dims[-1] == "zm"

    # average vertical velocity to cell centers
    zt_vals = 0.5 * (phi_da.zm[1:].values + phi_da.zm[:-1].values)
    zt = xr.DataArray(
        zt_vals, coords=dict(zt=zt_vals), attrs=dict(units="m"), dims=("zt",)
    )

    # create new coordinates for cell-centered vertical velocity
    coords = OrderedDict(phi_da.coords)
    del coords["zm"]
    coords["zt"] = zt

    phi_cc_vals = 0.5 * (phi_da[..., 1:].values + phi_da[..., :-1].values)

    dims = list(phi_da.dims)
    dims[dims.index("zm")] = "zt"

    phi_cc = xr.DataArray(
        phi_cc_vals,
        coords=coords,
        dims=dims,
        attrs=dict(units=phi_da.units, longname=phi_da.longname),
    )

    phi_cc.name = phi_da.name

    return phi_cc


def compute_vertical_flux(phi_da, w_da):
    """
    Compute vertical flux of `phi_da` using vertical velocities `w_da`. Both are
    expected to be xarray.DataArray
    """
    assert phi_da.time == w_da.time
    assert phi_da.dims[-1] == "zt"
    assert w_da.dims[-1] == "zm"

    w_cc = z_center_field(phi_da=w_da)

    w_horz_mean = w_cc.mean("x").mean("y")
    v_horz_mean = phi_da.mean("x").mean("y")
    dw = w_cc - w_horz_mean
    dv = phi_da - v_horz_mean

    v_flux = dw * dv
    v_flux.attrs["units"] = "{} {}".format(w_da.units, phi_da.units)
    v_flux.attrs["longname"] = "{} vertical flux".format(phi_da.longname)

    return v_flux


def _extract_horizontal(da, z):
    if not hasattr(da, "from_file"):
        # warnings.warn("input dataset doesn't have its `from_file` "
        # "attribute set so can't store horizontal cross-sections "
        # "for optimisation")
        da_slice = da.where(da.zt == z, drop=True).squeeze()
        da_slice.name = da.name
    else:
        fn = da.from_file

        fn_slice = os.path.join(
            os.path.dirname(fn),
            "k-slices",
            os.path.basename(fn).replace(".nc", ".z{}m.nc".format(z.values)),
        )

        if not os.path.exists(os.path.dirname(fn_slice)):
            raise Exception(
                "Remember to symlink k-slices into `{}`".format(os.path.dirname(fn))
            )

        if not os.path.exists(fn_slice):
            da_slice = da.isel(time=0, drop=True).where(da.zt == z, drop=True).squeeze()

            # copy over the shorthand name so that they can be used when naming the
            # cumulant
            da_slice.name = da.name

            da_slice.to_netcdf(fn_slice)
            da_slice.close()

        da_slice = xr.open_dataarray(fn_slice, decode_times=False)
        # da_slice = da_slice.transpose('xt', 'yt')

    return da_slice


def get_height_variation_of_characteristic_scales(
    v1_3d,
    z_max,
    width_method=cumulant_analysis.WidthEstimationMethod.MASS_WEIGHTED,
    v2_3d=None,
    z_min=0.0,
    mask=None,
    sample_angle=None,
):

    datasets = []

    z_ = v1_3d.zt[np.logical_and(v1_3d.zt > z_min, v1_3d.zt <= z_max)]

    for z in tqdm(z_):
        v1 = _extract_horizontal(v1_3d, z=z)

        v1 = v1.rename(dict(xt="x", yt="y"))
        if v2_3d is None:
            v2 = v1
        else:
            v2 = _extract_horizontal(v2_3d, z=z)
            v2 = v2.rename(dict(xt="x", yt="y"))

        if mask is not None:
            if "zt" in mask.coords:
                mask_2d = mask.sel(zt=v1.zt)
            else:
                mask_2d = mask

            mask_2d = mask_2d.rename(dict(xt="x", yt="y"))
            mask_2d.attrs.update(mask.attrs)
        else:
            mask_2d = None

        scales = cumulant_analysis.charactistic_scales(
            v1=v1,
            v2=v2,
            mask=mask_2d,
            sample_angle=sample_angle,
            width_est_method=width_method,
        )

        datasets.append(scales)

    d = xr.concat(datasets, dim="zt")

    # since these were all computations for the same cumulant we can set just
    # one value which will be used for all. This will allow merge across
    # cumulants later
    d["cumulant"] = d.cumulant.values[0]

    return d


def process(
    base_name,
    variable_sets,
    z_min,
    z_max,
    width_method,
    mask=None,
    sample_angle=None,
    debug=False,
):
    param_datasets = []
    for var_name_1, var_name_2 in variable_sets:
        v1_3d = get_data(base_name=base_name, var_name=var_name_1)

        if var_name_1 != var_name_2:
            v2_3d = get_data(base_name=base_name, var_name=var_name_2)
        else:
            v2_3d = None

        print(
            "Extract cumulant length-scales for C({},{})".format(var_name_1, var_name_2)
        )
        characteristic_scales = get_height_variation_of_characteristic_scales(
            v1_3d=v1_3d,
            v2_3d=v2_3d,
            z_max=z_max,
            z_min=z_min,
            mask=mask,
            sample_angle=sample_angle,
            width_method=width_method,
        )

        param_datasets.append(characteristic_scales)

    ds = xr.concat(param_datasets, dim="cumulant")

    ds["dataset_name"] = base_name
    ds["time"] = v1_3d.time

    return ds


def get_fn(base_name, var_name):
    return os.path.join("{}.{}.nc".format(base_name, var_name))


def _check_coords(da):
    if "xt" not in da.coords or "yt" not in da.coords:
        warnings.warn(
            "Coordinates for xt and yt are missing input, assuming "
            "dx=25m for now, but these files need regenerating "
            "after EGU"
        )
        dx = 25.0

        da["xt"] = dx * np.arange(len(da.xt)) - dx * len(da.xt) / 2
        da["yt"] = dx * np.arange(len(da.yt)) - dx * len(da.yt) / 2

    return da


def get_data(base_name, var_name):
    compute_flux = False

    fn = get_fn(base_name=base_name, var_name=var_name)
    if var_name.endswith("_flux") and not os.path.exists(fn):
        var_name = var_name.replace("_flux", "")
        compute_flux = True
    try:
        phi_da = xr.open_dataarray(fn, decode_times=False, chunks=dict(zt=1))
        phi_da.name = var_name
        print("Using {}".format(fn))
    except IOError:
        print("Error: Couldn't find {}".format(fn))
        raise

    if var_name == "w":
        phi_da = z_center_field(phi_da=phi_da)

    # so we can store horizontal slices later
    phi_da.attrs["from_file"] = os.path.realpath(fn)

    if not compute_flux:
        return _check_coords(phi_da)
    else:
        fn_w = get_fn(base_name, var_name, "w")
        w_da = xr.open_dataarray(
            fn_w,
            decode_times=False,
        )

        phi_flux_da = compute_vertical_flux(phi_da=phi_da, w_da=w_da)
        phi_flux_da.name = "{}_flux".format(var_name)

        return _check_coords(phi_flux_da)


def get_cross_section(base_name, var_name, z, z_max=700.0, method=None):

    ar = get_data(base_name=base_name, var_name=var_name)

    da = ar.sel(zt=z, drop=True, method=method).squeeze()
    da["zt"] = ((), z, dict(units="m"))

    return da


FN_FORMAT = "{base_name}.cumulant_scales_profile.{v1}.{v2}.{mask}.{z_max}.nc"

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    DEFAULT_VARS = "w_zt,w_zt d_q,d_q d_t,d_t w_zt,d_q w_zt,d_t".split(" ")

    argparser.add_argument("base_name", help="e.g. `rico_gcss`", type=str)
    argparser.add_argument("--vars", default=DEFAULT_VARS, nargs="+")
    argparser.add_argument("--z_max", default=700.0, type=float, help="maximum height")
    argparser.add_argument("--z_min", default=0.0, type=float, help="minimum height")
    argparser.add_argument(
        "--mask-name", default=None, type=str, help="name of mask file to use"
    )
    argparser.add_argument(
        "--mask-field", default=None, type=str, help="name of mask field in mask file"
    )
    argparser.add_argument(
        "--invert-mask", default=False, action="store_true", help="invert mask"
    )
    argparser.add_argument("--output-in-cwd", default=True, action="store_true")
    argparser.add_argument("--theta", default=None, type=float)
    argparser.add_argument("--debug", default=False, action="store_true")
    argparser.add_argument(
        "--width-method",
        type=lambda m: cumulant_analysis.WidthEstimationMethod[m],
        choices=list(cumulant_analysis.WidthEstimationMethod),
        default=cumulant_analysis.WidthEstimationMethod.MASS_WEIGHTED,
    )

    args = argparser.parse_args()

    out_filename = "{}.cumulant_length_scales.nc".format(args.base_name)

    if args.mask_name is not None:
        if args.mask_field is None:
            mask_field = args.mask_name
        else:
            mask_field = args.mask_field
        mask_description = mask_field

        fn_mask = "{}.{}.mask.nc".format(args.base_name, args.mask_name)
        if not os.path.exists(fn_mask):
            raise Exception("Can't find mask file `{}`".format(fn_mask))

        ds_mask = xr.open_dataset(fn_mask, decode_times=False)
        if mask_field not in ds_mask:
            raise Exception(
                "Can't find `{}` in mask, loaded mask file:\n{}"
                "".format(mask_field, str(ds_mask))
            )
        else:
            mask = ds_mask[mask_field]

        mask = mask.rename(dict(xt="x", yt="y"))

        if args.invert_mask:
            mask_attrs = mask.attrs
            mask = ~mask
            mask.name = "not_{}".format(mask.name)
            mask_description = "not__{}".format(mask_description)
            out_filename = out_filename.replace(
                ".nc", ".masked.{}.nc".format(mask_description)
            )
            mask.attrs.update(mask_attrs)
        else:
            out_filename = out_filename.replace(
                ".nc", ".masked.{}.nc".format(mask_description)
            )
    else:
        mask = None
        mask_description = "full domain"

    variable_sets = []
    for v_pair in args.vars:
        v_split = v_pair.split(",")
        if not len(v_split) == 2:
            raise NotImplementedError(
                "Not sure how to interpret `{}`" "".format(v_pair)
            )
        else:
            variable_sets.append(v_split)

    with np.errstate(all="raise"):
        import ipdb

        with ipdb.launch_ipdb_on_exception():
            data = process(
                base_name=args.base_name,
                variable_sets=variable_sets,
                z_min=args.z_min,
                z_max=args.z_max,
                mask=mask,
                sample_angle=args.theta,
                debug=args.debug,
                width_method=args.width_method,
            )

    data.attrs["mask"] = mask_description
    data.attrs["width_method"] = args.width_method.name.lower()

    out_filename = out_filename.replace(
        ".nc", ".{}_width.nc".format(args.width_method.name.lower())
    )

    if args.output_in_cwd:
        out_filename = out_filename.replace("/", "__")

    import ipdb

    with ipdb.launch_ipdb_on_exception():
        data.to_netcdf(out_filename, mode="w")
        print("Output written to {}".format(out_filename))
