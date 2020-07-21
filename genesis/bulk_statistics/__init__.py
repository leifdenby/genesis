import os
import warnings

import xarray as xr
import numpy as np


cp_d = 1005.46  # [J/kg/K]
L_v = 2.5008e6  # [J/kg]
rho0 = 1.2  # [kg/m^3]


def scale_field(da_in):
    if da_in.name.endswith("_flux"):
        if da_in.name == "q_flux":
            assert da_in.units == "m/s g/kg"
            # NB: units in output from UCLALES states `g/kg` but values are
            # `kg/kg` so we don't do scaling to kg here
            da = L_v * rho0 * da_in
            da.attrs["units"] = "W/m^2"
            da.attrs["tex_label"] = r"$\rho_0 L_v w'q_t'$"
            da.attrs["longname"] = da_in.longname
        elif da_in.name == "t_flux":
            assert da_in.units == "m/s K"
            da = cp_d * rho0 * da_in
            da.attrs["units"] = "W/m^2"
            da.attrs["tex_label"] = r"$\rho_0 c_{p,d} w'\theta_l'$"
            da.attrs["longname"] = da_in.longname
        else:
            raise NotImplementedError
    else:
        da = da_in

    return da


def load_mask(input_name, mask_name, mask_field=None, invert=False):

    if mask_field is None:
        mask_field = mask_name

    fn_mask = "{}.mask_3d.{}.nc".format(input_name, mask_name)
    fn_mask_2d = "{}.mask.{}.nc".format(input_name, mask_name)
    if os.path.exists(fn_mask):
        pass
    elif os.path.exists(fn_mask_2d):
        fn_mask = fn_mask_2d
        print("Using 2D xy mask")
    else:
        raise Exception(
            "Couldn't find mask file `{}` or `{}`" "".format(fn_mask, fn_mask_2d)
        )

    ds_mask = xr.open_dataset(fn_mask, decode_times=False)

    if mask_field not in ds_mask:
        raise Exception(
            "Can't find `{}` in mask, loaded mask file:\n{}"
            "".format(mask_field, str(ds_mask))
        )
    else:
        mask = ds_mask[mask_field]

    if invert:
        mask_attrs = mask.attrs
        mask = ~mask
        mask.attrs.update(mask_attrs)
        mask.name = "{}__inverted".format(mask.name)

    return mask


def load_field(fn, autoscale=True, mask=None):
    da_in = xr.open_dataarray(fn, decode_times=False, chunks=dict(zt=20))

    if autoscale:
        da_in = scale_field(da_in)

    if mask is not None:
        # have to keep a reference to the field name because xarray drops it
        field_name = da_in.name
        if len(da_in.zt) > len(mask.zt):
            # mask has smaller vertical extent than data, crop data
            warnings.warn("Mask has smaller vertical extent than data, " "masking data")
            if "time" not in mask.dims:
                mask = mask.expand_dims("time")
            da_in = da_in.sel(zt=mask.zt).where(mask, 0.0)
            da_in.name = field_name
        else:
            # ensure that we've only got mask on levels where scalar being
            # analysed is defined
            mask = mask.sel(zt=da_in.zt)
            da_in = da_in.where(mask, 0.0)
            da_in.name = field_name

    return da_in


def is_older(fn1, fn2):
    return os.path.getmtime(fn1) > os.path.getmtime(fn2)


def get_distribution_in_cross_sections(
    fn, dv_bin, z_slice=None, autoscale=True, mask=None
):

    label_components = ["{}".format(dv_bin)]
    if z_slice is not None:
        label_components.append(
            "z_{}_{}_{}".format(z_slice.start, z_slice.stop, z_slice.step)
        )
    if autoscale:
        label_components.append("autoscaled")
    if mask is not None:
        label_components.append("masked_by_{}".format(mask.name))

    fn_out = fn.replace(
        ".nc", ".cross_section_dist__{}.nc".format("__".join(label_components))
    )

    if os.path.exists(fn_out) and is_older(fn_out, fn):
        return xr.open_dataarray(fn_out, decode_times=False)
    else:
        da_in = load_field(fn, autoscale=autoscale, mask=mask)

        da_out = calc_distribution_in_cross_sections(
            da_in, ds_bin=dv_bin, z_slice=z_slice
        )
        if mask is not None:
            da_out.attrs["mask"] = mask.name

        da_out.to_netcdf(fn_out)
        return da_out


def calc_distribution_in_cross_sections(da_s, ds_bin, z_slice=None):
    """
    `s_da`: scalar being binned in horizontal cross-sections
    `ds_bin`: bin-size
    """

    def calc_bins(v_, dv_):
        try:
            v_min = v_.min()
            v_max = v_.max()
            dv_tot = v_max - v_min
            if dv_tot < dv_:
                warnings.warn(
                    "Provided bin-size is smaller than data range {} < {}".format(
                        float(dv_), float(dv_tot)
                    )
                )

            v_min_padded = dv_ * np.floor(v_min / dv_)
            v_max_padded = dv_ * np.ceil(v_max / dv_)
        except ValueError:
            return None, 1

        n = int(np.round((v_max_padded - v_min_padded) / dv_))

        if n == 0:
            return None, 1
        else:
            return (v_min_padded, v_max_padded), n

    z_var = da_s.coords["zt" if "zt" in da_s.coords else "zm"]

    # get data will be working on
    if z_slice is not None:
        z = z_var.sel(**{z_var.name: z_slice})
    else:
        z = z_var
    da_in = da_s.sel(**{z_var.name: z, "drop": True})

    # setup array for output
    v_range, n_bins = calc_bins(da_in, dv_=ds_bin)
    bins = np.linspace(v_range[0], v_range[1], num=n_bins + 1)

    def calc_distribution_in_cross_section(da_cross):
        count_in_bin = lambda v: xr.DataArray(v.shape[0])  # noqa
        da_counts = da_cross.groupby_bins(da_cross, bins=bins).apply(count_in_bin)

        return da_counts

        pass

    da_binned = da_in.groupby(z).apply(calc_distribution_in_cross_section)

    # XXX: it's not possible to serialise intervals, so we compute the bin
    #      centers here instead to use as the coordinates
    bins_label = "{}_bins".format(da_s.name)
    bin_centers_label = "{}_bin_centers".format(da_s.name)
    bin_centers = [0.5 * (b.left + b.right) for b in da_binned[bins_label].values]
    da_binned[bin_centers_label] = (da_binned[bins_label].dims, bin_centers)
    da_binned = da_binned.swap_dims({bins_label: bin_centers_label}).drop(bins_label)

    da_binned.name = "bin_counts"
    da_binned.attrs["bin_width"] = ds_bin
    da_binned[bin_centers_label].attrs["units"] = da_s.units
    da_binned[bin_centers_label].attrs["longname"] = da_s.longname
    if "tex_label" in da_s.attrs:
        da_binned[bin_centers_label].attrs["tex_label"] = da_s.tex_label
    da_binned = da_binned.expand_dims("time")
    da_binned["time"] = (da_in.time.dims, da_in.time, da_in.time.attrs)

    return da_binned


def make_cumulative_from_bin_counts(da_bin_counts, reverse=True):
    bin_centers_label = da_bin_counts.dims[-1]
    bin_center_values = da_bin_counts[bin_centers_label]

    n_cells = da_bin_counts.sum(dim=bin_centers_label)

    tot_per_bin = da_bin_counts * bin_center_values
    tot_per_bin.attrs["units"] = bin_center_values.units
    tot_per_bin.name = bin_centers_label.replace("bin_centers", "per_bin")

    if reverse:
        s = Ellipsis, slice(None, None, -1)
    else:
        s = slice(None)

    scaled_cumsum = tot_per_bin[s].cumsum(dim=bin_centers_label)[s] / n_cells
    scaled_cumsum.attrs["units"] = tot_per_bin.units
    if "tex_label" in bin_center_values.attrs:
        scaled_cumsum.attrs["tex_label"] = "cumulative {}".format(
            bin_center_values.tex_label
        )
    scaled_cumsum.name = "cumulative {}".format(
        bin_centers_label.replace("_bin_centers", "")
    )

    return scaled_cumsum


def get_dataset(input_name, variables, output_fn_for=None, p=""):
    if input_name.endswith(".nc"):
        dataset = xr.open_dataset(input_name, decode_times=False)

        for var_name in enumerate(variables):
            if var_name.endswith("_flux") and var_name not in dataset:
                raise NotImplementedError("add_flux_var")
                # _add_flux_var(dataset, var_name.replace('_flux', ''))

        out_filename = input_name.replace(".nc", ".{}.pdf".format(output_fn_for))
    else:
        out_filename = input_name + ".{}.pdf".format(output_fn_for)

        # have to handle `w` seperately because it is staggered
        filenames = [
            p + "{}.{}.nc".format(input_name, var_name)
            for var_name in variables
            if not var_name == "w"
        ]
        missing = [fn for fn in filenames if not os.path.exists(fn)]

        if len(filenames) > 0:
            if len(missing) > 0:
                raise Exception("Missing files: {}".format(", ".join(missing)))

            dataset = xr.open_mfdataset(
                filenames, decode_times=False, concat_dim=None, chunks=dict(zt=20)
            )
        else:
            dataset = None

        if "w" in variables:
            d2 = xr.open_dataset(
                "{}.w.nc".format(input_name), decode_times=False, chunks=dict(zm=20)
            )

            if dataset is not None:
                dataset = xr.merge([dataset, d2])
            else:
                dataset = d2

    if output_fn_for is not None:
        return dataset, out_filename
    else:
        return dataset
