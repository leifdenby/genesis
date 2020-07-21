"""
Produce cross-correlation contour plots as function of height and at
cloud-base.  Regions of highest density percentile are contoured
"""
import os
import warnings

import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

try:
    from cloud_tracking_analysis import CloudData
    # from cloud_tracking_analysis.cloud_mask_methods import (
    #     cloudbase as get_cloudbase_mask,
    # )
    # from cloud_tracking_analysis.cloud_mask_methods import CloudbaseEstimationMethod

    HAS_CLOUD_TRACKING = True
except ImportError:
    HAS_CLOUD_TRACKING = False

from . import get_dataset
from ..utils.plot_types import joint_hist_contoured, JointHistPlotError
from ..objects import projected_2d as objs_2d


Z_LEVELS_DEFAULT = np.arange(12.5, 650.0, 100.0)


def get_approximate_cloudbase_height(qc, z_tol=100.0):
    z_cloud_underside = qc.zt.where(qc > 0.0).min(dim="zt")

    m = z_cloud_underside < z_tol + z_cloud_underside.min()
    z_cb = z_cloud_underside.where(m)

    return z_cb


def get_cloudbase_height(
    ds_tracking, da_cldbase_2d, t0, t_age_max, dx, z_base_max=700.0
):

    # da_cldbase_2d_ = da_cldbase_2d.sel(time=t0)

    object_set = objs_2d.ObjectSet(ds=ds_tracking)

    # clouds that are going to do vertical transport
    object_set = object_set.filter(
        cloud_type__in=[objs_2d.CloudType.SINGLE_PULSE, objs_2d.CloudType.ACTIVE]
    )

    object_set = object_set.filter(present=True, kwargs=dict(t0=t0))

    # avoid mid-level convection clouds
    object_set = object_set.filter(
        cloudbase_max_height_by_histogram_peak__lt=z_base_max,
        kwargs=dict(t0=t0, dx=dx, da_cldbase=da_cldbase_2d),
    )

    # remove clouds that are more than 3min old
    object_set = object_set.filter(cloud_age__lt=t_age_max, kwargs=dict(t0=t0))

    # nrcloud_cloudbase = get_cloudbase_mask(
    #    object_set=object_set, t0=t0, method=CloudbaseEstimationMethod.DEFAULT
    # )

    raise NotImplementedError
    # cldbase = object_set.cloud_data.get('cldbase', tn=tn)
    # m = nrcloud_cloudbase == 0
    # cldbase_heights_2d = cldbase.where(~m)

    # cldbase_heights_2d.attrs['num_clouds'] = len(object_set)

    # return cldbase_heights_2d


def extract_from_3d_at_heights_in_2d(da_3d, z_2d):
    z_unique = np.unique(z_2d)
    z_unique = z_unique[~np.isnan(z_unique)]
    v = xr.concat(
        [da_3d.sel(zt=z_).where(z_2d == z_, drop=True) for z_ in z_unique], dim="zt"
    )
    return v.max(dim="zt")


def get_cloudbase_data(cloud_data, v, t0, t_age_max=200.0, z_base_max=700.0):
    raise NotImplementedError

    # v__belowcloud = cloud_data.get_from_3d(var_name=v, z=z_slice, t=t0)

    # dx = cloud_set.cloud_data.dx
    # try:
    # w__belowcloud = cloud_data.get_from_3d(var_name='w', z=z_slice+dx/2., t=t0)
    # except:
    # pass

    # return v__belowcloud.where(m, drop=True)


def main( # noqa
    ds_3d,
    ds_cb=None,
    normed_levels=[10, 90],
    ax=None,
    add_cb_peak_ref_line=False,
    add_legend=True,
):
    colors = iter(sns.color_palette("cubehelix", len(ds_3d.zt)))
    sns.set_color_codes()

    lines = []

    v1, v2 = ds_3d.data_vars.keys()

    if v1 in ["q", "d_q"] and ds_3d[v1].units == "g/kg":
        warnings.warn(
            "Scaling variable `q` by 1000 since UCLALES "
            "incorrectly states the units as g/kg even "
            "though they are in fact in kg/kg"
        )
        xscale = 1000.0
    else:
        xscale = 1.0

    if v2 in ["q", "d_q"] and ds_3d[v2].units == "g/kg":
        warnings.warn(
            "Scaling variable `q` by 1000 since UCLALES "
            "incorrectly states the units as g/kg even "
            "though they are in fact in kg/kg"
        )
        yscale = 1000.0
    else:
        yscale = 1.0

    if ax is None:
        ax = plt.gca()

    for z in tqdm.tqdm(ds_3d.zt):
        ds_ = ds_3d.sel(zt=z, method="nearest").squeeze()

        c = next(colors)
        try:
            xd = ds_[v1].values.flatten() * xscale
            yd = ds_[v2].values.flatten() * yscale

            _, _, cnt = joint_hist_contoured(
                xd=xd, yd=yd, normed_levels=normed_levels, ax=ax
            )

            for n, l in enumerate(cnt.collections):
                l.set_color(c)
                if n == 0:
                    l.set_label("z={}m".format(ds_.zt.values))
                    lines.append(l)

            if 0.0 in cnt.levels or len(cnt.levels) != len(normed_levels):
                ax.scatter(xd.mean(), yd.mean(), marker=".", color=c)

        except JointHistPlotError:
            print("error", ds_.zt.values, "skipping")
        except Exception:
            print("error", ds_.zt.values)
            raise

    if ds_cb is not None:
        import ipdb

        with ipdb.launch_ipdb_on_exception():
            if v1 in ds_cb.variables and v2 in ds_cb.variables:
                xd = ds_cb[v1].values.flatten() * xscale
                xd = xd[~np.isnan(xd)]
                yd = ds_cb[v2].values.flatten() * yscale
                yd = yd[~np.isnan(yd)]

                (x_bins, y_bins), bin_counts, cnt = joint_hist_contoured(
                    xd=xd, yd=yd, normed_levels=normed_levels, ax=ax,
                )

                if add_cb_peak_ref_line:
                    idx_max = np.argmax(bin_counts)
                    x_ref = x_bins.flatten()[idx_max]
                    y_ref = y_bins.flatten()[idx_max]
                    kwargs = dict(linestyle="--", alpha=0.3, color="grey")
                    ax.axhline(y_ref, **kwargs)
                    ax.axvline(x_ref, **kwargs)

                if 0.0 in cnt.levels or len(cnt.levels) != len(normed_levels):
                    ax.scatter(xd.mean(), yd.mean(), marker=".", color="red")

                for n, l in enumerate(cnt.collections):
                    l.set_color("red")

                    if n == 0:
                        if "method" in ds_cb[v1].attrs:
                            assert ds_cb[v1].method == ds_cb[v2].method
                            l.set_label("into cloudbase\n({})".format(ds_cb[v1].method))
                        else:
                            l.set_label("into cloudbase")
                        lines.append(l)
            else:
                warnings.warn("Skipping cloud base plot, missing one or more variables")

    if add_legend:
        x_loc = 1.04
        if add_legend == "far_right":
            x_loc = 1.2
        ax.legend(
            handles=lines,
            labels=[l.get_label() for l in lines],
            loc="center left",
            bbox_to_anchor=(x_loc, 0.5,),
            borderaxespad=0,
        )

    sns.despine()

    ax.set_xlabel(xr.plot.utils.label_from_attrs(ds_3d[v1]))
    ax.set_ylabel(xr.plot.utils.label_from_attrs(ds_3d[v2]))

    if type(ds_.time.values) == float:
        ax.set_title("t={}hrs".format(ds_.time.values / 60 / 60))
    else:
        ax.set_title("t={}".format(ds_.time.values))

    if axis_lims_spans_zero(ax.get_xlim()):
        ax.axvline(0.0, linestyle="--", alpha=0.2, color="black")
    if axis_lims_spans_zero(ax.get_ylim()):
        ax.axhline(0.0, linestyle="--", alpha=0.2, color="black")

    return ax, lines


def axis_lims_spans_zero(lims):
    return np.sign(lims[0]) != np.sign(lims[1])


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__)

    argparser.add_argument("input_name")
    argparser.add_argument("tracking_identifier", type=str)
    argparser.add_argument("var1", type=str)
    argparser.add_argument("var2", type=str)
    argparser.add_argument("--z", type=float, nargs="+", default=Z_LEVELS_DEFAULT)
    argparser.add_argument("--mask", type=str, default=None)
    argparser.add_argument(
        "--output-format", default="png", type=str, choices=["png", "pdf"]
    )
    args = argparser.parse_args()

    input_name = args.input_name
    var_name1 = args.var1
    var_name2 = args.var2
    dataset_name_with_time = input_name.split("/")[-1]
    dataset_name = input_name.split("/")[-1].split(".")[0]
    case_name = input_name.split("/")[0]

    ds_3d = get_dataset(
        dataset_name_with_time,
        variables=[var_name1, var_name2],
        p="{}/3d_blocks/full_domain/".format(case_name),
    )

    if args.mask is not None:
        mask_3d = get_dataset(
            dataset_name_with_time,
            variables=["mask_3d.{}".format(args.mask)],
            p="{}/masks/".format(case_name),
        )[args.mask]
    else:
        mask_3d = None

    t0 = ds_3d.time.values

    import cloud_tracking_analysis.cloud_data

    cloud_tracking_analysis.cloud_data.ROOT_DIR = os.getcwd()
    cloud_data = CloudData(
        dataset_name, args.tracking_identifier, dataset_pathname=case_name
    )

    ds_cb = get_cloudbase_data(cloud_data=cloud_data, t0=t0)

    if mask_3d is not None:
        ds_3d = ds_3d.where(mask_3d)

    ds_3d = ds_3d.sel(zt=args.z)
    main(ds_3d=ds_3d, ds_cb=ds_cb)

    name = input_name.replace("/", "__")

    title = "{} {}".format(name, plt.gca().get_title())
    out_fn = "{}.cross_correlation.{}.{}.png".format(name, var_name1, var_name2)
    if args.mask is not None:
        title += "\nmasked by {}".format(mask_3d.longname)
        out_fn = out_fn.replace(".png", ".{}.png".format(args.mask))

    out_fn = out_fn.replace(".png", ".{}".format(args.output_format))

    plt.gca().set_title(title)
    plt.savefig(out_fn)
    print("Saved plot to {}".format(out_fn))
