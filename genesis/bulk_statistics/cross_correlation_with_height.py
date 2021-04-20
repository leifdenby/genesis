"""
Produce cross-correlation contour plots as function of height and at
cloud-base.  Regions of highest density percentile are contoured
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
import xarray as xr

from ..utils.plot_types import JointHistPlotError, joint_hist_contoured
from . import get_dataset

Z_LEVELS_DEFAULT = np.arange(12.5, 650.0, 100.0)


def extract_from_3d_at_heights_in_2d(da_3d, z_2d):
    z_unique = np.unique(z_2d)
    z_unique = z_unique[~np.isnan(z_unique)]
    v = xr.concat(
        [da_3d.sel(zt=z_).where(z_2d == z_, drop=True) for z_ in z_unique], dim="zt"
    )
    return v.max(dim="zt")


def main(
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
                    xd=xd,
                    yd=yd,
                    normed_levels=normed_levels,
                    ax=ax,
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
            bbox_to_anchor=(
                x_loc,
                0.5,
            ),
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
