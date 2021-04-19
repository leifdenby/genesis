"""
Routines for plotting cumulant characteristics from netCDF datafile
"""
if __name__ == "__main__":  # noqa
    import matplotlib

    matplotlib.use("Agg")

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from tqdm import tqdm

from . import calc as cumulant_analysis
from .calc import WidthEstimationMethod


def _load_data(dataset_names, var_names, z_max, z_step):
    datasets = []

    var_names = ["w_zt" if v == "w" else v for v in var_names]

    for dn in dataset_names:
        ds = xr.open_mfdataset(
            ["{}.{}.nc".format(dn, vn) for vn in var_names], decode_times=False
        )

        ds = ds.sel(zt=slice(0, z_max)).isel(zt=slice(None, None, z_step))
        ds.attrs["name"] = dn

        datasets.append(ds)

    return datasets


def plot(
    datasets,
    var_names,
    est_method=WidthEstimationMethod.MASS_WEIGHTED,
    figwidth=7.0,
    add_figlegend=False,
):
    z = datasets[0].zt

    v1, v2 = var_names

    figheight = figwidth / 7.0 * 2.3

    fig, axes = plt.subplots(
        ncols=len(datasets) * 2,
        nrows=len(z),
        figsize=(figwidth * len(datasets), figheight * len(z)),
    )

    loc_x = plticker.MultipleLocator(base=200)

    def sp(ds_, ax1, ax2):
        cumulant_analysis.covariance_plot(ds_[v1], ds_[v2], log_scale=False, ax=ax1)
        ax1.set_title("")

        cumulant_analysis.covariance_direction_plot(
            ds_[v1], ds_[v2], ax=ax2, width_est_method=est_method
        )
        ax2.set_title("")

        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax2.get_legend().remove()
        ax2.xaxis.set_minor_locator(loc_x)
        ax2.axhline(0.0, linestyle="--", color="grey")

        ax2.grid(which="minor", axis="both", linestyle="--")

    # plot in order of decreasing height
    for n, z_ in enumerate(tqdm(z[::-1])):

        for n_d in range(len(datasets)):
            ds_ = datasets[n_d].sel(zt=z_).squeeze().rename(dict(xt="x", yt="y"))
            sp(ds_, axes[n, n_d * 2], axes[n, n_d * 2 + 1])

        axes[n, 0].text(
            -1.0, 1.3, "z={}m".format(z_.values), transform=axes[n, 0].transAxes
        )

    for n_d in range(len(datasets)):
        ax = axes[0, n_d * 2]
        ax.text(
            1.7,
            1.4,
            datasets[n_d].name,
            transform=ax.transAxes,
            horizontalalignment="center",
        )

    fig.tight_layout()

    handles, labels = axes[0, 1].get_legend_handles_labels()
    labels = [label.split("=")[0].strip() + "$" for label in labels]
    labels = [f"{label}: principle dir." if r"{p}" in label else f"{label}: perpendicular dir." for label in labels]
    plt.figlegend(
        handles, labels, loc="upper right", bbox_to_anchor=(1.0, 0.0),
    )

    return fig, axes


FN_FORMAT_PLOT = "cumulant_with_height__{v1}__{v2}.{filetype}"

if __name__ == "__main__":
    import seaborn as sns

    sns.set(style="ticks")

    import argparse

    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument("cumulant")
    argparser.add_argument(
        "input", nargs="+", help="input datasets, e.g. no_shear/rico.tn6"
    )
    argparser.add_argument(
        "--z_max",
        help="max height",
        default=700,
        type=float,
    )
    argparser.add_argument("--z_step", default=4)

    args = argparser.parse_args()

    var_names = args.cumulant.split(",")

    datasets = _load_data(
        args.input, var_names=set(var_names), z_max=args.z_max, z_step=args.z_step
    )

    import ipdb

    with ipdb.launch_ipdb_on_exception():
        plot(datasets, var_names)

    fn = FN_FORMAT_PLOT.format(v1=var_names[0], v2=var_names[1], filetype="pdf")

    plt.savefig(fn, bbox_inches="tight")

    print("Saved figure to {}".format(fn))
