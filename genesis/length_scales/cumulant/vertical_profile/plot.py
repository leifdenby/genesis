"""
Routines for plotting cumulant characteristics from netCDF datafile
"""
if __name__ == "__main__":  # noqa
    import matplotlib

    matplotlib.use("Agg")

import warnings

import xarray as xr
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..calc import fix_cumulant_name
from ....utils import wrap_angles


FULL_SUITE_PLOT_PARTS = dict(
    l=(0, 0),  # noqa
    q=(0, slice(1, None)),
    t=(1, slice(0, None)),
    w=(2, slice(0, 2)),
    q_flux=(2, 2),
    t_flux=(2, 3),
    l_flux=(2, 4),
)


def plot_full_suite(data, marker=""):
    gs = GridSpec(3, 5)
    figure = plt.figure(figsize=(10, 12))

    aspect = 0.3

    ax = None
    for var_name, s in FULL_SUITE_PLOT_PARTS.items():
        ax = plt.subplot(gs[s], sharey=ax, adjustable="box-forced")

        d_ = data.sel(cumulant="C(l,l)", drop=True)
        z_cb = d_.where(d_.width_principle > 0.1, drop=True).zt.min()
        ax.axhline(z_cb, linestyle=":", color="grey", alpha=0.6)

        for p in data.dataset_name.values:
            lines = []

            cumulant = "C({},{})".format(var_name, var_name)
            d = data.sel(dataset_name=p, drop=True).sel(cumulant=cumulant, drop=True)

            (line,) = plt.plot(
                d.width_principle,
                d.zt,
                marker=marker,
                label="{} principle".format(str(p)),
            )
            (line2,) = plt.plot(
                d.width_perpendicular,
                d.zt,
                marker=marker,
                label="{} orthog.".format(str(p)),
                linestyle="--",
                color=line.get_color(),
            )

            plt.title(fix_cumulant_name(cumulant))

            lines.append(line)
            lines.append(line2)

            plt.xlabel("characteristic width [m]")

        if s[1] == 0 or type(s[1]) == slice and s[1].start == 0:
            plt.ylabel("height [m]")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

    plt.tight_layout()

    for ax in figure.axes:
        # once the plots have been rendered we want to resize the xaxis so
        # that the aspect ratio is the same, note sharey on the subplot above
        _, _, w, h = ax.get_position().bounds
        ax.set_xlim(0, d.zt.max() / h * w * aspect)

    plt.subplots_adjust(bottom=0.10)
    _ = plt.figlegend(lines, [l.get_label() for l in lines], loc="lower center", ncol=2)


def plot_angles(
    data,
    marker=".",
    linestyle="",
    z_max=None,
    cumulants=[],
    split_subplots=True,
    with_legend=True,
    fig=None,
    **kwargs
):

    if len(cumulants) == 0:
        cumulants = data.cumulant.values

    if z_max is not None:
        data = data.copy().where(data.zt < z_max, drop=True)

    if fig is None and split_subplots:
        fig = plt.figure(figsize=(2.5 * len(cumulants), 4))

    ax = None

    axes = []

    data.principle_axis.values = np.rad2deg(
        wrap_angles(np.deg2rad(data.principle_axis))
    )

    for i, cumulant in enumerate(cumulants):
        lines = []
        n = data.cumulant.values.tolist().index(cumulant)
        _ = data.isel(cumulant=n, drop=True).squeeze()
        if split_subplots:
            ax = plt.subplot(1, len(cumulants), i + 1, sharey=ax)
        else:
            ax = plt.gca()
        for p in data.dataset_name.values:
            d = data.sel(dataset_name=p, drop=True).sel(cumulant=cumulant, drop=True)

            (line,) = plt.plot(
                d.principle_axis,
                d.zt,
                marker=marker,
                linestyle=linestyle,
                label="{}, principle axis orientation".format(str(p)),
                **kwargs
            )

            lines.append(line)

        plt.title(fix_cumulant_name(cumulant))
        plt.tight_layout()
        plt.xlabel("angle [deg]")

        if i == 0:
            plt.ylabel("height [m]")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

        axes.append(ax)

    if with_legend:
        plt.subplots_adjust(bottom=0.24)
        _ = plt.figlegend(
            lines, [l.get_label() for l in lines], loc="lower center", ncol=2
        )

    [axes[0].get_shared_x_axes().join(axes[0], ax) for ax in axes[1:]]
    axes[0].autoscale()
    [ax.axvline(0, linestyle="--", color="grey") for ax in axes]

    return axes


def plot(
    data,
    plot_type,
    z_max=None,
    cumulants=[],
    split_subplots=True,
    with_legend=True,
    fill_between_alpha=0.2,
    **kwargs
):

    if plot_type not in ["angles", "scales"]:
        raise Exception

    if len(cumulants) == 0:
        cumulants = data.cumulant.values

    if z_max is not None:
        data = data.copy().where(data.zt < z_max, drop=True)

    _ = data.dataset_name.count()

    if split_subplots:
        fig, axes = plt.subplots(
            ncols=len(cumulants),
            figsize=(2.5 * len(cumulants), 4.0),
            sharex=True,
            sharey=True,
        )
    else:
        fig, ax = plt.subplots()

    if plot_type == "angles":
        data.principle_axis.values = np.rad2deg(
            wrap_angles(np.deg2rad(data.principle_axis))
        )

    for i, cumulant in enumerate(cumulants):
        lines = []
        n = data.cumulant.values.tolist().index(cumulant)
        _ = data.isel(cumulant=n, drop=True).squeeze()

        if split_subplots:
            if len(cumulants) > 1:
                ax = axes[i]
            else:
                ax = axes

        for p in data.dataset_name.values:
            d = (
                data.sel(dataset_name=p, drop=True)
                .sel(cumulant=cumulant, drop=True)
                .dropna(dim="zt")
            )
            d_ = d.where(d.is_covariant == 0, drop=True)

            if plot_type == "angles":
                (line,) = d.principle_axis.plot(
                    ax=ax,
                    y="zt",
                    label="{} principle orientation".format(str(p)),
                    **kwargs
                )
                if d_.zt.count() > 0:
                    d_ = d.where(~d.is_covariant, drop=True)
                    d_.principle_axis.plot(
                        ax=ax, y="zt", color=line.get_color(), marker="_", linestyle=""
                    )
                ax.set_xlabel("principle axis [deg]")
            elif plot_type == "scales":
                (line,) = d.width_principle.plot(
                    ax=ax, y="zt", label="{} principle".format(str(p)), **kwargs
                )

                (line2,) = d.width_perpendicular.plot(
                    ax=ax,
                    y="zt",
                    label="{} perpendicular".format(str(p)),
                    color=line.get_color(),
                    linestyle="--",
                    **kwargs
                )
                lines.append(line2)
                ax.fill_betweenx(
                    y=line.get_ydata(),
                    x1=line.get_xdata(),
                    x2=line2.get_xdata(),
                    color=line.get_color(),
                    alpha=fill_between_alpha,
                )
                if d_.zt.count() > 0:
                    d_ = d.where(d.is_covariant, drop=True)
                    d_.width_principle.plot(
                        ax=ax, y="zt", color=line.get_color(), marker="_", linestyle=""
                    )
                    d_.width_perpendicular.plot(
                        ax=ax, y="zt", color=line.get_color(), marker="_", linestyle=""
                    )
                ax.set_xlabel("characteristic width [m]")
            else:
                raise NotImplementedError(plot_type)
            lines.append(line)

        ax.set_title(fix_cumulant_name(cumulant))
        ax.set_ylabel(["height [m]", ""][i > 0])
        sns.despine()

    plt.tight_layout()
    if with_legend:
        _ = plt.figlegend(
            lines, [l.get_label() for l in lines], loc="lower center", ncol=2
        )
        fig.text(
            0.5, 0.1 - data.dataset_name.count() * 0.1, " ", transform=fig.transFigure
        )

    return axes


def _make_output_filename(input_filenames):
    N = len(input_filenames[0])

    for n in range(N)[::-1]:
        s = input_filenames[0][-n:]
        if all([s_[-n:] == s for s_ in input_filenames]) and s.startswith("."):
            return s[1:].replace(".nc", ".pdf")

    raise Exception(
        "Can't find common root between input filenames: {}"
        "".format(", ".join(input_filenames))
    )


FN_FORMAT = "{base_name}.cumulant_profile_{plot_type}.{mask}.pdf"


if __name__ == "__main__":
    import matplotlib

    sns.set(style="ticks")

    def _parse_cumulant_arg(s):
        v_split = s.split(",")
        if not len(v_split) == 2:
            raise NotImplementedError("Not sure how to interpret `{}`" "".format(s))
        else:
            return v_split

    import argparse

    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument("input", help="input netCDF files", nargs="+")
    argparser.add_argument(
        "--cumulants",
        help="cumulants to plot",
        nargs="*",
        default=[],
        type=_parse_cumulant_arg,
    )
    argparser.add_argument(
        "--z_max",
        help="max height",
        default=None,
        type=float,
    )
    argparser.add_argument(
        "--plot-angles", help="plot angles", default=False, action="store_true"
    )

    argparser.add_argument("--x-max", default=None, type=float)

    args = argparser.parse_args()

    dataset = xr.open_mfdataset(args.input, concat_dim="dataset_name")

    if "dataset_name" not in dataset.dims:
        dataset = dataset.expand_dims("dataset_name")

    do_full_suite_plot = None
    variables = set([v for cumulant in args.cumulants for v in cumulant])

    if len(variables) == 0:
        variables = FULL_SUITE_PLOT_PARTS.keys()
        do_full_suite_plot = True
    else:
        do_full_suite_plot = False

    cumulants = ["C({},{})".format(v1, v2) for (v1, v2) in args.cumulants]

    missing_cumulants = [c for c in cumulants if c not in dataset.cumulant.values]

    if do_full_suite_plot:
        if not len(missing_cumulants) == 0:
            warnings.warn(
                "Not all variables for full suite plot, missing: {}"
                "".format(", ".join(missing_cumulants))
            )

            if args.plot_angles:
                plot_angles(dataset, z_max=args.z_max, cumulants=cumulants)
            else:
                plot(dataset, z_max=args.z_max, cumulants=cumulants)
        else:
            plot_full_suite(dataset)
    else:
        if not len(missing_cumulants) == 0:
            raise Exception(
                "Not all variables for plot, missing: {}"
                "".format(", ".join(missing_cumulants))
            )

        else:
            import ipdb

            with ipdb.launch_ipdb_on_exception():
                if args.plot_angles:
                    plot_angles(dataset, z_max=args.z_max, cumulants=cumulants)
                else:
                    plot(dataset, z_max=args.z_max, cumulants=cumulants)

    if args.x_max:
        for ax in plt.gcf().axes:
            ax.set_xlim(0, args.x_max)

    sns.despine()

    fn = _make_output_filename(args.input)

    if args.plot_angles:
        fn = fn.replace(".pdf", ".angles.pdf")
    plt.savefig(fn, bbox_inches="tight")

    print("Saved figure to {}".format(fn))
