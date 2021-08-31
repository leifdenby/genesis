"""
Routines for plotting cumulant characteristics from netCDF datafile
"""
if __name__ == "__main__":  # noqa
    import matplotlib

    matplotlib.use("Agg")

import warnings

import xarray as xr
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..calc import fix_cumulant_name
from ....utils import wrap_angles

# marker for more than x0, x1, x2, ... asymmetry
ASYMMETRY_MARKERS = ["", "", "_", "^", "s", "p", "h"]


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
            ds = data.sel(dataset_name=p, drop=True).sel(cumulant=cumulant, drop=True)

            (line,) = plt.plot(
                ds.width_principle,
                ds.zt,
                marker=marker,
                label="{} principle direction width $L^p$".format(str(p)),
            )
            (line2,) = plt.plot(
                ds.width_perpendicular,
                ds.zt,
                marker=marker,
                label=r"{} orthog. dir. width $L^{{\bot}}$".format(str(p)),
                linestyle="--",
                color=line.get_color(),
            )

            plt.title(fix_cumulant_name(cumulant))

            lines.append(line)
            lines.append(line2)

            plt.xlabel("cumulant width [m]")

        if s[1] == 0 or type(s[1]) == slice and s[1].start == 0:
            plt.ylabel("height [m]")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

    plt.tight_layout()

    for ax in figure.axes:
        # once the plots have been rendered we want to resize the xaxis so
        # that the aspect ratio is the same, note sharey on the subplot above
        _, _, w, h = ax.get_position().bounds
        ax.set_xlim(0, ds.zt.max() / h * w * aspect)

    plt.subplots_adjust(bottom=0.10)
    _ = plt.figlegend(lines, [l.get_label() for l in lines], loc="lower center", ncol=2)


def _calc_marker_style(ds_point):
    r = int(np.floor(ds_point.width_principle / ds_point.width_perpendicular))

    try:
        return ASYMMETRY_MARKERS[r]
    except IndexError:
        return ASYMMETRY_MARKERS[-1]


def _make_marker_legend():
    marker_handles = []
    for n, marker in enumerate(ASYMMETRY_MARKERS):
        if n < 2:
            continue
        marker_line = mlines.Line2D(
            [],
            [],
            color="black",
            markerfacecolor="none",
            marker=marker,
            linestyle="None",
            markersize=10,
            label=f"$r_a$ > {n}",
        )
        marker_handles.append(marker_line)

    return marker_handles


def _add_asymmetry_markers(ax, ds, color):
    ylim = ax.get_ylim()
    y_vals = np.array(list(filter(lambda v: ylim[0] < v < ylim[1], ax.get_yticks())))

    for y_ in y_vals:
        ds_point = ds.sel(zt=y_, method="nearest")
        marker = _calc_marker_style(ds_point)
        kws = dict(marker=marker)
        if marker != "_":
            kws["facecolor"] = "none"
        ax.scatter(
            ds_point.principle_axis,
            ds_point.zt,
            color=color,
            s=100.0,
            **kws,
        )


def _plot_angles_profile(ds, ax, p, **kwargs):
    (line,) = ds.principle_axis.plot(
        ax=ax, y="zt", label="{} principle orientation".format(str(p)), **kwargs
    )

    ds_not_covariant = ds.where(ds.is_covariant == 0, drop=True)
    if ds_not_covariant.zt.count() > 0:
        ds.principle_axis.plot(
            ax=ax, y="zt", color=line.get_color(), marker="_", linestyle=""
        )
    ax.set_xlabel("principle axis [deg]")
    return line


def _plot_scales_profile(ds, ax, p, fill_between_alpha, **kwargs):
    (line,) = ds.width_principle.plot(
        ax=ax,
        y="zt",
        label="{} principle direction width ($L^p$)".format(str(p)),
        **kwargs,
    )

    if not "color" in kwargs:
        kwargs["color"] = line.get_color()

    (line2,) = ds.width_perpendicular.plot(
        ax=ax,
        y="zt",
        label=r"{} perpendicular direction width ($L^{{\bot}}$)".format(str(p)),
        linestyle="--",
        **kwargs,
    )

    ax.fill_betweenx(
        y=line.get_ydata(),
        x1=line.get_xdata(),
        x2=line2.get_xdata(),
        color=line.get_color(),
        alpha=fill_between_alpha,
    )

    ds_not_covariant = ds.where(ds.is_covariant == 0, drop=True)
    if ds_not_covariant.zt.count() > 0:
        ds_not_covariant.width_principle.plot(
            ax=ax, y="zt", color=line.get_color(), marker="_", linestyle=""
        )
        ds_not_covariant.width_perpendicular.plot(
            ax=ax, y="zt", color=line.get_color(), marker="_", linestyle=""
        )
    ax.set_xlabel("cumulant width [m]")

    return line, line2


def plot(
    data,
    plot_type,
    z_max=None,
    cumulants=[],
    split_subplots=True,
    with_legend=True,
    fill_between_alpha=0.2,
    add_asymmetry_markers=None,
    reference_line_heights=[],
    figwidth=2.0,
    line_colors="default",
    **kwargs,
):
    scale_limits = kwargs.pop("scale_limits", {})

    N_datasets = len(data.dataset_name.values)
    if line_colors == "default":
        line_colors = sns.color_palette(n_colors=N_datasets)
    else:
        assert len(line_colors) == N_datasets

    if plot_type not in ["angles", "scales"]:
        raise Exception

    if plot_type == "angles" and add_asymmetry_markers is None:
        add_asymmetry_markers = True

    if len(cumulants) == 0:
        cumulants = data.cumulant.values

    if z_max is not None:
        data = data.copy().where(data.zt < z_max, drop=True)

    _ = data.dataset_name.count()

    if split_subplots:
        figheight = figwidth / 2.5 * 4.0
        fig, axes = plt.subplots(
            ncols=len(cumulants),
            figsize=(figwidth * len(cumulants), figheight),
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

        for z_ in reference_line_heights:
            ax.axhline(z_, linestyle="--", color="grey", alpha=0.6)

        for i_c, p in enumerate(data.dataset_name.values):
            ds = (
                data.sel(dataset_name=p, drop=True)
                .sel(cumulant=cumulant, drop=True)
                .dropna(dim="zt")
            )
            kwargs_ds = dict(kwargs)
            if line_colors is not None:
                kwargs_ds["color"] = line_colors[i_c]

            line2 = None
            if plot_type == "angles":
                line = _plot_angles_profile(ds=ds, ax=ax, p=p, **kwargs_ds)
                if add_asymmetry_markers:
                    _add_asymmetry_markers(ax=ax, ds=ds, color=line.get_color())

            elif plot_type == "scales":
                line, line2 = _plot_scales_profile(
                    ds=ds,
                    ax=ax,
                    p=p,
                    fill_between_alpha=fill_between_alpha,
                    **kwargs_ds,
                )

                if cumulant in scale_limits:
                    ax.set_xlim(0, scale_limits[cumulant])

            else:
                raise NotImplementedError(plot_type)
            lines.append(line)
            if line2:
                lines.append(line2)

        # set larger size since we're plotting subscripts with subscripts in
        # the cumulant name
        ax.set_title(fix_cumulant_name(cumulant), size=16.0)
        ax.set_ylabel(["height [m]", ""][i > 0])
        sns.despine()

    plt.tight_layout()
    if with_legend:
        lgd = plt.figlegend(
            lines, [l.get_label() for l in lines], loc="lower center", ncol=2
        )
        y_offset = 0.1 if line2 is None else 0.0
        fig.text(
            0.5,
            y_offset - data.dataset_name.count() * 0.1,
            " ",
            transform=fig.transFigure,
        )
        if add_asymmetry_markers:
            n = data.dataset_name.count()
            plt.figlegend(
                handles=_make_marker_legend(),
                title="asymmetry\nratio ($r_a$)",
                bbox_to_anchor=(1, 0),
                loc="lower left",
                bbox_transform=fig.transFigure,
            )
            fig.add_artist(lgd)

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
                plot(dataset, z_max=args.z_max, cumulants=cumulants, plot_type="angles")
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
                    plot(
                        dataset,
                        z_max=args.z_max,
                        cumulants=cumulants,
                        plot_type="plot_angles",
                    )
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
