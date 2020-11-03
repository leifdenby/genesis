import seaborn as sns
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math

from ..utils.plot_types import adjust_fig_to_fit_figlegend


class PlotGrid:
    def __init__(
        self,
        height=6,
        ratio=5,
        space=0.4,
        dropna=True,
        width=None,
        xlim=None,
        ylim=None,
        size=None,
        extra_x_marg=False,
    ):
        # Set up the subplot grid
        try:
            ratio_x, ratio_y = ratio
        except TypeError:
            ratio_x = ratio_y = ratio
        if width is None:
            width = height
        f = plt.figure(figsize=(width, height))
        N_x_marg = 1 if not extra_x_marg else 2
        gs = plt.GridSpec(ratio_y + 1, ratio_x + N_x_marg)

        ax_joint = f.add_subplot(gs[N_x_marg:, :-1])
        ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
        ax_marg_y = f.add_subplot(gs[N_x_marg:, -1], sharey=ax_joint)

        if extra_x_marg:
            ax_marg_x2 = f.add_subplot(gs[1, :-1], sharex=ax_joint)

        self.fig = f
        self.ax_joint = ax_joint
        self.ax_marg_x = ax_marg_x
        self.ax_marg_y = ax_marg_y

        if extra_x_marg:
            self.ax_marg_x2 = ax_marg_x2

        # Turn off tick visibility for the measure axis on the marginal plots
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)
        plt.setp(ax_marg_y.get_xticklabels(), rotation=30.0)
        ax_marg_y.get_xaxis().set_major_locator(plt.MaxNLocator(2))

        if extra_x_marg:
            plt.setp(ax_marg_x2.get_xticklabels(), visible=False)

        # Turn off the ticks on the density axis for the marginal plots
        # plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
        # plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
        # plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
        # plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
        # plt.setp(ax_marg_x.get_yticklabels(), visible=False)
        # plt.setp(ax_marg_y.get_xticklabels(), visible=False)
        # ax_marg_x.yaxis.grid(False)
        # ax_marg_y.xaxis.grid(False)

        sns.despine(f)
        sns.despine(ax=ax_marg_x, left=False)
        sns.despine(ax=ax_marg_y, bottom=False)
        if extra_x_marg:
            sns.despine(ax=ax_marg_x2, left=False)

        f.tight_layout()
        f.subplots_adjust(hspace=space, wspace=space)


def plot(
    ds,
    x,
    v,
    dx,
    mean_profile_components="all",
    include_x_mean=True,
    add_profile_legend=True,
    fig_width=7.0,
):  # noqa
    def _get_finite_range(vals):
        # turns infs into nans and then we can uses nanmax nanmin
        v = vals.where(~np.isinf(vals), np.nan)
        return np.nanmin(v), np.nanmax(v)

    x_min, x_max = _get_finite_range(ds[x])
    if dx is None:
        bins = np.linspace(x_min, x_max, 10)
    else:
        bins = np.arange(math.floor(x_min / dx) * dx, math.ceil(x_max / dx) * dx, dx)

    nx = ds.nx
    ny = ds.ny

    bin_var = f"{v}__sum"
    # this will be used on all variables, so lets set it to something simpler
    # here
    ds[f"{v}__mean"].attrs["long_name"] = "horz. mean flux"
    # make height more meaningful
    ds.zt.attrs["long_name"] = "altitude"

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # fig, axes = plt.subplots(ncols=2, nrows=2, sharex="col", sharey="row", figsize=(10,6))
    fig_height = fig_width - 1.0 if not include_x_mean else fig_width + 0.5
    g = PlotGrid(ratio=(2, 5), height=fig_height, width=fig_width, extra_x_marg=True)
    ds_ = ds.groupby_bins(x, bins=bins, labels=bin_centers)
    da_flux_per_bin = ds_.sum(dim="object_id", dtype=np.float64)[bin_var] / (nx * ny)
    # put in zeroes so that we don't get empty parts of the plots
    da_flux_per_bin = da_flux_per_bin.fillna(0.0)
    if len(da_flux_per_bin["{}_bins".format(x)]) < 2:
        raise Exception(
            "Please set a smaller bin size on `{x}`, currently"
            "there's only one bin with the range of values in "
            "`{x}`".format(x=x)
        )
    da_flux_per_bin.attrs.update(ds[f"{v}__mean"].attrs)
    pc = da_flux_per_bin.plot(y="zt", ax=g.ax_joint, add_colorbar=False, robust=True)
    box = g.ax_joint.get_position()
    cb_pad, cb_height = 0.08, 0.02
    cax = g.fig.add_axes(
        [box.xmin, box.ymin - cb_pad - cb_height, box.width, cb_height]
    )
    # make pointed ends to the colorbar because we used "robust=True" above
    cb = g.fig.colorbar(pc, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(xr.plot.utils.label_from_attrs(da_flux_per_bin))

    # g.ax_joint.colorbar(pc, bbox_to_anchor=[0.0, -0.2], orientation='horizontal')

    ref_var = "{}__mean".format(v)

    def scale_flux(da_flux):
        if da_flux.sampling == "full domain":
            return da_flux
        else:
            return da_flux * ds.areafrac.sel(sampling=da_flux.sampling)

    da_flux_tot = ds[ref_var].groupby("sampling").apply(scale_flux)
    if str(mean_profile_components) != "all":
        da_flux_tot = da_flux_tot.sel(sampling=mean_profile_components)
    da_flux_tot = da_flux_tot.sortby("sampling", ascending=False)
    da_flux_tot.attrs.update(ds[f"{v}__mean"].attrs)
    lines_profile = da_flux_tot.plot(
        y="zt", ax=g.ax_marg_y, hue="sampling", add_legend=add_profile_legend
    )

    # add a square marker for the mask
    mask_marker = "s"
    mask_markersize = 4
    for n, (line, sampling) in enumerate(
        zip(g.ax_marg_y.get_lines(), da_flux_tot.sampling.values)
    ):
        if sampling == "mask":
            line.set_marker(mask_marker)
            line.set_markersize(mask_markersize)

            if add_profile_legend:
                lgd_line = g.ax_marg_y.get_legend().get_lines()[n]
                lgd_line.set_marker(mask_marker)
                lgd_line.set_markersize(mask_markersize)

    if add_profile_legend:
        lgd = g.ax_marg_y.get_legend()
        lgd.set_bbox_to_anchor([1.0, 1.2])
        lgd._loc = lgd.codes["center left"]

    # work out the color of the "all objects" or "objects" line from the mean
    # profile plot
    objects_line_color = None
    for line, sampling in zip(g.ax_marg_y.get_lines(), da_flux_tot.sampling.values):
        if sampling in ["objects", "all objects"]:
            objects_line_color = line.get_color()
            break

    if objects_line_color is None:
        raise Exception("Couldn't find the color for the objects mean profile")

    # number of objects distribution with x
    Nobj_bin_counts, _, _ = ds[x].plot.hist(
        bins=bins, histtype="step", ax=g.ax_marg_x, color=objects_line_color
    )
    g.ax_marg_x.set_ylabel("num objects [1]")
    g.ax_marg_x.set_yscale("log")
    p10_max = np.ceil(np.log10(np.max(Nobj_bin_counts)))
    g.ax_marg_x.yaxis.set_ticks(10.0 ** np.arange(0.0, p10_max))
    g.ax_marg_x.set_ylim(1, 10.0 ** p10_max)

    # aggregate flux contribution with x
    nz = da_flux_per_bin.zt.count()
    da_flux_per_bin_mean = da_flux_per_bin.sum(dim="zt", dtype=np.float64) / nz
    da_flux_per_bin_mean.attrs.update(ds[f"{v}__mean"].attrs)
    da_flux_per_bin_mean.plot(ax=g.ax_marg_x2, color=objects_line_color)
    g.ax_marg_x2.set_xlabel("")
    g.ax_marg_x2.set_title("")

    # if we have a "all objects" sampling then there will also be one which is
    # the filtered one, we add a few vertical lines to guide the eye to these

    if v == "qv_flux":
        g.ax_marg_y.xaxis.set_ticks([0.0, 0.02, 0.04])
        g.ax_marg_y.set_xlim(0.0, 0.05)

    g.ax_joint.set_title("")
    g.ax_marg_x.set_title("")
    g.ax_marg_y.set_title("")
    g.ax_marg_x.set_xlabel("")
    g.ax_marg_y.set_ylabel("")
    g.ax_joint.set_xlabel(xr.plot.utils.label_from_attrs(ds[x]))

    g.ax_joint.set_ylim(ds.where(ds.zt > 0, drop=True).zt.min(), None)

    def add_newline(s):
        return s[: s.index("[")] + "\n" + s[s.index("[") :]

    g.ax_marg_y.set_xlabel(add_newline(g.ax_marg_y.get_xlabel()))

    return g.ax_joint


def plot_with_areafrac(ds, figsize=(12, 8), legend_ncols=3):
    """
    Plot per-sampling region mean flux, area-fraction and
    total flux profiles
    """
    fig, axes = plt.subplots(ncols=3, figsize=figsize, sharey=True)

    SCALAR_TO_LATEX = dict(
        qv="q_v",
    )
    s_latex = SCALAR_TO_LATEX.get(ds.scalar, ds.scalar)

    ax = axes[0]
    ds.flux_mean.attrs["long_name"] = r"$\overline{{w'{}'}}$".format(s_latex)
    ds.flux_mean.plot(ax=ax, y="z", hue="sampling")
    ax.set_xlim(0, None)

    ax = axes[1]
    da_areafrac_procent = 100.0 * ds.areafrac
    da_areafrac_procent.attrs["units"] = "%"
    da_areafrac_procent.attrs["long_name"] = "area fraction"
    da_areafrac_procent.plot(ax=ax, y="z", hue="sampling")

    ax = axes[2]

    def scale_flux(da_flux):
        if da_flux.sampling == "full domain":
            return da_flux
        else:
            return da_flux * ds.areafrac.sel(sampling=da_flux.sampling)

    da_flux_tot = ds.flux_mean.groupby("sampling").apply(scale_flux)
    da_flux_tot.attrs["long_name"] = r"$\sigma\ \overline{{w'{}'}}$".format(s_latex)
    da_flux_tot.attrs["units"] = ds.flux_mean.units
    da_flux_tot.plot(ax=ax, y="z", hue="sampling")

    sns.despine(fig)
    [ax.get_legend().remove() for ax in axes]
    [ax.set_title("") for ax in axes]
    [ax.set_ylim(0, None)]
    [ax.set_ylabel("") for ax in axes[1:]]

    if ds.scalar == "qv":
        axes[0].set_xlim(0, 1200)
        axes[2].set_xlim(0, 150)

    plt.tight_layout()

    # ax.get_legend_handles_labels is empty with xarray...
    # handles, labels = ax.get_legend_handles_labels()
    hue_label = "sampling"
    handles = ax.get_lines()
    labels = ds[hue_label].values
    figlegend = fig.legend(
        handles, labels, loc="center right", title=hue_label, ncol=legend_ncols
    )

    adjust_fig_to_fit_figlegend(fig=fig, figlegend=figlegend, direction="right")

    return fig, axes
