import seaborn as sns
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
import re
import textwrap

from ..utils.plot_types import adjust_fig_to_fit_figlegend, PlotGrid
from ..utils.xarray import scalar_density_2d, _make_equally_spaced_bins


def _label_from_attrs(da, width=20):
    long_name = da.attrs["long_name"]
    label = "\n".join(textwrap.wrap(long_name, width=width))

    units = f"[{da.units}]"
    if len(label.split("\n")[-1]) > width:
        label += f"\n{units}"
    else:
        label += f" {units}"
    return label


def plot(
    ds,
    x,
    v,
    dx,
    domain_num_cells,
    v_scaling="robust",
    mean_profile_components="all",
    include_x_mean=True,
    include_height_profile=True,
    add_profile_legend=True,
    add_height_histogram=True,
    fig_width=7.0,
    fig_height=None,
):  # noqa
    """
    Using values in `ds` plot a decomposition with height (assumed to be `zt`)
    into the total contribution (scaled by domain size) per bin of the object
    property `x` (with bin-width `dx`) to the scalar variable `v`

    This means that with for example for `v=qv_flux` that summing across all
    bins at a given height will give horizontal mean moisture flux of all objects.

    Mean profiles are assumed to be named `{v}__mean` and the total
    contribution to `v` for each object at a height `zt` is given by
    `{v}__sum`.
    """

    bins = _make_equally_spaced_bins(ds[x], dx=dx)

    bin_var = f"{v}__sum"
    # this will be used on all variables, so lets set it to something simpler
    # here
    ds[f"{v}__mean"].attrs["long_name"] = "horz. mean\nvertical flux"
    # make height more meaningful
    ds.zt.attrs["long_name"] = "altitude"

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # fig, axes = plt.subplots(ncols=2, nrows=2, sharex="col", sharey="row", figsize=(10,6))
    fig_height = fig_width - 1.0 if not include_x_mean else fig_width + 0.5
    g = PlotGrid(
        ratio=4,
        height=fig_height,
        width=fig_width,
        extra_x_marg=True,
        extra_y_marg=add_height_histogram,
    )
    ds_ = ds.groupby_bins(x, bins=bins, labels=bin_centers)
    da_flux_per_bin = (
        ds_.sum(dim="object_id", dtype=np.float64)[bin_var] / domain_num_cells
    )
    # put in zeroes so that we don't get empty parts of the plots
    da_flux_per_bin = da_flux_per_bin.fillna(0.0)
    if len(da_flux_per_bin["{}_bins".format(x)]) < 2:
        raise Exception(
            "Please set a smaller bin size on `{x}`, currently"
            "there's only one bin with the range of values in "
            "`{x}`".format(x=x)
        )
    da_flux_per_bin.attrs.update(ds[f"{v}__mean"].attrs)
    dist_kws = dict(add_colorbar=False, robust=True)
    if v_scaling == "robust":
        dist_kws["robust"] = True
    elif len(v_scaling) == 2:
        dist_kws["vmin"] = v_scaling[0]
        dist_kws["vmax"] = v_scaling[1]
    pc = da_flux_per_bin.plot(y="zt", ax=g.ax_joint, **dist_kws)
    box = g.ax_joint.get_position()
    cb_pad, cb_height = 0.08, 0.02
    cax = g.fig.add_axes(
        [box.xmin, box.ymin - cb_pad - cb_height, box.width, cb_height]
    )
    # make pointed ends to the colorbar because we used "robust=True" above
    cb = g.fig.colorbar(pc, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(_label_from_attrs(da_flux_per_bin, width=30))

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
    g.ax_marg_y.set_xlabel(_label_from_attrs(da_flux_tot, width=16))

    # make the underlying dataarray available later
    setattr(g.ax_marg_y, "_source_data", da_flux_tot)

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
    g.ax_marg_x2.set_ylabel(_label_from_attrs(da_flux_per_bin_mean, width=16))

    if add_height_histogram:
        # add a histogram of the number of objects contributing to the flux at
        # each height
        # count per height the number of objects which contribute to the moisture flux
        N_objs_per_z = (
            xr.ones_like(ds.object_id)
            .where(ds[f"{v}__sum"] != 0, 0)
            .sum(dim="object_id")
            .sel(zt=slice(0.0, None))
        )
        ax = g.ax_marg_y2
        N_objs_per_z.plot.step(ax=ax, y="zt", color=objects_line_color)
        ax.set_xlabel("num objects\ncontributing [1]")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_xlim(0, None)

    # if we have a "all objects" sampling then there will also be one which is
    # the filtered one, we add a few vertical lines to guide the eye to these

    if v == "qv_flux":
        g.ax_marg_y.xaxis.set_ticks([0.0, 0.02, 0.04])
        g.ax_marg_y.set_xlim(0.0, 0.05)

    # XXX: quick hack to remove the height profile plot, we really should have
    # a different way of defining what kind of marginal plots we want
    # if not include_height_profile:
    # g.ax_marg_y.set_visible(False)
    # g.fig.set_size_inches(4, 7)

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

    return g


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


def make_2d_decomposition(ds, z, x, y, v, domain_num_cells, method, dx=None, dy=None):
    """
    Decomposition of variable `v` at a given height `z` with variables `x`
    and `y` in dataset `ds`
    """
    if z == "all":
        scaling = 1.0 / int(ds.zt.count())
        ds_ = ds.sum(dim="zt", keep_attrs=True)
    else:
        scaling = 1.0
        ds_ = ds.sel(zt=z, method="nearest")

    nbins_default = 10
    if dx is None:
        dx = (ds_[x].max() - ds_[x].min()) / nbins_default
    if dy is None:
        dy = (ds_[y].max() - ds_[y].min()) / nbins_default

    da_source = ds_[f"{v}__sum"]

    ds_[f"{v}__domaintot"] = da_source / domain_num_cells * scaling
    regex = r"sum\sof\s(?P<long_name>[\w\s]+)\sper\sobject"
    base_long_name = re.match(regex, da_source.long_name).groupdict()["long_name"]
    ds_[f"{v}__domaintot"].attrs[
        "long_name"
    ] = f"contribution to domain mean {base_long_name}"
    if z == "all":
        ds_[f"{v}__domaintot"].attrs["long_name"] += " height mean"
    else:
        ds_[f"{v}__domaintot"].attrs["long_name"] += f" at z={z}m"
    ds_[f"{v}__domaintot"].attrs["units"] = da_source.units

    da_sd = scalar_density_2d(
        ds=ds_,
        x=x,
        y=y,
        v=f"{v}__domaintot",
        dx=dx,
        dy=dy,
        drop_nan_and_inf=True,
        method=method,
    )

    return da_sd


def plot_2d_decomposition(ds, z, x, y, v, domain_num_cells, method, dx=None, dy=None):
    """
    Plot decomposition of variable `v` at a given height `z` with variables `x`
    and `y` in dataset `ds`
    """

    da_sd = make_2d_decomposition(
        ds=ds,
        z=z,
        x=x,
        y=y,
        v=v,
        domain_num_cells=domain_num_cells,
        dx=dx,
        dy=dy,
        method=method,
    )
    da_sd.plot()

    return plt.gca()
