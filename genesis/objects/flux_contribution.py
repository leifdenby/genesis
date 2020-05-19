import seaborn as sns
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from ..utils.plot_types import adjust_fig_to_fit_figlegend


class PlotGrid():
    def __init__(self, height=6, ratio=5, space=.4,
                 dropna=True, width=None, xlim=None, ylim=None, size=None):
        # Set up the subplot grid
        try:
            ratio_x, ratio_y = ratio
        except TypeError:
            ratio_x = ratio_y = ratio
        if width is None:
            width = height
        f = plt.figure(figsize=(width, height))
        gs = plt.GridSpec(ratio_y + 1, ratio_x + 1)

        ax_joint = f.add_subplot(gs[1:, :-1])
        ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
        ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)

        self.fig = f
        self.ax_joint = ax_joint
        self.ax_marg_x = ax_marg_x
        self.ax_marg_y = ax_marg_y

        # Turn off tick visibility for the measure axis on the marginal plots
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)
        plt.setp(ax_marg_y.get_xticklabels(), rotation=30.)
        ax_marg_y.get_xaxis().set_major_locator(plt.MaxNLocator(2))

        # Turn off the ticks on the density axis for the marginal plots
        #plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
        #plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
        #plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
        #plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
        #plt.setp(ax_marg_x.get_yticklabels(), visible=False)
        #plt.setp(ax_marg_y.get_xticklabels(), visible=False)
        #ax_marg_x.yaxis.grid(False)
        #ax_marg_y.xaxis.grid(False)
        
        sns.despine(f)
        sns.despine(ax=ax_marg_x, left=False)
        sns.despine(ax=ax_marg_y, bottom=False)
        f.tight_layout()
        f.subplots_adjust(hspace=space, wspace=space)


def plot(ds, x, v, dx, mean_profile_components='all'):
    def _only_finite(vals):
        return vals[~np.logical_or(np.isnan(vals), np.isinf(vals))]

    x_min, x_max = _only_finite(ds[x]).min(), _only_finite(ds[x]).max()
    if dx is None:
        bins = np.linspace(x_min, x_max, 10)
    else:
        bins = np.arange(
            math.floor(x_min/dx)*dx,
            math.ceil(x_max/dx)*dx,
        dx)

    nx = ds.nx
    ny = ds.ny

    bin_var = "{}__sum".format(v)

    bin_centers = 0.5*(bins[1:] + bins[:-1])
    #fig, axes = plt.subplots(ncols=2, nrows=2, sharex="col", sharey="row", figsize=(10,6))
    g = PlotGrid(ratio=(3, 5), height=6, width=7)
    ds_ = ds.groupby_bins(x, bins=bins, labels=bin_centers)
    da_flux_per_bin = ds_.sum(dim='object_id', dtype=np.float64)[bin_var]/(nx*ny)
    if len(da_flux_per_bin["{}_bins".format(x)]) < 2:
        raise Exception("Please set a smaller bin size on `{x}`, currently"
                        "there's only one bin with the range of values in "
                        "`{x}`".format(x=x))
    da_flux_per_bin.plot(y='zt', ax=g.ax_joint, add_colorbar=False, robust=True)

    Nobj_bin_counts, _, _= ds[x].plot.hist(bins=bins, histtype='step',
                                           ax=g.ax_marg_x)
    g.ax_marg_x.set_ylabel('num objects [1]')
    g.ax_marg_x.set_yscale('log')
    p10_max = np.ceil(np.log10(np.max(Nobj_bin_counts)))
    g.ax_marg_x.yaxis.set_ticks(10.0**np.arange(0., p10_max))
    g.ax_marg_x.set_ylim(1, 10.0**p10_max)

    ref_var = "{}__mean".format(v)
    def scale_flux(da_flux):
        if da_flux.sampling == 'full domain':
            return da_flux
        else:
            return da_flux*ds.areafrac.sel(sampling=da_flux.sampling)
    da_flux_tot = ds[ref_var].groupby('sampling').apply(scale_flux)
    if mean_profile_components != "all":
        da_flux_tot = da_flux_tot.sel(sampling=mean_profile_components)
    da_flux_tot = da_flux_tot.sortby('sampling', ascending=False)
    da_flux_tot.attrs['long_name'] = 'horz. mean flux'
    da_flux_tot.attrs['units'] = ds[bin_var].units
    da_flux_tot.plot(y='zt', ax=g.ax_marg_y,
                     hue="sampling", add_legend=True)
    g.ax_marg_y.get_legend().set_bbox_to_anchor([1.0, 1.2])

    # da_inobject_mean_flux = ds.sum(dim='object_id', dtype=np.float64)[bin_var]/(nx*ny)
    # da_inobject_mean_flux.plot(y='zt', ax=g.ax_marg_y, marker='.')

    if v == 'qv_flux':
        g.ax_marg_y.xaxis.set_ticks([0., 0.02, 0.04])
        g.ax_marg_y.set_xlim(0., 0.05)

    g.ax_joint.set_title('')
    g.ax_marg_x.set_title('')
    g.ax_marg_y.set_title('')
    g.ax_marg_x.set_xlabel('')
    g.ax_marg_y.set_ylabel('')
    g.ax_joint.set_xlabel(xr.plot.utils.label_from_attrs(ds[x]))

    add_newline = lambda s: s[:s.index('[')] + "\n" + s[s.index('['):]
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
    ds.flux_mean.attrs['long_name'] = (
        r"$\overline{{w'{}'}}$".format(s_latex)
    )
    ds.flux_mean.plot(ax=ax, y='z', hue='sampling')
    ax.set_xlim(0, None)

    ax = axes[1]
    da_areafrac_procent = 100.0*ds.areafrac
    da_areafrac_procent.attrs['units'] = "%"
    da_areafrac_procent.attrs['long_name'] = "area fraction"
    da_areafrac_procent.plot(ax=ax, y='z', hue='sampling')

    ax = axes[2]
    def scale_flux(da_flux):
        if da_flux.sampling == 'full domain':
            return da_flux
        else:
            return da_flux*ds.areafrac.sel(sampling=da_flux.sampling)
    da_flux_tot = ds.flux_mean.groupby('sampling').apply(scale_flux)
    da_flux_tot.attrs['long_name'] = (
        r"$\sigma\ \overline{{w'{}'}}$".format(s_latex)
    )
    da_flux_tot.attrs['units'] = ds.flux_mean.units
    g = da_flux_tot.plot(ax=ax, y='z', hue='sampling')

    sns.despine(fig)
    [ax.get_legend().remove() for ax in axes]
    [ax.set_title('') for ax in axes]
    [ax.set_ylim(0, None)]
    [ax.set_ylabel('') for ax in axes[1:]]

    if ds.scalar == 'qv':
        axes[0].set_xlim(0, 1200)
        axes[2].set_xlim(0, 150)

    plt.tight_layout()

    # ax.get_legend_handles_labels is empty with xarray...
    # handles, labels = ax.get_legend_handles_labels()
    hue_label = "sampling"
    handles = ax.get_lines()
    labels = ds[hue_label].values
    figlegend = fig.legend(handles, labels, loc='center right', title=hue_label,
                           ncol=legend_ncols)

    adjust_fig_to_fit_figlegend(fig=fig, figlegend=figlegend, direction='right')

    return fig, axes
