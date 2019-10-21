import seaborn as sns
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import math


class PlotGrid():
    def __init__(self, height=6, ratio=5, space=.4,
                 dropna=True, xlim=None, ylim=None, size=None):
        # Set up the subplot grid
        f = plt.figure(figsize=(height, height))
        gs = plt.GridSpec(ratio + 1, ratio + 1)

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


def plot(ds, x, v, nx, ny, dx, da_prof_ref):
    if dx is None:
        bins = np.linspace(ds[x].min(), ds[x].max(), 10)
    else:
        bins = np.arange(
            math.floor(ds[x].min()/dx)*dx,
            math.ceil(ds[x].max()/dx)*dx,
        dx)

    bin_centers = 0.5*(bins[1:] + bins[:-1])
    #fig, axes = plt.subplots(ncols=2, nrows=2, sharex="col", sharey="row", figsize=(10,6))
    g = PlotGrid()
    ds_ = ds.groupby_bins(x, bins=bins, labels=bin_centers)
    da_flux_per_bin = ds_.sum(dim='object_id', dtype=np.float64)[v]/(nx*ny)
    da_flux_per_bin.plot(y='zt', ax=g.ax_joint, add_colorbar=False, robust=True)

    # ds.r_equiv.plot.hist(bins=bins, histtype='step', ax=g.ax_marg_x)
    Nobj_bin_counts, _, _= ds[x].plot.hist(bins=bins, histtype='step',
                                           ax=g.ax_marg_x)
    g.ax_marg_x.set_ylabel('num objects [1]')
    g.ax_marg_x.set_yscale('log')
    p10_max = np.ceil(np.log10(np.max(Nobj_bin_counts)))
    g.ax_marg_x.yaxis.set_ticks(10.0**np.arange(0., p10_max))
    g.ax_marg_x.set_ylim(1, 10.0**p10_max)

    da_prof_ref.plot(y='zt', ax=g.ax_marg_y)
    da_inobject_mean_flux = ds.sum(dim='object_id', dtype=np.float64)[v]/(nx*ny)
    da_inobject_mean_flux.attrs['long_name'] = 'horz. mean flux'
    da_inobject_mean_flux.attrs['units'] = ds[v].units
    da_inobject_mean_flux.plot(y='zt', ax=g.ax_marg_y, marker='.')

    if v == 'qv_flux__sum':
        g.ax_marg_y.xaxis.set_ticks([0., 0.01, 0.02])
        g.ax_marg_y.set_xlim(0., 0.03)

    g.ax_joint.set_title('')
    g.ax_marg_x.set_title('')
    g.ax_marg_y.set_title('')
    g.ax_marg_x.set_xlabel('')
    g.ax_marg_y.set_ylabel('')
    g.ax_joint.set_xlabel(xr.plot.utils.label_from_attrs(ds[x]))

    return g.ax_joint
