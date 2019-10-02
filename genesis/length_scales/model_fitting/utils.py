"""
utilities for making statistical fits of models to minkowski scales
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def cdf(v, ax=None):
    N = len(v)
    y = np.arange(1, N+1)/N
    x = np.sort(v)#[::-1]
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, marker='.')
    ax.set_title('ecdf')

def rank(v, ax):
    y = np.sort(v)
    x = np.arange(len(y))
    ax.plot(x, y, marker='.')
    ax.set_title('rank')

def fixed_bin_hist(v, dv, ax, **kwargs):
    if dv is None:
        nbins = None
        vrange = None
    else:
        vmin = np.floor(v.min()/dv)*dv
        vmax = np.ceil(v.max()/dv)*dv
        nbins = int((vmax-vmin)/dv)
        vrange = (vmin, vmax)
    ax.hist(v, range=vrange, bins=nbins, **kwargs)

def dist_plot(v, dv_bin, fit=None, axes=None, log_dists=True,
              components='default', **kwargs):
    da_v = None
    if isinstance(v, xr.DataArray):
        da_v = v
        v = da_v.values

    if components == 'default':
        components = ['hist', 'pdf', 'ecfd', 'rank']

    assert len(components) == len(axes)

    Nc = len(components)
    if axes is None:
        fig, axes = plt.subplots(ncols=Nc, figsize=(4*Nc, 4))
    else:
        fig = axes[0].figure

    iax = 0
    if 'hist' in components:
        ax = axes[iax]
        iax += 1
        fixed_bin_hist(v=v, dv=dv_bin, ax=ax, density=False, **kwargs)
        ax.set_title('hist')

    if 'pdf' in components:
        ax = axes[iax]
        iax += 1
        fixed_bin_hist(v=v, dv=dv_bin, ax=ax, density=True, **kwargs)
        ax.set_title('pdf')

        if fit:
            if fit[0] == 'exp':
                beta, vrange_fit = fit[1:]
                beta_std = None
                if type(beta) == tuple:
                    beta, beta_std = beta
                Ntot = len(v)
                C = np.exp(vrange_fit[0]/beta)
                v_ = np.linspace(*vrange_fit, 100)
                ax.plot(v_, C/beta*np.exp(-v_/beta), color='red')
                ax.axvline(beta+vrange_fit[0], linestyle='--', color='red')
                if da_v is not None:
                    units = da_v.units
                else:
                    units = ''
                if beta_std is None:
                    ax.text(0.9, 0.3, r"$\beta={:.0f}{}$".format(beta, units),
                            transform=ax.transAxes, horizontalalignment='right')
                else:
                    ax.text(0.9, 0.3, r"$\beta={:.0f}\pm{:.0f}{}$".format(
                                beta, beta_std, units),
                            transform=ax.transAxes, horizontalalignment='right')
            else:
                raise NotImplementedError(fit)

    if 'ecfd' in components:
        ax = axes[iax]
        iax += 1
        cdf(v, ax=ax)

    if 'rank' in components:
        ax = axes[iax]
        iax += 1
        rank(v, ax=ax)

    if log_dists:
        [ax.set_yscale('log') for ax in axes[:2]]

    if da_v is not None:
        xr_lab = xr.plot.utils.label_from_attrs
        labels = [
            'hist' in components and (xr_lab(da_v), 'num objects [1]'),
            'pdf' in components and (xr_lab(da_v), 'density [1/{}]'.format(da_v.units)),
            'ecfd' in components and (xr_lab(da_v), 'fraction of objects'),
            'rank' in components and ('object num', xr_lab(da_v)),
        ]

        labels = filter(lambda v: v, labels)

        for ax, (xl, yl) in zip(axes, labels):
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)

    return fig, axes
