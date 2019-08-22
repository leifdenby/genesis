"""
utilities for making statistical fits of models to minkowski scales
"""
import numpy as np
import matplotlib.pyplot as plt

def cdf(v, ax=None):
    y = len(v)/(np.arange(len(v))+1.)
    x = np.sort(v)[::-1]
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, marker='.')
    ax.set_title('cdf')

def rank(v, ax):
    y = np.sort(v)
    x = np.arange(len(y))
    ax.plot(x, y, marker='.')
    ax.set_title('rank')

def fixed_bin_hist(v, dv, ax, **kwargs):
    vmin = np.floor(v.min()/dv)*dv
    vmax = np.ceil(v.max()/dv)*dv
    nbins = int((vmax-vmin)/dv)
    ax.hist(v, range=(vmin, vmax), bins=nbins, **kwargs)

def dist_plot(v, dv_bin, fit=None, axes=None, log_dists=True, **kwargs):
    if axes is None:
        fig, axes = plt.subplots(ncols=4, figsize=(16, 4))
    else:
        fig = axes[0].figure
    ax = axes[0]
    fixed_bin_hist(v=v, dv=dv_bin, ax=ax, density=False, **kwargs)
    ax.set_title('hist')

    ax = axes[1]
    fixed_bin_hist(v=v, dv=dv_bin, ax=ax, density=True, **kwargs)
    ax.set_title('pdf')

    if fit:
        if fit[0] == 'exp':
            beta, vrange_fit = fit[1:]
            Ntot = len(v)
            C = np.exp(vrange_fit[0]/beta)
            v_ = np.linspace(*vrange_fit, 100)
            ax.plot(v_, C/beta*np.exp(-v_/beta), color='red')
        else:
            raise NotImplementedError(fit)

    ax = axes[2]
    cdf(v, ax=ax)

    ax = axes[3]
    rank(v, ax=ax)

    if log_dists:
        [ax.set_yscale('log') for ax in axes[:2]]
    return fig, axes
