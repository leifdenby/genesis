from collections import Counter
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # reduce moved in py3
    from functools import reduce
except ImportError:
    pass


def _find_bin_on_percentile(q, bin_counts):
    bc = bin_counts.flatten()
    sort_indx = np.argsort(bc)

    bin_counts_sorted = bc[sort_indx]

    bin_count_cum = np.cumsum(bin_counts_sorted)/np.sum(bin_counts)

    i = np.argmin(np.abs(q/100. - bin_count_cum))

    v = bc[sort_indx[i]]

    return v

class JointHistPlotError(Exception):
    pass

def _estimate_bin_count(xd):
    import operator
    n = reduce(operator.mul, xd.shape, 1)
    bins = int((float(n))**(1./4.))
    return bins

def _raw_calc_joint_hist(xd, yd, bins=None):
    x_range = (np.nanmin(xd), np.nanmax(xd))
    y_range = (np.nanmin(yd), np.nanmax(yd))

    if np.any([np.isnan(x_range), np.isnan(y_range)]):
        raise JointHistPlotError

    if bins is None:
        bins = _estimate_bin_count(xd=xd)

    bin_counts, x_bins, y_bins = np.histogram2d(
        xd, yd, bins=bins, range=(x_range, y_range)
    )

    x_c = 0.5*(x_bins[1:] + x_bins[:-1])
    y_c = 0.5*(y_bins[1:] + y_bins[:-1])

    return (x_c, y_c), bin_counts


def calc_joint_hist(xd, yd, bins=None):
    if isinstance(xd, xr.DataArray):
        (x_c, y_c), bin_counts = _raw_calc_joint_hist(
            xd=xd.values.flatten(),
            yd=yd.values.flatten(),
            bins=bins
        )

        da__x_c = xr.DataArray(name=xd.name, dims=(xd.name,), data=x_c,
                               attrs=xd.attrs)
        da__y_c = xr.DataArray(name=yd.name, dims=(yd.name,), data=y_c,
                               attrs=yd.attrs)

        da = xr.DataArray(
            data=bin_counts, name='bin_counts', dims=(xd.name, yd.name),
            coords={ xd.name: da__x_c, yd.name: da__y_c }
        )

        return da
    else:
        return _raw_calc_joint_hist(xd=xd, yd=yd, bins=bins)


def joint_hist_contoured(xd, yd, bins=None, normed_levels=None, ax=None,
                         **kwargs):
    """
    Create joint histogram with contour levels at `normed_levels` percentiles
    """
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    (x_c, y_c), bin_counts = _raw_calc_joint_hist(xd=xd, yd=yd, bins=bins)
    x_, y_ = np.meshgrid(x_c, y_c, indexing='ij')

    if 'plot_hist2d' in kwargs:
        ax.pcolormesh(x_bins, y_bins, bin_counts)
        cnt = None
    elif normed_levels is not None:
        levels = [
            _find_bin_on_percentile(bin_counts=bin_counts, q=q)
            for q in normed_levels
        ]

        cnt = ax.contour(x_, y_, bin_counts, levels=levels, **kwargs)

        # attempt of adapting the number of bins below so that we get exactly
        # one line per contour. This should reduce some of the messiness when
        # we're using very few datapoints
        if all([len(seg) == 1 for seg in cnt.allsegs]):
            pass
        elif any([len(seg) == 0 for seg in cnt.allsegs]):
            pass
        else:
            if bins is None:
                bins = _estimate_bin_count(xd)
            bins = int(bins*0.95)
            for col in cnt.collections:
                col.remove()
            return joint_hist_contoured(
                xd=xd, yd=yd, bins=bins, normed_levels=normed_levels, ax=ax,
                **kwargs
            )
    else:
        cnt = ax.contour(x_, y_, bin_counts, **kwargs)

    return (x_, y_), bin_counts, cnt


def _get_counts(x, y):
    c = Counter([tuple(v) for v in zip(x, y)])
    counts = c.items()
    x_, y_ = zip(*c.keys())
    vals = np.array(list(c.values())).astype(int)

    return (np.array(x_), np.array(y_)), vals


def make_marker_plot(x, y, scale=100., ax=None, marker='o', **kwargs):
    """
    Like a scatter point only instead of overplotting points at the same
    location the marker size is increased.
    """
    if ax is None:
        ax = plt.gca()
    
    if not len(x.shape) == 1:
        raise Exception("Only pass in 1D arrays")

    pts, vals = _get_counts(x, y)
    x_, y_ = pts
    z_ = vals

    s = np.argsort(z_)
    x_, y_, z_ = x_[s], y_[s], z_[s]

    s = z_/np.nanmax(z_)*scale

    sc = ax.scatter(x_, y_, s=s, marker=marker, edgecolors='none', **kwargs)
    color = sc.get_facecolor()

    # idxs = [np.argmin(np.abs(np.percentile(z_, q) - z_)) for q in [5, 50, 95]]

    idxs = [np.argmax(z_), np.argmin(np.abs(z_ - np.median(z_))), np.argmin(z_)]
    idxs = set(idxs)

    for i in idxs:
        ax.scatter(x_[i],y_[i],s=s[i], vmin=0,vmax=1,edgecolors='none',
                   label='{:d}'.format(z_[i]), marker=marker, color=color,
                   **kwargs)
    # add the legend
    ax.legend(scatterpoints=1, handletextpad=1.0, labelspacing=scale/500., loc='upper right')

    return ax


def multi_jointplot(x, y, z, ds, **kwargs):
    """
    like seaborn jointplot but making it possible to plot for multiple (up to
    four) datasets
    """
    sns.set_color_codes()
    colors = ['b', 'r', 'g', 'orange']
    cmaps = ['Blues', 'Reds', 'Greens', 'Oranges']

    if np.any(np.isinf(ds[x])) or np.any(np.isinf(ds[y])):
        ds = ds.copy()
        ds = ds.where(~np.logical_or(np.isinf(ds[x]), np.isinf(ds[y])))
        warnings.warn("inf values will be filtered out")

    xlim = kwargs.get('xlim', np.array([ds[x].min(), ds[x].max()]))
    ylim = kwargs.get('ylim', np.array([ds[y].min(), ds[y].max()]))
    bins_x = 20
    bins_y = 10

    z_values = ds[z].values

    if len(z_values) > len(cmaps):
        raise NotImplementedError('Need to add some more colourmaps to handle'
                                  ' this number of datasets')

    g = sns.JointGrid(x=x, y=y, data=ds.sel(**{z: z_values}), **kwargs)
    for c, cmap, z_ in zip(colors, cmaps, z_values):
        ds_ = ds.sel(**{z:z_}).dropna(dim='object_id')
        def point_hist():
            bin_counts, pts = np.histogramdd([ds_[x], ds_[y]],
                bins=(bins_x, bins_y), range=(xlim,ylim)
            )
            x_e, y_e = pts
            s = 1500./np.sum(bin_counts)
            center = lambda x_: 0.5*(x_[1:] + x_[:-1])
            x_c, y_c = np.meshgrid(center(pts[0]), center(pts[1]), indexing='ij')
            assert bin_counts.shape == x_c.shape
            g.ax_joint.scatter(x_c, y_c, s=s*bin_counts, marker='o', color=c,
                               alpha=0.6)
        point_hist()
        # _ = g.ax_joint.scatter(ds_[x], ds_[y], color=c, alpha=0.5, marker='.')
        # sns.kdeplot(ds_[x], ds_[y], cmap=cmap, ax=g.ax_joint, n_levels=5)
        _ = g.ax_marg_x.hist(ds_[x], alpha=.6, color=c, range=xlim, density=True, bins=bins_x)
        _ = g.ax_marg_y.hist(ds_[y], alpha=.6, color=c, orientation="horizontal", range=ylim, density=True, bins=bins_y)

    g.ax_joint.set_xlabel(xr.plot.utils.label_from_attrs(ds[x]))
    g.ax_joint.set_ylabel(xr.plot.utils.label_from_attrs(ds[y]))

    LABEL_FORMAT = "{name}: {count} objects"
    g.ax_joint.legend(
        labels=[LABEL_FORMAT.format(
            name=z_, 
            count=int(ds.sel(**{z:z_}).dropna(dim='object_id').object_id.count())
            ) for z_ in z_values],
        bbox_to_anchor=[0.5, -0.4], loc="lower center"
    )

    return g
