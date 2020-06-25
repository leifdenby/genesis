from collections import Counter
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate

try:
    # reduce moved in py3
    from functools import reduce
except ImportError:
    pass


def _find_bin_on_percentile(q, bin_counts, debug_plot=False):
    """
    Find value for percentile by ranking all bins and through linear
    interpolation on the cumulative sum of points finding the bin
    count which would match the percentile requested
    """
    bc = np.sort(bin_counts.flatten())

    y = np.cumsum(bc)/np.sum(bc)
    x = np.arange(len(bc))/len(bc)

    y, idxs_unique = np.unique(y, return_index=True)
    x = x[idxs_unique]
    bc = bc[idxs_unique]

    idxs_nearest = np.sort(np.argsort(np.abs(y - q/100.))[:2])

    # y = ax+b
    # y1 - y0 = (x1 - x0)a
    # a = (y1 - y0)/(x1 - x0)
    x0, x1 = x[idxs_nearest]
    y0, y1 = y[idxs_nearest]
    a = (y1 - y0)/(x1 - x0)

    # y1 = a*x1 + b
    # b = y1 - a*x1
    b = y1 - a*x1

    # y2 = a*x2 + b
    # x2 = (y2 - b)/a
    x_fit = (q/100. - b)/a

    fn = interpolate.interp1d(x, bc, fill_value="extrapolate")
    v = fn(x_fit)

    if debug_plot:
        ax = plt.gcf().axes[0]
        ax.plot(x, y, marker='.')
        ax.plot(x[idxs_nearest], y[idxs_nearest], marker='x')
        ax.plot(x, a*x+b)
        ax.axvline(x_fit)

        ax.set_ylim(0., 1.1)
        ax.set_xlim(0.5, None)

        ax = plt.gcf().axes[1]
        ax.plot(x, bc, marker='.')
        ax.axhline(v)
        ax.axvline(x_fit)
        ax.set_xlim(0.5, None)

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

    # add bins at edges we are certain will have zero counts, this will ensure
    # that when we contour around bins later that there are enough sample
    # points that the contouring will produce closed contours
    dx = (x_range[1] - x_range[0])/bins
    dy = (y_range[1] - y_range[0])/bins
    x_range = (x_range[0] - dx, x_range[1] + dx)
    y_range = (y_range[0] - dy, y_range[1] + dy)
    bins += 2

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

        linestyles = [":", "--", "-"]
        if len(normed_levels) > len(linestyles):
            raise NotImplementedError
        else:
            linestyles = linestyles[-len(normed_levels):]

        cnt = ax.contour(x_, y_, bin_counts, levels=levels, linestyles=linestyles, **kwargs)

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


def multi_jointplot(x, y, z, ds, joint_type='pointhist', lgd_ncols=1, **kwargs):
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
        if joint_type == 'pointhist':
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
        elif joint_type == 'scatter':
            _ = g.ax_joint.scatter(ds_[x], ds_[y], color=c, alpha=0.5, marker='.')
        elif joint_type == 'kde':
            _ = g.ax_joint.scatter(ds_[x], ds_[y], color=c, alpha=0.5, marker='.')
            sns.kdeplot(ds_[x], ds_[y], cmap=cmap, ax=g.ax_joint, n_levels=5)
        else:
            raise NotImplementedError(joint_type)
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
        bbox_to_anchor=[0.5, -0.25], loc="lower center", ncol=lgd_ncols,
    )

    return g

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


def adjust_fig_to_fit_figlegend(fig, figlegend, direction='bottom'):
    # Draw the plot to set the bounding boxes correctly
    fig.draw(fig.canvas.get_renderer())

    if direction == 'right':
        # Calculate and set the new width of the figure so the legend fits
        legend_width = figlegend.get_window_extent().width / fig.dpi
        figure_width = fig.get_figwidth()
        fig.set_figwidth(figure_width + legend_width)

        # Draw the plot again to get the new transformations
        fig.draw(fig.canvas.get_renderer())

        # Now calculate how much space we need on the right side
        legend_width = figlegend.get_window_extent().width / fig.dpi
        space_needed = legend_width / (figure_width + legend_width) + 0.02
        # margin = .01
        # _space_needed = margin + space_needed
        right = 1 - space_needed

        # Place the subplot axes to give space for the legend
        fig.subplots_adjust(right=right)
    elif direction == 'top':
        # Calculate and set the new width of the figure so the legend fits
        legend_height = figlegend.get_window_extent().height / fig.dpi
        figure_height = fig.get_figheight()
        fig.set_figheight(figure_height + legend_height)

        # Draw the plot again to get the new transformations
        fig.draw(fig.canvas.get_renderer())

        # Now calculate how much space we need on the right side
        legend_height = figlegend.get_window_extent().height / fig.dpi
        space_needed = legend_height / (figure_height + legend_height) + 0.02
        # margin = .01
        top = 1 - space_needed

        # Place the subplot axes to give space for the legend
        fig.subplots_adjust(top=top)
    elif direction == 'bottom':
        # Calculate and set the new width of the figure so the legend fits
        legend_height = figlegend.get_window_extent().height / fig.dpi
        figure_height = fig.get_figheight()
        fig.set_figheight(figure_height + legend_height)

        # Draw the plot again to get the new transformations
        fig.draw(fig.canvas.get_renderer())

        # Now calculate how much space we need on the right side
        legend_height = figlegend.get_window_extent().height / fig.dpi
        space_needed = legend_height / (figure_height + legend_height) + 0.02
        # margin = .01
        bottom = 2*space_needed

        # Place the subplot axes to give space for the legend
        fig.subplots_adjust(bottom=bottom)
    else:
        raise NotImplementedError(direction)
