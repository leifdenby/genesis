import numpy as np
import xarray as xr


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


def _raw_calc_joint_hist(xd, yd, bins=None):
    x_range = (np.nanmin(xd), np.nanmax(xd))
    y_range = (np.nanmin(yd), np.nanmax(yd))

    if np.any([np.isnan(x_range), np.isnan(y_range)]):
        raise JointHistPlotError

    if bins is None:
        import operator
        n = reduce(operator.mul, xd.shape, 1)
        bins = int((float(n))**(1./4.))

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
    else:
        cnt = ax.contour(x_, y_, bin_counts, **kwargs)

    return (x_, y_), bin_counts, cnt
