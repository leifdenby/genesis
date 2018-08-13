import numpy as np


def _find_bin_on_percentile(q, bin_counts):
    bc = bin_counts.flatten()
    sort_indx = np.argsort(bc)

    bin_counts_sorted = bc[sort_indx]

    bin_count_cum = np.cumsum(bin_counts_sorted)/np.sum(bin_counts)

    i = np.argmin(np.abs(q/100. - bin_count_cum))

    v = bc[sort_indx[i]]

    return v


def joint_hist_contoured(xd, yd, bins=None, normed_levels=None, ax=None,
                         **kwargs):
    """
    Create joint histogram with contour levels at `normed_levels` percentiles
    """
    x_range = (np.nanmin(xd), np.nanmax(xd))
    y_range = (np.nanmin(yd), np.nanmax(yd))

    if bins is None:
        import operator
        n = reduce(operator.mul, xd.shape, 1)
        bins = (float(n))**(1./4.)

    bin_counts, x_bins, y_bins = np.histogram2d(
        xd, yd, bins=bins, range=(x_range, y_range)
    )
    x_c = 0.5*(x_bins[1:] + x_bins[:-1])
    y_c = 0.5*(y_bins[1:] + y_bins[:-1])

    x_, y_ = np.meshgrid(x_c, y_c, indexing='ij')

    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    if normed_levels is not None:
        levels = [
            _find_bin_on_percentile(bin_counts=bin_counts, q=q)
            for q in normed_levels
        ]
        cnt = ax.contour(x_, y_, bin_counts, levels=levels, **kwargs)
    else:
        cnt = ax.contour(x_, y_, bin_counts, **kwargs)
    return (x_, y_), bin_counts, cnt
