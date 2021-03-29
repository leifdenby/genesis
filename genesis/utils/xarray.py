import itertools
import functools
from pathlib import Path

import scipy.stats
import numpy as np
import xarray as xr
import tqdm


def apply_all(ds, fn, dims=None, process_desc=None):
    """
    Use coordinate dims in ds to provide arguments to fn
    """
    if dims is None:
        dims = ds.coords.keys()

    args = list(itertools.product(*[ds[d].values for d in dims]))

    def process(**kwargs):
        da = fn(**kwargs)
        if da is not None:
            for k, v in kwargs.items():
                da.coords[k] = v
            da = da.expand_dims(list(kwargs.keys()))
        return da

    if process_desc:
        progress_fn = functools.partial(tqdm.tqdm, desc=process_desc)
    else:
        progress_fn = lambda v: v  # noqa

    data = [process(**dict(zip(dims, a))) for a in progress_fn(args)]

    if all([da is None for da in data]):
        return None
    else:
        return xr.merge(data).squeeze()


def cache_to_file(path, func, fname=None, *args, **kwargs):
    """
    If `fname` is not `None` cach the resuls of calling `func(*args, **kwargs)` to `path/fname`
    """
    ds = None
    if path and fname:
        p = Path(path) / fname
        if p.exists():
            ds = xr.open_dataset(str(p))

    if ds is None:
        ds = func(*args, **kwargs)

    if path and fname:
        p = Path(path) / fname
        if not p.exists():
            ds.to_netcdf(str(p))

    return ds


def binned_statistic_2d(ds, x, y, v, bins=10, statistic="sum", drop_nan_and_inf=False):
    """
    Bin variable `v` by values of variables `x` and `y` applying `statistic` to each bin
    """
    # we can't bin on values which are either nan or inf, so we filter those out here
    def should_mask(x_):
        return np.logical_or(np.isnan(x_), np.isinf(x_))

    values_mask = np.logical_or(
        should_mask(ds[x]), np.logical_or(should_mask(ds[y]), should_mask(ds[v]))
    )

    if drop_nan_and_inf:
        ds_ = ds.where(~values_mask, drop=True)
    else:
        if np.any(values_mask):
            raise Exception(
                "There are some inf or nan values, but we can't bin on these."
                " Set `drop_nan_and_inf=True` to filter these out"
            )
        else:
            ds_ = ds

    da_x = ds_[x]
    da_y = ds_[y]
    da_v = ds_[v]

    res = scipy.stats.binned_statistic_2d(
        da_x, da_y, da_v, statistic=statistic, bins=bins
    )

    def center(x_):
        return 0.5 * (x_[1:] + x_[:-1])

    da_x_bin = xr.DataArray(
        center(res.x_edge), attrs=da_x.attrs, dims=(da_x.name), name=da_x.name
    )

    da_y_bin = xr.DataArray(
        center(res.y_edge), attrs=da_y.attrs, dims=(da_y.name), name=da_y.name
    )

    da_binned = xr.DataArray(
        res.statistic.T,
        dims=(da_y.name, da_x.name),
        coords={da_x.name: da_x_bin, da_y.name: da_y_bin},
        attrs=dict(
            long_name=f"{statistic} per bin of {da_v.long_name}", units=da_v.units
        ),
    )
    return da_binned


def kde_weighted_dist_2d(ds, x, y, v, bins=10, statistic="sum", drop_nan_and_inf=False):
    """
    Kernel-density approximatation of normalized (sums to one, not integrates
    to one) distribution of variable `v` by values of variables `x` and `y`
    scaled by sum of `v` so that total sum remains the same
    """
    # we can't bin on values which are either nan or inf, so we filter those out here
    def should_mask(x_):
        return np.logical_or(np.isnan(x_), np.isinf(x_))

    values_mask = np.logical_or(
        should_mask(ds[x]), np.logical_or(should_mask(ds[y]), should_mask(ds[v]))
    )

    if drop_nan_and_inf:
        ds_ = ds.where(~values_mask, drop=True)
    else:
        if np.any(values_mask):
            raise Exception(
                "There are some inf or nan values, but we can't bin on these."
                " Set `drop_nan_and_inf=True` to filter these out"
            )
        else:
            ds_ = ds

    da_x = ds_[x]
    da_y = ds_[y]
    da_v = ds_[v]
    pdf = scipy.stats.gaussian_kde(dataset=[da_x, da_y], weights=da_v)  # noqa

    if type(bins) == int:
        nxbins = nybins = bins
        x_ = np.linspace(da_x.min(), da_x.max(), nxbins)
        y_ = np.linspace(da_y.min(), da_y.max(), nybins)
    elif len(bins) == 2 and np.atleast_1d(bins).dtype == int:
        nxbins, nybinx = bins
        x_ = np.linspace(da_x.min(), da_x.max(), nxbins)
        y_ = np.linspace(da_y.min(), da_y.max(), nybins)
    else:
        x_, y_ = bins
    xx, yy = np.meshgrid(x_, y_, indexing="ij")

    # sample the PDF
    kde_arr = pdf([xx.ravel(), yy.ravel()]).reshape((len(x_), len(y_)))

    da_x_kde = xr.DataArray(x_, attrs=da_x.attrs, dims=(da_x.name), name=da_x.name)
    da_y_kde = xr.DataArray(y_, attrs=da_y.attrs, dims=(da_y.name), name=da_y.name)

    # normalize the values so the sum is the same as of the input array
    kde_arr = kde_arr / np.sum(kde_arr) * da_v.sum().values

    da_kde = xr.DataArray(
        kde_arr.T,
        dims=(da_y.name, da_x.name),
        coords={da_x.name: da_x_kde, da_y.name: da_y_kde},
        attrs=dict(
            long_name=da_v.long_name.replace("per object", ""), units=da_v.units
        ),
    )
    return da_kde
