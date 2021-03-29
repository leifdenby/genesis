import itertools
import functools
from pathlib import Path

import scipy.stats
import numpy as np
import xarray as xr
import tqdm
import math


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


def _make_equally_spaced_bins(x, dx, n_pad=2):
    def _get_finite_range(vals):
        # turns infs into nans and then we can uses nanmax nanmin
        v = vals.where(~np.isinf(vals), np.nan)
        return np.nanmin(v), np.nanmax(v)

    x_min, x_max = _get_finite_range(x)
    bins = np.arange(
        (math.floor(x_min / dx) - n_pad) * dx,
        (math.ceil(x_max / dx) + n_pad) * dx + dx,
        dx,
    )

    return bins


def scalar_density_2d(ds, x, y, v, dx, dy, method="kde", drop_nan_and_inf=False):
    """
    Compute the 2D distribution (as a density) of the scalar `v` with the
    variables `x` and `y` with discretisation `dx`, `dy` so that the integral
    over `x` and `y` (with increments `dx` and `dy`) equals the sum of `v`

    `method` sets how the PDF of `v` is approximated:

        kde: compute a kernel density estimate to approximate the distribution
             of `v` with `x` and `y` and sample at resolution `dx` and `dy`

        hist: compute density by binning with bin-size `dx` and `dy`

    OBS: for `kde` method only positive values of `v` are currently considered
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

    if method == "kde":
        ds_ = ds_.where(ds_[v] > 0.0, drop=True)

    x_bins = _make_equally_spaced_bins(ds_[x], dx=dx)
    y_bins = _make_equally_spaced_bins(ds_[y], dx=dy)

    da_x = ds_[x]
    da_y = ds_[y]
    da_v = ds_[v]

    def center(x_):
        return 0.5 * (x_[1:] + x_[:-1])

    x_ = center(x_bins)
    y_ = center(y_bins)

    if method == "hist":
        # bin the values to compute a distribution of the scalar values `v`
        bins = (x_bins, y_bins)
        res = scipy.stats.binned_statistic_2d(
            da_x, da_y, da_v, statistic="sum", bins=bins
        ).statistic
    elif method == "kde":
        # use a kernel-density estimation to create a PDF for `v` as a function
        # of `x` and `y` and sample a discrete points separated by `dx` and
        # `dy`
        xx, yy = np.meshgrid(x_, y_, indexing="ij")

        # the PDF is calculated so that the integral (dx*dy) over the PDF is ~ 1.0
        pdf = scipy.stats.gaussian_kde(dataset=[da_x, da_y], weights=da_v)

        # sample the PDF
        res = pdf([xx.ravel(), yy.ravel()]).reshape((len(x_), len(y_)))
    else:
        raise NotImplementedError(method)

    # scale the values so that the integral over all x and y (with step
    # dxdy) equals the total quantity of the scalar, so that we produce the
    # "scalar density"
    arr_sd = res / res.sum() * da_v.sum().item() / (dx * dy)

    da_x_sd = xr.DataArray(x_, attrs=da_x.attrs, dims=(da_x.name), name=da_x.name)
    da_y_sd = xr.DataArray(y_, attrs=da_y.attrs, dims=(da_y.name), name=da_y.name)

    long_name = da_v.long_name.replace("per object", "")
    units = f"{da_v.units} / ({da_x.units} {da_y.units})"
    da_sd = xr.DataArray(
        arr_sd.T,
        dims=(da_y.name, da_x.name),
        coords={da_x.name: da_x_sd, da_y.name: da_y_sd},
        attrs=dict(long_name=long_name, units=units),
    )
    return da_sd
