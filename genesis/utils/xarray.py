import itertools
import functools
from pathlib import Path

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
    if fname:
        p = Path(path) / fname
        if p.exists():
            ds = xr.open_dataset(str(p))

    if ds is None:
        ds = func(*args, **kwargs)

    if fname:
        p = Path(path) / fname
        if not p.exists():
            ds.to_netcdf(str(p))

    return ds
