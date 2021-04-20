"""
Utilities for working on 2D projected fields
"""
import numba
import numpy as np
import xarray as xr


@numba.jit
def _replace(a, values, other=np.nan):
    nx, ny = a.shape
    b = other * np.ones_like(a)
    for i in range(nx):
        for j in range(ny):
            b[i, j] = values[a[i, j]]
    return b


def make_filled(da_labels, da_property, object_coord="cloud_id", other=np.nan):
    """
    Make a filled 2D array where for each object in `da_labels` the label index
    is replaced by the value in `da_property`. Where `da_property` doesn't contain a
    label the value will be set to `other`
    """
    N_max = int(da_labels.max())
    values = other * np.ones((N_max + 1))
    values[da_property[object_coord].astype(int).values] = da_property.values

    a = da_labels.fillna(0).astype(int).values
    replaced_values = _replace(a, numba.typed.List(values), other=other)
    return xr.DataArray(
        replaced_values,
        dims=da_labels.dims,
        coords=da_labels.coords,
        attrs=da_property.attrs,
    )


def create_mask_from_object_set(ds_tracking, object_set, t0):
    """
    Create a 2D mask from labels in `ds` at time `t0` by only including
    objects in `object_set` which are present at t0
    """
    da_is_present = object_set.get_value("present", t0=t0)
    da_labels = ds_tracking[f"nr{object_set.object_type}"].sel(time=t0)
    return make_filled(da_labels, da_is_present, object_coord=f"object_id")


def extract_from_3d_at_heights_in_2d(da_3d, z_2d):
    """
    Extract from 3D data array at heights given in `z_2d`
    """
    z_unique = np.unique(z_2d)
    z_unique = z_unique[~np.isnan(z_unique)]
    v = xr.concat([da_3d.sel(zt=z_).where(z_2d == z_) for z_ in z_unique], dim="zt")
    return v.max(dim="zt")
