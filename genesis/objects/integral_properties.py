import numpy as np
import xarray as xr

import dask_image.ndmeasure

VAR_MAPPINGS = dict(
    length_m="minkowski",
    width_m="minkowski",
    thickness_m="minkowski",
    num_cells="minkowski",
    volume="minkowski",
    theta="com_incline_and_orientation_angle",
    phi="com_incline_and_orientation_angle",
    x_c="centroid",
    y_c="centroid",
    z_c="centroid",
)

def calc_com_incline_and_orientation_angle(da_mask, plot_ax=None):
    """
    Calculate approximate shear angle of object (theta) and xy-orientation
    angle (phi) from the change of xy-position of the center-of-mass computed
    separately at every height
    """
    if np.any(da_mask.isnull()):
        m = ~da_mask.isnull()
    else:
        m = da_mask

    # need to center coordinates on "center of mass" (assuming constant density)
    if len(da_mask.x.shape) == 3:
        x_3d = da_mask.x
        y_3d = da_mask.y
        z_3d = da_mask.z
    else:
        x_3d, y_3d, z_3d = xr.broadcast(da_mask.x, da_mask.y, da_mask.z)

    # compute mean xy-position at every height z, this is the effective
    # centre-of-mass
    kws = dict(dtype='float64', dim=('x', 'y'))
    x_c = x_3d.where(m).mean(**kws)  # other=nan so that these get excluded from mean calculation
    y_c = y_3d.where(m).mean(**kws)

    if plot_ax:
        x_c.plot(y='z', ax=ax)
        ds.mask.sel(y=0, method='nearest').plot(y='z', ax=ax, rasterized=True)

    try:
        dx = np.gradient(x_c)
        dy = np.gradient(y_c)

        dx_mean = np.nanmean(dx)
        dy_mean = np.nanmean(dy)

        dl_mean = np.sqrt(dx_mean**2. + dy_mean**2.)
        dz_mean = np.nanmean(np.gradient(x_c.z))

        theta = np.arctan2(dz_mean, dl_mean)
        phi = np.arctan2(dy_mean, dx_mean)
    except ValueError:
        phi = theta = np.nan

    return xr.merge([
        xr.DataArray(phi, name='phi', attrs=dict(long_name='xy-plane angle', units='rad')),
        xr.DataArray(theta, name='theta', attrs=dict(long_name='z-axis slope angle', units='rad')),
    ])


def calc_xy_proj_length(da_mask):
    if np.any(da_mask.isnull()):
        m = ~da_mask.isnull()
    else:
        m = da_mask

    if len(da_mask.x.shape) == 3:
        x_3d = da_mask.x
        y_3d = da_mask.y
    else:
        x_3d, y_3d, _ = xr.broadcast(da_mask.x, da_mask.y, da_mask.z)

    x_min, x_max = x_3d.where(m).min(), x_3d.where(m).max()
    y_min, y_max = y_3d.where(m).min(), y_3d.where(m).max()

    lx = x_max - x_min
    ly = y_max - y_min

    l = np.sqrt(lx**2. + ly**2.)
    l.attrs['long_name'] = 'xy-projected length'
    l.attrs['units'] = x_3d.units
    return l


def calc_z_proj_length(da_mask):
    if np.any(da_mask.isnull()):
        m = ~da_mask.isnull()
    else:
        m = da_mask

    if len(da_mask.x.shape) == 3:
        x_3d = da_mask.x
        y_3d = da_mask.y
    else:
        _, _, z_3d = xr.broadcast(da_mask.x, da_mask.y, da_mask.z)

    z_min, z_max = z_3d.where(m).min(), z_3d.where(m).max()

    l = z_max - z_min
    l.attrs['long_name'] = 'z-projected length'
    l.attrs['units'] = z_3d.units
    return l

def calc_z_max(da_mask):
    if len(da_mask.x.shape) == 3:
        x_3d = da_mask.x
        y_3d = da_mask.y
    else:
        _, _, z_3d = xr.broadcast(da_mask.x, da_mask.y, da_mask.z)

    z_max = z_3d.where(m).max()
    z_max.attrs['long_name'] = 'max height'
    z_max.attrs['units'] = z_3d.units
    return z_max


def calc_z_max__dask(da_objs):
    if len(da_objs.x.shape) == 3:
        x_3d = da_objs.x
        y_3d = da_objs.y
    else:
        _, _, z_3d = xr.broadcast(da_objs.z, da_objs.y, da_objs.x)

    assert z_3d.shape == da_objs.shape
    assert z_3d.dims == da_objs.dims

    idx = np.unique(da_objs)[1:]
    z_max_vals = dask_image.ndmeasure.maximum(z_3d, da_objs, idx).compute()

    z_max = xr.DataArray(data=z_max_vals, coords=[idx], dims=['object_id'])
    z_max.attrs['long_name'] = 'max height'
    z_max.attrs['units'] = z_3d.units
    return z_max

def calc_z_proj_length__dask(da_objs):
    if len(da_objs.x.shape) == 3:
        x_3d = da_objs.x
        y_3d = da_objs.y
    else:
        _, _, z_3d = xr.broadcast(*[da_objs[d] for d in da_objs.dims])

    assert z_3d.shape == da_objs.shape
    assert z_3d.dims == da_objs.dims

    idx = np.unique(da_objs)[1:]
    z_max_vals = dask_image.ndmeasure.maximum(z_3d, da_objs, idx).compute()
    z_min_vals = dask_image.ndmeasure.minimum(z_3d, da_objs, idx).compute()

    l_vals = z_max_vals - z_min_vals

    l = xr.DataArray(data=l_vals, coords=[idx], dims=['object_id'])
    l.attrs['long_name'] = 'z-projected length'
    l.attrs['units'] = z_3d.units
    return l

def calc_z_min__dask(da_objs):
    if len(da_objs.x.shape) == 3:
        x_3d = da_objs.x
        y_3d = da_objs.y
    else:
        _, _, z_3d = xr.broadcast(*[da_objs[d] for d in da_objs.dims])

    idx = np.unique(da_objs)[1:]
    z_min_vals = dask_image.ndmeasure.minimum(z_3d, da_objs, idx).compute()

    assert z_3d.shape == da_objs.shape
    assert z_3d.dims == da_objs.dims

    z_min = xr.DataArray(data=z_min_vals, coords=[idx], dims=['object_id'])
    z_min.attrs['long_name'] = 'min height'
    z_min.attrs['units'] = z_3d.units
    return z_min

def calc_centroid__dask(da_objs):
    if len(da_objs.x.shape) == 3:
        x_3d = da_objs.x
        y_3d = da_objs.y
    else:
        x_3d, y_3d, z_3d = xr.broadcast(da_objs.z, da_objs.y, da_objs.x)

    assert z_3d.shape == da_objs.shape
    assert z_3d.dims == da_objs.dims

    idx = np.unique(da_objs)[1:]
    x_c_vals = dask_image.ndmeasure.mean(x_3d, da_objs, idx)
    y_c_vals = dask_image.ndmeasure.mean(y_3d, da_objs, idx)
    z_c_vals = dask_image.ndmeasure.mean(z_3d, da_objs, idx)

    x_c = xr.DataArray(data=x_c_vals, coords=[idx], dims=['object_id'])
    x_c.attrs['long_name'] = 'centroid x-position'
    x_c.attrs['units'] = x_3d.units
    x_c.name = "x_c"

    y_c = xr.DataArray(data=y_c_vals, coords=[idx], dims=['object_id'])
    y_c.attrs['long_name'] = 'centroid y-position'
    y_c.attrs['units'] = y_3d.units
    y_c.name = "y_c"

    z_c = xr.DataArray(data=z_c_vals, coords=[idx], dims=['object_id'])
    z_c.attrs['long_name'] = 'centroid z-position'
    z_c.attrs['units'] = z_3d.units
    z_c.name = "z_c"

    ds = xr.merge([x_c, y_c, z_c])

    return ds

def calc_xy_proj_length__dask(da_objs):
    x_3d, y_3d, _ = xr.broadcast(da_objs.x, da_objs.y, da_objs.z)

    idx = np.unique(da_objs)[1:]
    x_min = dask_image.ndmeasure.minimum(x_3d, da_objs, idx)
    x_max = dask_image.ndmeasure.maximum(x_3d, da_objs, idx)
    y_min = dask_image.ndmeasure.minimum(y_3d, da_objs, idx)
    y_max = dask_image.ndmeasure.maximum(y_3d, da_objs, idx)

    lx = x_max - x_min
    ly = y_max - y_min

    l = np.sqrt(lx**2. + ly**2.)
    da_l = xr.DataArray(data=l, coords=[idx], dims=['object_id'])
    da_l.attrs['long_name'] = 'xy-projected length'
    da_l.attrs['units'] = x_3d.units
    return da_l

def calc_vertical_flux__dask(da_objs, w, scalar):
    flux_per_cell = w*scalar

    idx = np.unique(da_objs)[1:]
    qflux_per_obj
    obj_flux = dask_image.ndmeasure.sum(flux_per_cell, da_objs, idx).compute()

    obj_flux = xr.DataArray(data=obj_flux_vals, coords=[idx], dims=['object_id'])
    obj_flux.attrs['long_name'] = 'min height'
    obj_flux.attrs['units'] = "{} {}".format(scalar.units, w.units)
    return obj_flux
