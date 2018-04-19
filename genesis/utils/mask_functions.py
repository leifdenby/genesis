import os

import numpy as np
import xarray as xr

import skimage.morphology
import scipy.ndimage
from scipy.constants import pi


def w_pos(w_zt):
    return w_zt > 0.0
w_pos.description = "positive cell-centered vertical velocity"

def w_1(w_zt):
    return w_zt > 1.0
w_pos.description = "1 m/s vertical velocity"


def coldpool_coarse(tv0100):
    return tv0100 < -0.1
coldpool_coarse.description = 'coarse coldpool detection using -0.1K from mean of theta_v at 100m'


def moist_updrafts(q_flux):
    return q_flux > 0.3e-3
moist_updrafts.description = 'regions of vertical moisture flux greater than 0.3 m/s kg/kg'


L_SMOOTHING_DEFUALT = 1000.
L_EDGE_DEFAULT = 2000.
SHEAR_DIRECTION_Z_MAX_DEFAULT = 600.


def coldpool_edge(tv0100, l_smoothing=L_SMOOTHING_DEFUALT,
                  l_edge=L_EDGE_DEFAULT,):
    ds = xr.Dataset(coords=tv0100.coords)

    ds['coldpool_coarse'] = coldpool_coarse(tv0100=tv0100)

    # remove holes in coldpool mask
    print("Removing holes in coldpool mask")
    dx = np.max(np.gradient(ds.xt))
    nx_disk = int(l_smoothing/dx)
    selem = skimage.morphology.disk(nx_disk)
    ds['coldpool'] = (
        ds.coldpool_coarse.dims,
        skimage.morphology.closing(ds.coldpool_coarse, selem=selem)
    )
    ds.coldpool.attrs['smoothing_length'] = l_smoothing

    # make an edge mask of width `l_edge` centered on `ds.coldpool`'s edge
    print("Defining edge through dilation and erosion")
    dx = np.max(np.gradient(ds.xt))
    nx_disk = int(0.5*l_edge/dx)
    selem = skimage.morphology.disk(nx_disk)
    ds['m_inner'] = (
        ds.coldpool_coarse.dims,
        skimage.morphology.erosion(ds.coldpool, selem=selem)
    )
    ds['m_outer'] = (
        ds.coldpool_coarse.dims,
        skimage.morphology.dilation(ds.coldpool, selem=selem)
    )

    ds['coldpool_edge'] = (
        ds.coldpool.dims, 
        np.logical_and(ds.m_outer, ~ds.m_inner)
    )

    ds['l_smoothing'] = (
        (),
        l_smoothing,
        dict(units="m")
    )

    ds['l_edge'] = (
        (),
        l_edge,
        dict(units="m")
    )

    return ds
coldpool_edge.description = "Coldpool edge from theta_v"


def coldpool_edge_shear_direction_split(
    tv0100, ds_profile, l_smoothing=L_SMOOTHING_DEFUALT, l_edge=L_EDGE_DEFAULT,
    shear_calc_z_max=SHEAR_DIRECTION_Z_MAX_DEFAULT, profile_time_tolerance=60.
    ):
    """
    Computes a mask for the edge of coldpools in the upshear direction by
    comparing the direction of the coldpool edge to the mean shear (up to
    `shear_calc_z_max`)
    """
    ds_edge = coldpool_edge(
        tv0100=tv0100, l_smoothing=l_smoothing, l_edge=l_edge
    )

    ds = xr.Dataset(coords=ds_edge.coldpool.coords)

    ds['shear_calc_time_tolerance'] = (
        (),
        profile_time_tolerance,
        dict(units="s")
    )

    ds['shear_calc_z_max'] = (
        (),
        shear_calc_z_max,
        dict(units="m")
    )

    ds['l_edge'] = (
        (),
        l_edge,
        dict(units="m")
    )


    # make a stencil which will pick out only neighbouring cells
    dx = np.max(np.gradient(ds.xt))
    nx_disk = int(0.5*l_edge/dx)
    m_neigh = skimage.morphology.disk(nx_disk)
    m_neigh[nx_disk, nx_disk] = 0

    # convolve with stencil to count how many coldpool cells are near a
    # particular edge cell
    n_coldpool = np.where(
        ds_edge.coldpool_edge,
        scipy.ndimage.convolve(
            # cast to int here so we have range that bool doesn't supply
            ds_edge.coldpool.astype(int),
            m_neigh,
            mode='wrap'),
        np.nan
    )
    ds['n_coldpool'] = (ds_edge.coldpool.dims, n_coldpool)


    def _find_mean_dir(ds, x_):
        l_ = np.where(
            # only compute for cells which actually are "near" coldpool,
            # will depend on m_neigh size, should make sure n_neigh is big enough
            ds.n_coldpool > 0,
            # use positions for cells inside inner-most region of coldpool,
            # at the edge we ignore the points outside the domain
            # (working out wrapping with the positions is too hard for now...)
            scipy.ndimage.convolve(
                np.where(ds_edge.m_inner, x_, 0),
                m_neigh, mode='constant', cval=0.0
                ),
            np.nan
        )

        # from sum of mean of all neighbouring directions we subtract the
        # central position
        return l_/ds.n_coldpool - np.where(ds.n_coldpool > 0, x_, np.nan)

    x, y = np.meshgrid(ds.xt, ds.yt)

    print("Finding x-component of direction for edge")
    lx = _find_mean_dir(ds=ds, x_=x)
    print("Finding y-component of direction for edge")
    ly = _find_mean_dir(ds=ds, x_=y)

    print("Defining edge direction vector for each point")
    dims = tuple(['component',] + list(ds_edge.coldpool.dims))
    # use raw values for significant speedup
    ds['edge_direction'] = (dims, [lx.values, ly.values])
    ds.edge_direction.values /= np.linalg.norm(
        ds.edge_direction.values, axis=0
    )

    print("Identifying mean shear direction")
    time = ds.time.values
    p_sel = ds_profile.sel(time=time, method='nearest',
                           tolerance=profile_time_tolerance)

    u_wind, v_wind = p_sel.u.squeeze(), p_sel.v.squeeze()

    # have to select on u and v separately here because UCLALES (incorrectly)
    # labels v-wind as existing at the cell interface
    z_max = shear_calc_z_max
    dudz_mean = np.gradient(u_wind.sel(zt=slice(0, z_max))).mean()
    dvdz_mean = np.gradient(v_wind.sel(zm=slice(0, z_max))).mean()

    shear_dir = np.array([dudz_mean, dvdz_mean])
    shear_dir /= np.linalg.norm(shear_dir)

    ds['mean_shear_direction'] = np.arctan2(*shear_dir)*180./pi
    ds.mean_shear_direction.attrs['units'] = 'deg'


    # compute similarity in direction between shear and coldpool edge
    nx, ny = ds_edge.coldpool.shape
    co_dir = np.dot(
        shear_dir,
        ds.edge_direction.values.reshape((2, nx*ny))
    ).reshape((nx, ny))

    co_dir = np.where(ds_edge.coldpool_edge > 0, co_dir, np.nan)

    ds['coldpool_edge_upshear'] = (
        ds_edge.coldpool.dims,
        co_dir < 0.0,
        dict(longname='coldpool edge in upshear direction')
    )
    ds['coldpool_edge_downshear'] = (
        ds_edge.coldpool.dims,
        co_dir > 0.0,
        dict(longname='coldpool edge in downshear direction')
    )

    return ds
coldpool_edge_shear_direction_split.description = "Coolpool edge split into up- and downshear direction"
