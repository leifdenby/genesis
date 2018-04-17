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


def _split_coldpool_inner_outer(ds, l_edge):
    dx = np.max(np.gradient(ds.xt))
    nx_disk = int(0.5*l_edge/dx)
    selem = skimage.morphology.disk(nx_disk)
    m_inner = skimage.morphology.erosion(ds.coldpool, selem=selem)
    m_outer = skimage.morphology.dilation(ds.coldpool, selem=selem)

    return m_inner, m_outer


def coldpool_edge(tv0100, l_smoothing=1000., l_edge=2000.):
    ds = xr.merge([tv0100,])

    ds['coldpool_coarse'] = coldpool_coarse(tv0100=tv0100)

    # remove holes in coldpool mask
    dx = np.max(np.gradient(ds.xt))
    nx_disk = int(l_smoothing/dx)
    selem = skimage.morphology.disk(nx_disk)
    ds['coldpool'] = (
        ds.coldpool_coarse.dims,
        skimage.morphology.closing(ds.coldpool_coarse, selem=selem)
    )
    ds.coldpool.attrs['smoothing_length'] = l_smoothing

    # make an edge mask of width `l_edge` centered on `ds.coldpool`'s edge
    m_inner, m_outer = _split_coldpool_inner_outer(ds=ds, l_edge=l_edge)
    ds['coldpool_edge'] = (ds.coldpool.dims, np.logical_and(m_outer, ~m_inner))
    ds.coldpool_edge.attrs['edge_width'] = l_edge

    return ds.coldpool_edge


def _find_upshear_downshear_coldpool_edge(
        tv0100, ds_profile, l_smoothing=1000., l_edge=2000., z_max=600.,
        profile_time_tolerance=60.
    ):
    ds = xr.merge([tv0100,])

    print("Defining coldpool edge")
    ds['coldpool_edge'] = coldpool_edge(
        tv0100=tv0100,
        l_smoothing=l_smoothing,
        l_edge=l_edge
    )

    # make a stencil which will pick out only neighbouring cells
    dx = np.max(np.gradient(ds.xt))
    nx_disk = int(0.5*l_edge/dx)
    m_neigh = skimage.morphology.disk(nx_disk)
    m_neigh[nx_disk,nx_disk] = 0

    # convolve with stencil to count how many coldpool cells are near a
    # particular edge cell
    n_coldpool = np.where(
        ds.coldpool_edge,
        scipy.ndimage.convolve(
            # cast to int here so we have range that bool doesn't supply
            ds.coldpool.astype(int),
            m_neigh,
            mode='wrap'),
        np.nan
    )
    ds['n_coldpool'] = (ds.coldpool.dims, n_coldpool)

    m_inner, _ = _split_coldpool_inner_outer(ds=ds, l_edge=l_edge)
    def _find_mean_dir(ds, x_):
        l_ = np.where(
            # only compute for cells which actually are "near" coldpool,
            # will depend on m_neigh size, should make sure n_neigh is big enough
            ds.n_coldpool > 0,
            # use positions for cells inside inner-most region of coldpool,
            # at the edge we ignore the points outside the domain
            # (working out wrapping with the positions is too hard for now...)
            scipy.ndimage.convolve(np.where(m_inner, x_, 0), m_neigh, mode='constant', cval=0.0),
            np.nan
        )

        # from sum of mean of all neighbouring directions we subtract the
        # central position
        return l_/ds.n_coldpool - np.where(ds.n_coldpool > 0, x_, 0)

    x, y = np.meshgrid(ds.xt, ds.yt)

    print("Finding x-component of direction for edge")
    ds['lx'] = (ds.coldpool.dims, _find_mean_dir(ds=ds, x_=x))
    print("Finding y-component of direction for edge")
    ds['ly'] = (ds.coldpool.dims, _find_mean_dir(ds=ds, x_=y))

    dims = tuple(['component',] + list(ds.tv0100.dims))
    ds['edge_direction'] = (dims, [ds.lx, ds.ly])
    ds.edge_direction.values /= np.linalg.norm(
        ds.edge_direction.values, axis=0
    )

    time = ds.time.values
    p_sel = ds_profile.sel(time=time, method='nearest',
                           tolerance=profile_time_tolerance)

    u_wind, v_wind = p_sel.u.squeeze(), p_sel.v.squeeze()

    # have to select on u and v separately here because UCLALES (incorrectly)
    # labels v-wind as existing at the cell interface
    dudz_mean = np.gradient(u_wind.sel(zt=slice(0, z_max))).mean()
    dvdz_mean = np.gradient(v_wind.sel(zm=slice(0, z_max))).mean()

    shear_dir = np.array([dudz_mean, dvdz_mean])
    shear_dir /= np.linalg.norm(shear_dir)

    # compute similarity in direction between shear and coldpool edge
    nx, ny = ds.tv0100.shape
    co_dir = np.dot(
        shear_dir,
        ds.edge_direction.values.reshape((2, nx*ny))
    ).reshape((nx, ny))

    ds['co_dir'] = (
        ds.tv0100.dims,
        np.where(ds.coldpool_edge > 0, co_dir, np.nan)
    )

    ds.attrs['mean_shear_direction'] = np.arctan2(*shear_dir)*180./pi

    ds['coldpool_edge_upshear'] = ds.co_dir > 0.0
    ds['coldpool_edge_downshear'] = ds.co_dir < 0.0

    return ds


def coldpool_edge_upshear22(tv0100, ds_profile, l_smoothing=1000., l_edge=2000.,
                          z_max=600., profile_time_tolerance=60.):
    """
    Computes a mask for the edge of coldpools in the upshear direction by
    comparing the direction of the coldpool edge to the mean shear (up to
    `z_max`)
    """
    ds = _find_upshear_downshear_coldpool_edge(
        tv0100=tv0100, ds_profile=ds_profile, l_smoothing=l_smoothing,
        l_edge=l_edge, z_max=z_max,
        profile_time_tolerance=profile_time_tolerance
    )

    return ds.coldpool_edge_upshear
coldpool_edge_upshear22.description = "Coolpool edge in upshear direction"


def coldpool_edge_downshear(tv0100, ds_profile, l_smoothing=1000., l_edge=2000.,
                          z_max=600., profile_time_tolerance=60.):
    """
    Computes a mask for the edge of coldpools in the downshear direction by
    comparing the direction of the coldpool edge to the mean shear (up to
    `z_max`)
    """
    ds = _find_downshear_downshear_coldpool_edgecoldpool_edge_downshear(
        tv0100=tv0100, ds_profile=ds_profile, l_smoothing=l_smoothing,
        l_edge=l_edge, z_max=z_max,
        profile_time_tolerance=profile_time_tolerance
    )

    return ds.coldpool_edge_downshear
coldpool_edge_downshear.description = "Coolpool edge in downshear direction"
