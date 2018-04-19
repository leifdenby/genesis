import os

import numpy as np
import xarray as xr

import skimage.morphology
import scipy.ndimage
from scipy.constants import pi

def _aux_field(func):
    def func_wrapper(*args, **kwargs):
        aux_filename_base = kwargs.pop('aux_filename_base')
        fn_aux = aux_filename_base.format(func.__name__)
        ds_aux = None
        if os.path.exists(fn_aux):
            ds_aux = xr.open_dataset(fn_aux, decode_times=False)

            overwrite_file = False
            for (k, v) in kwargs.items():
                if not isinstance(v, xr.DataArray) and not isinstance(v, xr.Dataset):
                    if ds_aux[k] != v:
                        print("warning: aux field stored attribute {}={} "
                              "instead of {}={}".format(k, ds_aux[k].values,
                                                        k, v)
                             )
                        print("deleting aux field file")
                        overwrite_file = True
                        break

            if overwrite_file:
                ds_aux.close()
                os.remove(fn_aux)
                ds_aux = None

        # XXX: should verify attributes here against kwargs which aren't
        # xarray arrays or dataset
        if ds_aux is None:
            ds_aux = func(*args, **kwargs)
            ds_aux.to_netcdf(fn_aux)

        return ds_aux

    return func_wrapper




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


@_aux_field
def _find_coldpool_edge(tv0100, l_smoothing, l_edge):
    ds_aux = xr.Dataset(coords=tv0100.coords)

    ds_aux['coldpool_coarse'] = coldpool_coarse(tv0100=tv0100)

    print("Removing holes in coldpool mask")
    # remove holes in coldpool mask
    dx = np.max(np.gradient(ds_aux.xt))
    nx_disk = int(l_smoothing/dx)
    selem = skimage.morphology.disk(nx_disk)
    ds_aux['coldpool'] = (
        ds_aux.coldpool_coarse.dims,
        skimage.morphology.closing(ds_aux.coldpool_coarse, selem=selem)
    )
    ds_aux.coldpool.attrs['smoothing_length'] = l_smoothing

    print("Defining edge through dilation and erosion")
    # make an edge mask of width `l_edge` centered on `ds_aux.coldpool`'s edge
    dx = np.max(np.gradient(ds_aux.xt))
    nx_disk = int(0.5*l_edge/dx)
    selem = skimage.morphology.disk(nx_disk)
    ds_aux['m_inner'] = (
        ds_aux.coldpool_coarse.dims,
        skimage.morphology.erosion(ds_aux.coldpool, selem=selem)
    )
    ds_aux['m_outer'] = (
        ds_aux.coldpool_coarse.dims,
        skimage.morphology.dilation(ds_aux.coldpool, selem=selem)
    )

    ds_aux['coldpool_edge'] = (
        ds_aux.coldpool.dims, 
        np.logical_and(ds_aux.m_outer, ~ds_aux.m_inner)
    )

    ds_aux['l_smoothing'] = (
        (),
        l_smoothing,
        dict(units="m")
    )

    ds_aux['l_edge'] = (
        (),
        l_edge,
        dict(units="m")
    )

    return ds_aux


@_aux_field
def _split_coldpool_edge_based_on_shear_direction(
        ds_edge, ds_profile, profile_time_tolerance,
        l_edge, shear_calc_z_max
    ):

    ds_aux = xr.Dataset(coords=ds_edge.coldpool.coords)

    ds_aux['shear_calc_time_tolerance'] = (
        (),
        profile_time_tolerance,
        dict(units="s")
    )

    ds_aux['shear_calc_z_max'] = (
        (),
        shear_calc_z_max,
        dict(units="m")
    )

    ds_aux['l_edge'] = (
        (),
        l_edge,
        dict(units="m")
    )


    # make a stencil which will pick out only neighbouring cells
    dx = np.max(np.gradient(ds_aux.xt))
    nx_disk = int(0.5*l_edge/dx)
    m_neigh = skimage.morphology.disk(nx_disk)
    m_neigh[nx_disk,nx_disk] = 0

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
    ds_aux['n_coldpool'] = (ds_edge.coldpool.dims, n_coldpool)


    def _find_mean_dir(ds_aux, x_):
        l_ = np.where(
            # only compute for cells which actually are "near" coldpool,
            # will depend on m_neigh size, should make sure n_neigh is big enough
            ds_aux.n_coldpool > 0,
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
        return l_/ds_aux.n_coldpool - np.where(ds_aux.n_coldpool > 0, x_, np.nan)

    x, y = np.meshgrid(ds_aux.xt, ds_aux.yt)

    print("Finding x-component of direction for edge")
    lx = _find_mean_dir(ds_aux=ds_aux, x_=x)
    print("Finding y-component of direction for edge")
    ly = _find_mean_dir(ds_aux=ds_aux, x_=y)

    print("Defining edge direction vector for each point")
    dims = tuple(['component',] + list(ds_edge.coldpool.dims))
    # use raw values for significant speedup
    ds_aux['edge_direction'] = (dims, [lx.values, ly.values])
    ds_aux.edge_direction.values /= np.linalg.norm(
        ds_aux.edge_direction.values, axis=0
    )

    print("Identifying mean shear direction")
    time = ds_aux.time.values
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

    ds_aux['mean_shear_direction'] = np.arctan2(*shear_dir)*180./pi
    ds_aux.mean_shear_direction.attrs['units'] = 'deg'


    # compute similarity in direction between shear and coldpool edge
    nx, ny = ds_edge.coldpool.shape
    co_dir = np.dot(
        shear_dir,
        ds_aux.edge_direction.values.reshape((2, nx*ny))
    ).reshape((nx, ny))

    co_dir = np.where(ds_edge.coldpool_edge > 0, co_dir, np.nan)

    ds_aux['coldpool_edge_upshear'] = (ds_edge.coldpool.dims, co_dir < 0.0)
    ds_aux['coldpool_edge_downshear'] = (ds_edge.coldpool.dims, co_dir > 0.0)

    return ds_aux


L_SMOOTHING_DEFUALT = 1000.
L_EDGE_DEFAULT = 2000.
SHEAR_DIRECTION_Z_MAX_DEFAULT = 600.


def coldpool_edge(tv0100, aux_filename_base, l_smoothing=L_SMOOTHING_DEFUALT,
                  l_edge=2000.,):
    ds_aux = _find_coldpool_edge(
        tv0100=tv0100, aux_filename_base=aux_filename_base,
        l_smoothing=l_smoothing,
        l_edge=l_edge,
    )

    return ds_aux.coldpool_edge
coldpool_edge.description = "Coldpool edge from theta_v"


def coldpool_edge_upshear(tv0100, ds_profile, aux_filename_base, 
                          l_smoothing=L_SMOOTHING_DEFUALT,
                          l_edge=L_EDGE_DEFAULT,
                          z_max=SHEAR_DIRECTION_Z_MAX_DEFAULT,
                          profile_time_tolerance=60.):
    """
    Computes a mask for the edge of coldpools in the upshear direction by
    comparing the direction of the coldpool edge to the mean shear (up to
    `z_max`)
    """

    ds_edge = _find_coldpool_edge(
        tv0100=tv0100, aux_filename_base=aux_filename_base,
        l_smoothing=l_smoothing, l_edge=l_edge,
    )

    ds_aux = _split_coldpool_edge_based_on_shear_direction(
        aux_filename_base=aux_filename_base,
        ds_edge=ds_edge,
        l_edge=l_edge,
        profile_time_tolerance=profile_time_tolerance,
        shear_calc_z_max=z_max,
        ds_profile=ds_profile,
    )

    return ds_aux.coldpool_edge_upshear
coldpool_edge_upshear.description = "Coolpool edge in upshear direction"


def coldpool_edge_downshear(tv0100, ds_profile, aux_filename_base,
                            l_smoothing=L_SMOOTHING_DEFUALT,
                            l_edge=L_EDGE_DEFAULT,
                            z_max=SHEAR_DIRECTION_Z_MAX_DEFAULT,
                            profile_time_tolerance=60.):
    """
    Computes a mask for the edge of coldpools in the downshear direction by
    comparing the direction of the coldpool edge to the mean shear (up to
    `z_max`)
    """
    ds_edge = _find_coldpool_edge(
        tv0100=tv0100, aux_filename_base=aux_filename_base,
        l_smoothing=l_smoothing, l_edge=l_edge,
    )

    ds_aux = _split_coldpool_edge_based_on_shear_direction(
        aux_filename_base=aux_filename_base,
        ds_edge=ds_edge,
        l_edge=l_edge,
        profile_time_tolerance=profile_time_tolerance,
        shear_calc_z_max=z_max,
        ds_profile=ds_profile,
    )

    return ds_aux.coldpool_edge_downshear

coldpool_edge_downshear.description = "Coolpool edge in downshear direction"
