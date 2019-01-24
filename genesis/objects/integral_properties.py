import numpy as np
import xarray as xr


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
