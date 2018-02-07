"""
Contains routines to analyse coherent structures in 2D using 2nd-order cumulant
analysis
"""
import numpy as np
from numpy import linalg
from scipy.constants import pi
from intergrid import intergrid
import xarray as xr


def calc_2nd_cumulant(v1, v2):
    """
    Calculate 2nd-order cumulant of v1 and v2 in Fourier space
    """
    assert v1.shape == v2.shape
    Nx, Ny = v1.shape

    V1 = np.fft.fft2(v1)
    if np.all(v1 == v2):
        V2 = V1
    else:
        V2 = np.fft.fft2(v2)

    c_vv_fft = np.fft.ifft2(V1*V2.conjugate())

    # it's most handy to have this centered on (0,0)
    c_vv = c_vv_fft.real/(Nx*Ny)
    c_vv = np.roll(np.roll(c_vv, shift=Ny/2, axis=1), shift=Nx/2, axis=0)

    attrs = dict(
        longname="cumulant({},{})".format(v1.longname, v2.longname),
        units="{} {}".format(v1.units, v2.units))

    return xr.DataArray(c_vv, dims=v1.dims, coords=v1.coords, attrs=attrs)


def identify_principle_axis(x, y, C, sI_N=100):
    """
    Using 2nd-order cumulant identify principle axis of correlation in 2D.
    `sI_N` denotes window over which to look for maximum correlation
    """
    Nx, Ny = C.shape

    I_func = lambda x, y, m: np.array([
        [np.sum(m*y**2.), np.sum(m*x*y)],
        [np.sum(m*y*x),  np.sum(x**2.*m)]
    ])

    sI_win = slice(Nx/2 - sI_N/2, Nx/2 + sI_N/2), slice(Ny/2 - sI_N/2, Ny/2 + sI_N/2)

    I = I_func(x[sI_win], y[sI_win], C[sI_win])

    la, v = np.linalg.eig(I)

    theta = np.arctan2(v[0][1], v[0][0])
    
    return theta


def covariance_plot(v1, v2, s_N=200, extra_title=""):
    """
    Make a 2D covariance plot of v1 and v2 (both are expected to be
    xarray.DataArray) and identify principle axis. Covariance analysis is
    plotted over a window of s_N x s_N
    """
    import matplotlib.pyplot as plot

    assert v1.shape == v2.shape
    Nx, Ny = v1.shape

    assert np.all(v1.coords['x'] == v2.coords['x'])
    assert np.all(v1.coords['y'] == v2.coords['y'])

    x, y = np.meshgrid(v1.coords['x'], v1.coords['y'], indexing='ij')

    C_vv = calc_2nd_cumulant(v1, v2)
    theta = identify_principle_axis(x, y, C_vv, sI_N=100)

    s_win = slice(Nx/2 - s_N/2, Nx/2 + s_N/2), slice(Ny/2 - s_N/2, Ny/2 + s_N/2)
    x_, y_, C_vv_ = x[s_win], y[s_win], C_vv[s_win]
    plot.pcolormesh(x_, y_, C_vv_)
    plot.colorbar()

    ax = plot.gca()
    ax.set_aspect(1)
    #plot.xlim(-lx, lx)
    #plot.ylim(-lx, lx)
    plot.xlabel('x-distance [m]')
    plot.ylabel('y-distance [m]')

    fn_l = lambda x: np.tan(theta)*x
    s_line = slice(Nx/2 - s_N/2, Nx/2 + s_N/2)
    x__ = v1.coords['x'][s_line]

    plot.plot(x__, fn_l(x__), linestyle='--', color='red')
    plot.text(0.1, 0.1, r"$\theta_{{princip}}={:.2f}^{{\circ}}$"
              "".format(theta*180./pi + 180.), transform=ax.transAxes,
              color='red')

    if 'zt' in v1:
        z_var = 'zt'
    else:
        z_var = 'zm'

    plot.title(
        """Covariance length-scale
        for ({v1},{v2})
        t={t}{t_units} z={z}{z_units}
        """.format(v1=v1.longname, v2=v2.longname,
                   t=float(v1.time), t_units=v1.time.units,
                   z=float(v1[z_var]), z_units=v1[z_var].units)
    )

    return plot.gca()


def _line_sample(data, x, y, theta): 
    """
    Sample 2D dataset along a line define by points in (x,y)
    """
    x_ = x[:,0]
    y_ = y[0,:]

    x_l = x_
    y_l = np.tan(theta)*x_l

    maps = [x_, y_]
    lo = np.array([x_[0], y_[0]])
    hi = np.array([x_[-1], y_[-1]])

    interp_f = intergrid.Intergrid(data/data.max(), lo=lo, hi=hi, maps=maps, verbose=0)

    return x_l, interp_f(np.array([x_l, y_l]).T)*float(data.max())

def covariance_direction_plot(v1, v2, s_N=200):
    """
    Compute 2nd-order cumulant between v1 and v2 and sample and perpendicular
    to pricinple axis. `s_N` sets plot window
    """
    import matplotlib.pyplot as plot

    assert v1.shape == v2.shape
    Nx, Ny = v1.shape

    assert np.all(v1.coords['x'] == v2.coords['x'])
    assert np.all(v1.coords['y'] == v2.coords['y'])

    # TODO: don't actually need 2D coords here, but would have to fix indexing
    # below
    x, y = np.meshgrid(v1.coords['x'], v1.coords['y'], indexing='ij')

    C_vv = calc_2nd_cumulant(v1, v2)
    theta = identify_principle_axis(x, y, C_vv, sI_N=100)

    s_win = slice(Nx/2 - s_N/2, Nx/2 + s_N/2), slice(Ny/2 - s_N/2, Ny/2 + s_N/2)
    x_, y_, C_vv_ = x[s_win], y[s_win], C_vv[s_win]

    x_l, C_vv_l = _line_sample(data=C_vv_, x=x_, y=y_, theta=theta)
    plot.plot(x_l, C_vv_l, label=r'$\theta=\theta_{princip}$')
    x_l, C_vv_l = _line_sample(data=C_vv_, x=x_, y=y_, theta=theta+pi/2.)
    plot.plot(x_l, C_vv_l, label=r'$\theta=\theta_{princip} + 90^{\circ}$')

    plot.legend()
    plot.xlabel('distance [m]')
    plot.ylabel('covariance [{}]'.format(C_vv.units))
    ax = plot.gca()
    plot.text(0.05, 0.9, r"$\theta_{{princip}}={:.2f}^{{\circ}}$"
              "".format(theta*180./pi + 180.), transform=ax.transAxes)

    if 'zt' in v1:
        z_var = 'zt'
    else:
        z_var = 'zm'

    plot.title("{} sampled along and\n perpendicular to principle axis "
               "at z={z}{z_units}".format(
                   C_vv.longname, z=float(v1[z_var]), z_units=v1[z_var].units
               ))
