"""
Contains routines to analyse coherent structures in 2D using 2nd-order cumulant
analysis
"""
import numpy as np
from numpy import linalg
from scipy.constants import pi
import scipy.optimize
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


def identify_principle_axis(C, sI_N=100):
    """
    Using 2nd-order cumulant identify principle axis of correlation in 2D.
    `sI_N` denotes window over which to look for maximum correlation
    """
    assert C.dims == ('x', 'y')
    Nx, Ny = C.shape
    x_ = C.coords['x']
    y_ = C.coords['y']

    I_func = lambda x, y, m: np.array([
        [np.sum(m*y**2.), np.sum(m*x*y)],
        [np.sum(m*y*x),  np.sum(x**2.*m)]
    ])

    sI_x = slice(Nx/2 - sI_N/2, Nx/2 + sI_N/2)
    sI_y = slice(Ny/2 - sI_N/2, Ny/2 + sI_N/2)

    x, y = np.meshgrid(x_[sI_x], y_[sI_y], indexing='ij')
    I = I_func(x, y, C[sI_x, sI_y])

    la, v = np.linalg.eig(I)

    theta = np.arctan2(v[0][1], v[0][0])

    # easier to work with positive angles
    if theta < 0.0:
        theta += pi

    return theta


def covariance_plot(v1, v2, s_N=200, extra_title="", theta_win_N=100):
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
    assert v1.dims == ('x', 'y')

    x, y = np.meshgrid(v1.coords['x'], v1.coords['y'], indexing='ij')

    C_vv = calc_2nd_cumulant(v1, v2)
    theta = identify_principle_axis(C_vv, sI_N=theta_win_N)

    s_win = slice(Nx/2 - s_N/2, Nx/2 + s_N/2), slice(Ny/2 - s_N/2, Ny/2 + s_N/2)
    x_, y_, C_vv_ = x[s_win], y[s_win], C_vv[s_win]
    plot.pcolormesh(x_, y_, C_vv_)
    plot.colorbar()

    ax = plot.gca()
    ax.set_aspect(1)
    plot.xlabel('x-distance [m]')
    plot.ylabel('y-distance [m]')

    fn_l = lambda x: np.tan(theta)*x
    s_line = slice(Nx/2 - s_N/2, Nx/2 + s_N/2)
    x__ = v1.coords['x'][s_line]

    plot.plot(x__, fn_l(x__), linestyle='--', color='red')
    plot.text(0.1, 0.1, r"$\theta_{{princip}}={:.2f}^{{\circ}}$"
              "".format(theta*180./pi), transform=ax.transAxes,
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

def _get_line_sample_func(data, theta):
    x = data.coords['x']
    y = data.coords['y']
    assert data.dims == ('x', 'y')

    d_max = data.values.max()

    maps = [x, y]
    lo = np.array([x[0], y[0]])
    hi = np.array([x[-1], y[-1]])
    interp_f = intergrid.Intergrid(data.values/d_max,
                                   lo=lo, hi=hi, maps=maps, verbose=0)

    def sample(x_l):
        y_l = np.tan(theta)*x_l
        return interp_f(np.array([x_l, y_l]).T)*d_max

    return sample


def _line_sample(data, theta, max_dist): 
    """
    Sample 2D dataset along a line define by points in (x,y)
    """
    sample = _get_line_sample_func(data, theta)

    x = data.coords['x']
    x_ = x[np.abs(x) < max_dist]

    return x_, sample(x_)


def _find_width(data, theta, max_width=2000.):
    assert data.dims == ('x', 'y')

    sample = _get_line_sample_func(data, theta)

    x = data.coords['x']
    x_ = x[np.abs(x) < max_width/2.]

    d_max = data.values.max()

    def root_fn(x_l):
        return sample(x_l) - 0.5*sample(0.0)

    def find_edge(dir):
        # first find when data drops below zero away from x=0, this will set limit
        # range for root finding
        x_coarse = np.linspace(0.0, dir*max_width, 100)
        d_coarse = sample(x_coarse)

        i_isneg = np.min(np.argwhere(d_coarse < 0.0))
        x_lim = x_coarse[i_isneg] if d_coarse[i_isneg] < 0.0 else dir*max_width

        x_hwhm = scipy.optimize.brentq(f=root_fn, a=0.0, b=x_lim)

        return x_hwhm

    return find_edge(1.0) - find_edge(-1.0)


def covariance_direction_plot(v1, v2, s_N=200, theta_win_N=100):
    """
    Compute 2nd-order cumulant between v1 and v2 and sample and perpendicular
    to pricinple axis. `s_N` sets plot window
    """
    import matplotlib.pyplot as plot

    assert v1.shape == v2.shape
    Nx, Ny = v1.shape

    assert np.all(v1.coords['x'] == v2.coords['x'])
    assert np.all(v1.coords['y'] == v2.coords['y'])
    assert v1.dims == ('x', 'y')

    # TODO: don't actually need 2D coords here, but would have to fix indexing
    # below
    x, y = np.meshgrid(v1.coords['x'], v1.coords['y'], indexing='ij')

    C_vv = calc_2nd_cumulant(v1, v2)
    theta = identify_principle_axis(C_vv, sI_N=theta_win_N)

    x_l, C_vv_l = _line_sample(data=C_vv, theta=theta, max_dist=2000.)

    line, = plot.plot(x_l, C_vv_l, label=r'$\theta=\theta_{princip}$')
    width = _find_width(C_vv, theta)
    plot.axvline(-0.5*width, linestyle='--', color=line.get_color())
    plot.axvline(0.5*width, linestyle='--', color=line.get_color())

    x_l, C_vv_l = _line_sample(data=C_vv, theta=theta+pi/2., max_dist=2000.)
    line, = plot.plot(x_l, C_vv_l, label=r'$\theta=\theta_{princip} + 90^{\circ}$')
    width = _find_width(C_vv, theta+pi/2.)
    plot.axvline(-0.5*width, linestyle='--', color=line.get_color())
    plot.axvline(0.5*width, linestyle='--', color=line.get_color())

    plot.legend()
    plot.xlabel('distance [m]')
    plot.ylabel('covariance [{}]'.format(C_vv.units))
    ax = plot.gca()
    plot.text(0.05, 0.9, r"$\theta_{{princip}}={:.2f}^{{\circ}}$"
              "".format(theta*180./pi), transform=ax.transAxes)

    if 'zt' in v1:
        z_var = 'zt'
    else:
        z_var = 'zm'

    plot.title("{} sampled along and\n perpendicular to principle axis "
               "at z={z}{z_units}".format(
                   C_vv.longname, z=float(v1[z_var]), z_units=v1[z_var].units
               ))

def charactistic_scales(v1, v2, s_N=100):
    """
    From 2nd-order cumulant of v1 and v2 compute principle axis angle,
    characteristic length-scales along and perpendicular to principle axis (as
    full width at half maximum)
    """
    import matplotlib.pyplot as plot

    assert v1.shape == v2.shape
    Nx, Ny = v1.shape

    assert np.all(v1.coords['x'] == v2.coords['x'])
    assert np.all(v1.coords['y'] == v2.coords['y'])
    assert v1.dims == ('x', 'y')

    C_vv = calc_2nd_cumulant(v1, v2)
    theta = identify_principle_axis(C_vv, sI_N=s_N)

    width_principle_axis = _find_width(C_vv, theta)
    width_perpendicular = _find_width(C_vv, theta+pi/2.)

    return theta*180./pi, (width_principle_axis, width_perpendicular)
