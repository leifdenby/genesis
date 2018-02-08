"""
Contains routines to analyse coherent structures in 2D using 2nd-order cumulant
analysis
"""
import numpy as np
from scipy.constants import pi
import scipy.optimize
from intergrid import intergrid
import xarray as xr
from tqdm import tqdm


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

    return xr.DataArray(theta, attrs=dict(units='radians'),
                        coords=dict(zt=C.zt))


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

    s_line = slice(Nx/2 - s_N/2, Nx/2 + s_N/2)
    mu_l = v1.coords['x'][s_line]
    fn_line = _get_line_sample_func(C_vv, theta)

    plot.plot(*fn_line(mu=mu_l)[0], linestyle='--', color='red')
    plot.text(0.1, 0.1, r"$\theta_{{princip}}={:.2f}^{{\circ}}$"
              "".format(theta.values*180./pi), transform=ax.transAxes,
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

    def sample(mu):
        """
        mu is the distance along the rotated coordinate
        """
        x_ = np.cos(float(theta))*mu
        y_ = np.sin(float(theta))*mu
        return (x_, y_), interp_f(np.array([x_, y_]).T)*d_max

    return sample


def _line_sample(data, theta, max_dist): 
    """
    Sample 2D dataset along a line define by points in (x,y)
    """
    sample = _get_line_sample_func(data, theta)

    x = data.coords['x']
    mu_l = x[np.abs(x) < max_dist]

    return mu_l, sample(mu_l)[1]


def _find_width(data, theta, max_width=2000.):
    assert data.dims == ('x', 'y')

    x = data.coords['x']
    x_ = x[np.abs(x) < max_width/2.]

    sample_fn = _get_line_sample_func(data, theta)

    d_max = data.values.max()
    d_at_inf = sample_fn(x.max())[1]

    def sample_fn_normed(mu):
        """
        Normalize with value at infinity so that root finding method can find
        halfway value if there is net positive correlation of the entire domain
        """
        pts_xy, d_val = sample_fn(mu)

        return pts_xy, (d_val - d_at_inf)/(d_max - d_at_inf)


    def root_fn(mu):
        return sample_fn_normed(mu)[1] - 0.5

    def find_edge(dir):
        # first find when data drops below zero away from x=0, this will set limit
        # range for root finding
        mu_coarse = np.linspace(0.0, dir*max_width, 100)
        _, d_coarse = sample_fn_normed(mu=mu_coarse)

        try:
            i_isneg = np.min(np.argwhere(d_coarse < 0.0))
            mu_lim = mu_coarse[i_isneg]
        except ValueError:
            mu_lim = dir*max_width

        x_hwhm = scipy.optimize.brentq(f=root_fn, a=0.0, b=mu_lim)

        return x_hwhm

    width = find_edge(1.0) - find_edge(-1.0)

    return xr.DataArray(width, coords=dict(zt=data.zt), attrs=dict(units='m'))


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

    mu_l, C_vv_l = _line_sample(data=C_vv, theta=theta, max_dist=2000.)

    line, = plot.plot(mu_l, C_vv_l, label=r'$\theta=\theta_{princip}$')
    width = _find_width(C_vv, theta)
    plot.axvline(-0.5*width, linestyle='--', color=line.get_color())
    plot.axvline(0.5*width, linestyle='--', color=line.get_color())

    mu_l, C_vv_l = _line_sample(data=C_vv, theta=theta+pi/2., max_dist=2000.)
    line, = plot.plot(mu_l, C_vv_l, label=r'$\theta=\theta_{princip} + 90^{\circ}$')
    width = _find_width(C_vv, theta+pi/2.)
    plot.axvline(-0.5*width, linestyle='--', color=line.get_color())
    plot.axvline(0.5*width, linestyle='--', color=line.get_color())

    plot.legend()
    plot.xlabel('distance [m]')
    plot.ylabel('covariance [{}]'.format(C_vv.units))
    ax = plot.gca()
    plot.text(0.05, 0.9, r"$\theta_{{princip}}={:.2f}^{{\circ}}$"
              "".format(theta.values*180./pi), transform=ax.transAxes)

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

    theta_deg = xr.DataArray(theta.values*180./pi, dims=theta.dims,
                             attrs=dict(units='deg'))

    dataset = xr.Dataset(dict(
        principle_axis=theta_deg,
        width_principle=width_principle_axis,
        width_perpendicular=width_perpendicular
    ))
    dataset.attrs['cumulant'] = C_vv.longname

    return dataset
