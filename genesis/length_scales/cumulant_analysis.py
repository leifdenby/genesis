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
import warnings

try:
    import pyfftw.interfaces
    fft = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
    print("Using pyfftw")
except ImportError:
    import numpy.fft as fft
    print("Using numpy.fft fallback")

_var_name_mapping = {
    "q": r"q_t",
    "t": r"\theta_l",
    "q_flux": r"\overline{w'q'}",
    "t_flux": r"\overline{w'\theta_l'}",
    "w0400": r"w^{400m}",
}

def calc_2nd_cumulant(v1, v2=None, mask=None):
    """
    Calculate 2nd-order cumulant of v1 and v2 in Fourier space. If mask is
    supplied the region outside the mask is set to the mean of the masked
    region, so that this region does not contribute to the cumulant
    """
    if v2 is not None:
        assert v1.shape == v2.shape
    Nx, Ny = v1.shape

    if mask is not None:
        assert v1.shape == mask.shape

        # set outside the masked region to the mean so that values in this
        # region don't correlate with the rest of the domain
        v1 = v1.where(mask.values, other=v1.mean().values)

        if v2 is not None:
            v2 = v2.where(mask.values, other=v2.mean().values)

    V1 = fft.fft2(v1)
    if v2 is None:
        V2 = V1
    else:
        V2 = fft.fft2(v2)

    c_vv_fft = fft.ifft2(V1*V2.conjugate())

    # it's most handy to have this centered on (0,0)
    c_vv = c_vv_fft.real/(Nx*Ny)
    c_vv = np.roll(np.roll(c_vv, shift=Ny/2, axis=1), shift=Nx/2, axis=0)


    # let's give it a useful name and description
    longname = r"$C({},{})$".format(
        _var_name_mapping.get(v1.name, v1.longname),
        _var_name_mapping.get(v2.name, v2.longname),
    )
    v1_name = v1.name if v1.name is not None else v1.longname
    v2_name = v2.name if v2.name is not None else v2.longname
    name = "C({},{})".format(v1_name, v2_name)

    if mask is not None:
        longname = "{} masked by {}".format(longname, mask.longname)
        name = "{} `{}` mask".format(name, mask.name)

    attrs = dict(units="{} {}".format(v1.units, v2.units), longname=longname)

    return xr.DataArray(c_vv, dims=v1.dims, coords=v1.coords, attrs=attrs,
                        name=name)


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

    theta = np.arctan2(v[0][0], v[0][1])

    # easier to work with positive angles
    if theta < 0.0:
        theta += pi

    return xr.DataArray(theta, attrs=dict(units='radians'),
                        coords=dict(zt=C.zt))


def covariance_plot(v1, v2, s_N=200, extra_title="", theta_win_N=100,
                    mask=None):
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

    x, y = v1.coords['x'], v1.coords['y']
    x_c, y_c = x[Nx/2], y[Ny/2]
    x -= x_c
    y -= y_c

    C_vv = calc_2nd_cumulant(v1, v2, mask=mask)
    theta = identify_principle_axis(C_vv, sI_N=theta_win_N)

    s_x, s_y = slice(Nx/2 - s_N/2, Nx/2 + s_N/2), slice(Ny/2 - s_N/2, Ny/2 + s_N/2)
    x_, y_, C_vv_ = x[s_x], y[s_y], C_vv[s_x, s_y]
    plot.pcolormesh(x_, y_, C_vv_, rasterized=True)
    plot.colorbar()

    ax = plot.gca()
    ax.set_aspect(1)
    plot.xlabel('x-distance [m]')
    plot.ylabel('y-distance [m]')

    mu_l = x[s_x]
    fn_line = _get_line_sample_func(C_vv, theta)

    plot.plot(*fn_line(mu=mu_l)[0], linestyle='--', color='red')
    plot.text(0.1, 0.1, r"$\theta_{{princip}}={:.2f}^{{\circ}}$"
              "".format(theta.values*180./pi), transform=ax.transAxes,
              color='red')

    if 'zt' in v1:
        z_var = 'zt'
    else:
        z_var = 'zm'

    t_units = 's' if 'seconds' in v1.time.units else v1.time.units

    plot.title(
        """Covariance length-scale for\n{C_vv}
        t={t}{t_units} z={z}{z_units}
        """.format(C_vv=C_vv.longname,
                   t=float(v1.time), t_units=t_units,
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


def _find_width(data, theta, width_peak_fraction=0.5, max_width=5000.):
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
        return sample_fn_normed(mu)[1] - width_peak_fraction

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

        try:
            x_hwhm = scipy.optimize.brentq(f=root_fn, a=0.0, b=mu_lim)
        except ValueError as e:
            warnings.warn("Couldn't find width smaller than `{}` assuming"
                          " that the cumulant spreads to infinity".format(
                          max_width))
            x_hwhm = np.inf

        return x_hwhm

    width = find_edge(1.0) - find_edge(-1.0)

    return xr.DataArray(width, coords=dict(zt=data.zt), attrs=dict(units='m'))


def covariance_direction_plot(v1, v2, s_N=200, theta_win_N=100,
                              width_peak_fraction=0.5, mask=None):
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

    C_vv = calc_2nd_cumulant(v1, v2, mask=mask)
    theta = identify_principle_axis(C_vv, sI_N=theta_win_N)

    mu_l, C_vv_l = _line_sample(data=C_vv, theta=theta, max_dist=2000.)

    line_1, = plot.plot(mu_l, C_vv_l, label=r'$\theta=\theta_{princip}$')
    width = _find_width(C_vv, theta, width_peak_fraction)
    plot.axvline(-0.5*width, linestyle='--', color=line_1.get_color())
    plot.axvline(0.5*width, linestyle='--', color=line_1.get_color())

    mu_l, C_vv_l = _line_sample(data=C_vv, theta=theta+pi/2., max_dist=2000.)
    line_2, = plot.plot(mu_l, C_vv_l, label=r'$\theta=\theta_{princip} + 90^{\circ}$')
    width = _find_width(C_vv, theta+pi/2., width_peak_fraction)
    plot.axvline(-0.5*width, linestyle='--', color=line_2.get_color())
    plot.axvline(0.5*width, linestyle='--', color=line_2.get_color())

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
               "at z={z}{z_units}\n".format(
                   C_vv.longname, z=float(v1[z_var]), z_units=v1[z_var].units
               ))

    return [line_1, line_2]

def charactistic_scales(v1, v2=None, s_N=100, width_peak_fraction=0.5, mask=None):
    """
    From 2nd-order cumulant of v1 and v2 compute principle axis angle,
    characteristic length-scales along and perpendicular to principle axis (as
    full width at half maximum)
    """
    import matplotlib.pyplot as plot

    if v2 is not None:
        assert v1.shape == v2.shape
        assert np.all(v1.coords['x'] == v2.coords['x'])
        assert np.all(v1.coords['y'] == v2.coords['y'])

    Nx, Ny = v1.shape
    assert v1.dims == ('x', 'y')

    C_vv = calc_2nd_cumulant(v1, v2, mask=mask)
    theta = identify_principle_axis(C_vv, sI_N=s_N)

    width_principle_axis = _find_width(C_vv, theta, width_peak_fraction)
    width_perpendicular = _find_width(C_vv, theta+pi/2., width_peak_fraction)

    theta_deg = xr.DataArray(theta.values*180./pi, dims=theta.dims,
                             attrs=dict(units='deg'))

    dataset = xr.Dataset(dict(
        principle_axis=theta_deg,
        width_principle=width_principle_axis,
        width_perpendicular=width_perpendicular
    ))
    dataset['cumulant'] = C_vv.name

    return dataset
