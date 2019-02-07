"""
Contains routines to analyse coherent structures in 2D using 2nd-order cumulant
analysis
"""
import warnings
import re

import numpy as np
from scipy.constants import pi
import scipy.optimize
import scipy.integrate
from intergrid import intergrid
import xarray as xr
from tqdm import tqdm

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

RE_CUMULANT_NAME = re.compile('C\((\w+),(\w+)\)(.*)')

def fix_cumulant_name(name):
    name_mapping = {
        'q': 'q_t',
        't': r"\theta_l",
        'l': 'q_l',
        'q_flux': r"w'q_t'",
        't_flux': r"w'\theta_l'",
        'l_flux': r"w'q_l'",
    }

    v1, v2, extra = RE_CUMULANT_NAME.match(name).groups()

    extra = extra.strip()

    v1_latex = name_mapping.get(v1, v1)
    v2_latex = name_mapping.get(v2, v2)

    if len(extra) > 0:
        return r"$C({},{})$".format(v1_latex, v2_latex) + '\n' + extra
    else:
        return r"$C({},{})$".format(v1_latex, v2_latex)

def calc_2nd_cumulant(v1, v2=None, mask=None):
    """
    Calculate 2nd-order cumulant of v1 and v2 in Fourier space. If mask is
    supplied the region outside the mask is set to the mean of the masked
    region, so that this region does not contribute to the cumulant
    """
    if v2 is not None:
        assert v1.shape == v2.shape
    if v1.dims == ('x', 'y'):
        Nx, Ny = v1.shape
    else:
        Ny, Nx = v1.shape

    old_attrs = v1.attrs
    v1 = v1 - v1.mean()
    v1.attrs = old_attrs
    if v2 is not None:
        v2_old_attrs = v2.attrs
        v2 = v2 - v2.mean()
        v2.attrs = v2_old_attrs

    if mask is not None:
        assert v1.shape == mask.shape
        assert v1.dims == mask.dims

        # set outside the masked region to the mean so that values in this
        # region don't correlate with the rest of the domain
        v1 = v1.where(mask.values, other=0.0)

        if v2 is not None:
            v2 = v2.where(mask.values, other=0.0)

    V1 = fft.fft2(v1)
    if v2 is None:
        v2 = v1
        V2 = V1
    else:
        V2 = fft.fft2(v2)

    c_vv_fft = fft.ifft2(V1*V2.conjugate())

    # it's most handy to have this centered on (0,0)
    c_vv = c_vv_fft.real/(Nx*Ny)
    if v1.dims == ('x', 'y'):
        c_vv = np.roll(
            np.roll(c_vv, shift=int(Ny/2), axis=1),
            shift=int(Nx/2), axis=0
        )
    else:
        c_vv = np.roll(
            np.roll(c_vv, shift=int(Ny/2), axis=0),
            shift=int(Nx/2), axis=1
        )


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

    attrs = dict(units="{} {}".format(v1.units, v2.units), longname=longname)

    return xr.DataArray(c_vv, dims=v1.dims, coords=v1.coords, attrs=attrs,
                        name=name)


def identify_principle_axis(C, sI_N=100):
    """
    Using 2nd-order cumulant identify principle axis of correlation in 2D.
    `sI_N` denotes window over which to look for maximum correlation
    """
    if C.dims == ('x', 'y'):
        Nx, Ny = C.shape
    else:
        Ny, Nx = C.shape

    x_ = C.coords['x']
    y_ = C.coords['y']

    I_func = lambda x, y, m: np.array([
        [np.sum(m*y**2.), np.sum(m*x*y)],
        [np.sum(m*y*x),  np.sum(x**2.*m)]
    ])

    sI_x = slice(Nx//2 - sI_N//2, Nx//2 + sI_N//2)
    sI_y = slice(Ny//2 - sI_N//2, Ny//2 + sI_N//2)

    if C.dims == ('x', 'y'):
        x, y = np.meshgrid(x_[sI_x], y_[sI_y], indexing='ij')
        I = I_func(x, y, C[sI_x, sI_y])
    else:
        x, y = np.meshgrid(x_[sI_y], y_[sI_x], indexing='ij')
        I = I_func(x, y, C[sI_y, sI_x])

    la, v = np.linalg.eig(I)

    # sort eigenvectors by eigenvalue, the largest eigenvalue will be the
    # principle axis
    # sort_idx = np.argsort(np.abs(la))[::-1]
    sort_idx = np.argsort(la)
    la = la[sort_idx]
    v = v[sort_idx]

    # the pricinple axis
    v0 = v[0]

    if v0.dtype == np.complex128:
        if v0.imag.max() > 0:
            raise Exception("Encountered imaginary eigenvector, not sure"
                            " how to deal with this")
        else:
            v0 = v0.real

    theta = np.arctan2(v0[0], v0[1])

    # easier to work with positive angles
    if theta < 0.0:
        theta += pi

    return xr.DataArray(theta, attrs=dict(units='radians'),
                        coords=dict(zt=C.zt))


def covariance_plot(v1, v2, s_N=200, extra_title="", theta_win_N=100,
                    mask=None, sample_angle=None, ax=None, add_colorbar=True,
                    log_scale=True):
    """
    Make a 2D covariance plot of v1 and v2 (both are expected to be
    xarray.DataArray) and identify principle axis. Covariance analysis is
    plotted over a window of s_N x s_N
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    assert v1.shape == v2.shape
    if v1.dims == ('x', 'y'):
        Nx, Ny = v1.shape
    else:
        Ny, Nx = v1.shape

    assert np.all(v1.coords['x'] == v2.coords['x'])
    assert np.all(v1.coords['y'] == v2.coords['y'])

    x, y = v1.coords['x'], v1.coords['y']
    x_c, y_c = x[Nx//2], y[Ny//2]
    x -= x_c
    y -= y_c

    C_vv = calc_2nd_cumulant(v1, v2, mask=mask)
    if sample_angle is not None:
        theta = xr.DataArray(sample_angle*pi/180., attrs=dict(units='radians'),
                        coords=dict(zt=v1.zt))
    else:
        theta = identify_principle_axis(C_vv, sI_N=theta_win_N)

    s_x = slice(Nx//2 - s_N//2, Nx//2 + s_N//2)
    s_y = slice(Ny//2 - s_N//2, Ny//2 + s_N//2)

    if v1.dims == ('x', 'y'):
        x_, y_, C_vv_ = x[s_x], y[s_y], C_vv[s_x, s_y]
    else:
        x_, y_, C_vv_ = x[s_x], y[s_y], C_vv[s_y, s_x]

    if log_scale:
        C_vv_ = np.sign(C_vv_)*np.log(np.abs(C_vv_))

    im = C_vv_.plot.pcolormesh(rasterized=True, robust=True,
                               add_colorbar=add_colorbar, ax=ax)
    # im = ax.pcolormesh(x_, y_, C_vv_, rasterized=True)
    # if add_colorbar:
        # plt.gcf().colorbar(im)

    if add_colorbar and C_vv_.min() < 1.0e-3:
        # use scientific notation on the colorbar
        cb = im.colorbar
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()

    ax.set_aspect(1)
    ax.set_xlabel('x-distance [m]')
    ax.set_ylabel('y-distance [m]')

    mu_l = x[s_x]
    fn_line = _get_line_sample_func(C_vv, theta)

    ax.plot(*fn_line(mu=mu_l)[0], linestyle='--', color='red')
    ax.text(0.1, 0.1, r"$\theta_{{princip}}={:.2f}^{{\circ}}$"
              "".format(theta.values*180./pi), transform=ax.transAxes,
              color='red')

    if 'zt' in v1:
        z_var = 'zt'
    else:
        z_var = 'zm'

    try:
        t_units = 's' if 'seconds' in v1.time.units else v1.time.units

        ax.set_title(
            """Covariance length-scale for\n{C_vv}
            t={t}{t_units} z={z}{z_units}
            """.format(C_vv=C_vv.longname,
                       t=float(v1.time), t_units=t_units,
                       z=float(v1[z_var]), z_units=v1[z_var].units)
        )
    except AttributeError:
        pass

    return ax, cb


def _get_line_sample_func(data, theta):
    x = data.coords['x']
    y = data.coords['y']

    d_max = data.values.max()

    maps = [x, y]
    if data.dims == ('x', 'y'):
        lo = np.array([x[0], y[0]])
        hi = np.array([x[-1], y[-1]])
    else:
        lo = np.array([y[0], x[0]])
        hi = np.array([y[-1], x[-1]])

    interp_f = intergrid.Intergrid(data.values/d_max,
                                   lo=lo, hi=hi, maps=maps, verbose=0)

    def sample(mu):
        """
        mu is the distance along the rotated coordinate
        """
        x_ = np.cos(float(theta))*mu
        y_ = np.sin(float(theta))*mu
        if data.dims == ('x', 'y'):
            p = np.array([x_, y_]).T
        else:
            p = np.array([y_, x_]).T
        return (x_, y_), interp_f(p)*d_max

    return sample


def _line_sample(data, theta, max_dist): 
    """
    Sample 2D dataset along a line define by points in (x,y)
    """
    sample = _get_line_sample_func(data, theta)

    x = data.coords['x']
    mu_l = x[np.abs(x) < max_dist]

    return mu_l, sample(mu_l)[1]


def _find_width_through_mass_weighting(data, theta, max_width=5000.):
    sample_fn = _get_line_sample_func(data, theta)

    d_max = data.values.max()
    # use corner of domain as reference
    d_at_inf = data[0,0].values

    def sample_fn_normed(mu):
        """
        Normalize with value at infinity so that we find the width of the
        distribution above this value at "infinity"
        """
        pts_xy, d_val = sample_fn(mu)

        return pts_xy, (d_val - d_at_inf)/(d_max - d_at_inf)

    def mass_weighted_edge(dir):
        # we only integrate positive "mass" contributions, including negative
        # contributions didn't work...
        def fn(mu):
            val = sample_fn_normed(mu)[1]
            return np.maximum(0, val)

        fn_inertia = lambda mu: fn(mu)*np.abs(mu)
        fn_mass = lambda mu: fn(mu)

        if dir == 1:
            kw = dict(a=0, b=max_width/2.)
        else:
            kw = dict(a=-max_width/2., b=0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inertia, _ = scipy.integrate.quad(fn_inertia, **kw)
            mass, _ = scipy.integrate.quad(fn_mass, **kw)

        dist = inertia/mass

        return dir*dist

    width = mass_weighted_edge(1.0) - mass_weighted_edge(-1.0)

    return xr.DataArray(width, coords=dict(zt=data.zt), attrs=dict(units='m'))


def _find_width_through_cutoff(data, theta, width_peak_fraction=0.5,
                               max_width=5000.):
    x = data.coords['x']
    x_ = x[np.abs(x) < max_width/2.]

    sample_fn = _get_line_sample_func(data, theta)

    d_max = data.values.max()
    # d_at_inf = sample_fn(x.max())[1]
    # use corner of domain as reference
    d_at_inf = data[0,0].values

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
                              width_peak_fraction=0.5, mask=None,
                              max_dist=2000., with_45deg_sample=False,
                              sample_angle=None, ax=None):
    """
    Compute 2nd-order cumulant between v1 and v2 and sample and perpendicular
    to pricinple axis. `s_N` sets plot window
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    assert v1.shape == v2.shape
    if v1.dims == ('x', 'y'):
        Nx, Ny = v1.shape
    elif v1.dims == ('y', 'x'):
        Ny, Nx = v1.shape
    else:
        raise NotImplementedError

    assert np.all(v1.coords['x'] == v2.coords['x'])
    assert np.all(v1.coords['y'] == v2.coords['y'])

    # TODO: don't actually need 2D coords here, but would have to fix indexing
    # below
    if v1.dims == ('x', 'y'):
        x, y = np.meshgrid(v1.coords['x'], v1.coords['y'], indexing='ij')
    else:
        x, y = np.meshgrid(v1.coords['x'], v1.coords['y'])

    C_vv = calc_2nd_cumulant(v1, v2, mask=mask)
    if sample_angle is not None:
        theta = xr.DataArray(sample_angle*pi/180., attrs=dict(units='radians'),
                        coords=dict(zt=v1.zt))
    else:
        theta = identify_principle_axis(C_vv, sI_N=theta_win_N)

    mu_l, C_vv_l = _line_sample(data=C_vv, theta=theta, max_dist=max_dist)

    line_1, = ax.plot(mu_l, C_vv_l, label=r'$\theta=\theta_{princip}$')
    width = _find_width_through_mass_weighting(C_vv, theta)
    ax.axvline(-0.5*width, linestyle='--', color=line_1.get_color())
    ax.axvline(0.5*width, linestyle='--', color=line_1.get_color())

    mu_l, C_vv_l = _line_sample(data=C_vv, theta=theta+pi/2., max_dist=max_dist)
    line_2, = ax.plot(mu_l, C_vv_l, label=r'$\theta=\theta_{princip} + 90^{\circ}$')
    width = _find_width_through_mass_weighting(C_vv, theta+pi/2.)
    ax.axvline(-0.5*width, linestyle='--', color=line_2.get_color())
    ax.axvline(0.5*width, linestyle='--', color=line_2.get_color())

    if with_45deg_sample:
        mu_l, C_vv_l = _line_sample(data=C_vv, theta=theta+pi/4., max_dist=max_dist)
        line_2, = ax.plot(mu_l, C_vv_l, label=r'$\theta=\theta_{princip} + 45^{\circ}$')
        width = _find_width(C_vv, theta+pi/2., width_peak_fraction)
        ax.axvline(-0.5*width, linestyle='--', color=line_2.get_color())
        ax.axvline(0.5*width, linestyle='--', color=line_2.get_color())

    ax.legend(loc='upper right')
    ax.set_xlabel('distance [m]')
    ax.set_ylabel('covariance [{}]'.format(C_vv.units))
    ax.text(0.05, 0.8, r"$\theta_{{princip}}={:.2f}^{{\circ}}$"
              "".format(theta.values*180./pi), transform=ax.transAxes)

    if 'zt' in v1:
        z_var = 'zt'
    else:
        z_var = 'zm'

    ax.set_title("{} sampled along and\n perpendicular to principle axis "
               "at z={z}{z_units}\n".format(
                   C_vv.longname, z=float(v1[z_var]), z_units=v1[z_var].units
               ))

    return [line_1, line_2]

def charactistic_scales(v1, v2=None, l_theta_win=2000., mask=None,
                        sample_angle=None):
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

    dx = np.max(np.gradient(v1.x))

    C_vv = calc_2nd_cumulant(v1, v2, mask=mask)

    l_win = l_theta_win
    found_min_max_lengths = False
    m = 0
    if sample_angle is not None:
        theta = xr.DataArray(sample_angle*pi/180., attrs=dict(units='radians'),
                        coords=dict(zt=v1.zt))

        width_principle_axis = _find_width_through_mass_weighting(C_vv, theta)
        width_perpendicular = _find_width_through_mass_weighting(C_vv, theta+pi/2.)
    else:
        while True:
            s_N = int(l_win/dx)*2
            theta = identify_principle_axis(C_vv, sI_N=s_N)

            width_principle_axis = _find_width_through_mass_weighting(C_vv, theta)
            width_perpendicular = _find_width_through_mass_weighting(C_vv, theta+pi/2.)

            d_width = np.abs(width_perpendicular - width_principle_axis)
            mean_width = 0.5*(width_perpendicular + width_principle_axis)

            if np.isnan(width_perpendicular) or np.isnan(width_principle_axis):
                break

            if d_width/mean_width > 0.30 and width_perpendicular > width_principle_axis:
                l_win = 1.2*l_win

                m += 1
                if m > 10:
                    warnings.warn("Couldn't find principle axis")
                    width_principle_axis = np.nan
                    width_perpendicular = np.nan
                    theta = np.nan
                    break
            else:
                break

            # else:

    theta_deg = xr.DataArray(theta.values*180./pi, dims=theta.dims,
                             attrs=dict(units='deg'))

    dataset = xr.Dataset(dict(
        principle_axis=theta_deg,
        width_principle=width_principle_axis,
        width_perpendicular=width_perpendicular
    ))
    dataset['cumulant'] = C_vv.name

    return dataset
