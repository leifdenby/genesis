"""
Contains routines to analyse coherent structures in 2D using 2nd-order cumulant
analysis
"""
import re
import warnings
from enum import Enum

import numpy as np
import scipy.integrate
import scipy.optimize
import skimage.measure
import xarray as xr
from scipy.constants import pi

from ...utils.intergrid import intergrid

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

RE_CUMULANT_NAME = re.compile(r"C\((\w+),(\w+)\)(.*)")


def fix_cumulant_name(name):
    name_mapping = {
        "q": "q_t",
        "t": r"\theta_l",
        "l": "q_l",
        "cvrxp": r"\phi",
        "w_zt": r"w",
        "theta_l": r"\theta_l",
        "qv": r"q_v",
        "qv__norain": r"q_v*",
        "qc": r"q_c",
        "theta_l_v": r"\theta_{l,v}",
    }

    def s_map(s):
        suffix = ""
        prefix = ""
        if s.endswith("_flux"):
            prefix = "w"
            suffix = "'"
            var_name = s.replace("_flux", "")
        elif s.startswith("d_"):
            suffix = "'"
            var_name = s[2:]
        else:
            var_name = s

        return "{}{}{}".format(prefix, name_mapping.get(var_name, var_name), suffix)

    v1, v2, extra = RE_CUMULANT_NAME.match(name).groups()

    extra = extra.strip()

    v1_latex = s_map(v1)
    v2_latex = s_map(v2)

    if v1_latex.count("_") > 1:
        raise Exception("Please create latex name mapping for `{}`".format(v1_latex))
    if v2_latex.count("_") > 1:
        raise Exception("Please create latex name mapping for `{}`".format(v2_latex))

    if len(extra) > 0:
        return r"$c_{{{},{}}}$".format(v1_latex, v2_latex) + "\n" + extra
    else:
        return r"$c_{{{},{}}}$".format(v1_latex, v2_latex)


def calc_2nd_cumulant(v1, v2=None, mask=None, axes=(-2, -1)):
    """
    Calculate 2nd-order cumulant of v1 and v2 in Fourier space. If mask is
    supplied the region outside the mask is set to the mean of the masked
    region, so that this region does not contribute to the cumulant

    If axes is a sequence with only one item then the cumulant is only
    calculated along that axis and the others are left unchanged.
    """

    if v2 is not None:
        assert v1.shape == v2.shape
        assert v1.dims == v2.dims

    assert len(axes) in [1, 2]
    v_shape = v1.shape

    x_dim = v1.dims[axes[0]]
    Nx = v_shape[axes[0]]
    if len(axes) == 1:
        mean_axis = axes[0]
    else:
        mean_axis = None
        y_dim = v1.dims[axes[1]]
        Ny = v_shape[axes[1]]

    old_attrs = v1.attrs
    v1 = v1 - v1.mean(axis=mean_axis)
    v1.attrs = old_attrs
    if v2 is not None:
        v2_old_attrs = v2.attrs
        v2 = v2 - v2.mean(axis=mean_axis)
        v2.attrs = v2_old_attrs

    if mask is not None:
        assert v1.shape == mask.shape
        assert v1.dims == mask.dims

        # set outside the masked region to the mean so that values in this
        # region don't correlate with the rest of the domain
        v1 = v1.where(mask.values, other=0.0)

        if v2 is not None:
            v2 = v2.where(mask.values, other=0.0)

    V1 = fft.fft2(v1, axes=axes)
    if v2 is None:
        v2 = v1
        V2 = V1
    else:
        V2 = fft.fft2(v2, axes=axes)

    c_vv_fft = fft.ifft2(V1 * V2.conjugate(), axes=axes)

    c_vv = c_vv_fft.real / Nx
    if len(axes) == 2:
        c_vv /= Ny
    # it's most handy to have this centered on (0,0)
    c_vv = np.roll(c_vv, shift=int(Nx / 2), axis=axes[0])
    if len(axes) == 2:
        c_vv = np.roll(c_vv, shift=int(Ny / 2), axis=axes[1])

    # let's give it a useful name and description
    long_name = r"$C({},{})$".format(
        _var_name_mapping.get(v1.name, v1.long_name),
        _var_name_mapping.get(v2.name, v2.long_name),
    )
    v1_name = v1.name if v1.name is not None else v1.long_name
    v2_name = v2.name if v2.name is not None else v2.long_name
    name = "C({},{})".format(v1_name, v2_name)

    # create displacement coordinate for cumulant data
    def _calc_offset_coord(v_coord):
        if np.issubdtype(v_coord.dtype, np.datetime64):
            # for datetime we could use timedelta, but matplotlib can't plot
            # these and so we convert to number of seconds here
            v_center = v_coord.min() + 0.5 * (v_coord.max() - v_coord.min())
            v_coord_new = v_coord - v_center
            v_coord_new = xr.DataArray(
                v_coord_new.values.astype("timedelta64[ns]").astype(int) / 1.0e9,
                dims=v_coord.dims,
                attrs=dict(units="s"),
            )
        else:
            v_coord_new = v_coord - 0.5 * (v_coord.min() + v_coord.max())
        v_coord_new.attrs.update(v_coord.attrs)
        return v_coord_new

    x_cvv = _calc_offset_coord(v1[x_dim])
    # start of with original coords
    coords_cvv = v1.coords
    coords_cvv[x_dim] = x_cvv
    if len(axes) == 2:
        y_cvv = _calc_offset_coord(v1[y_dim])
        coords_cvv[y_dim] = y_cvv

    if mask is not None:
        long_name = "{} masked by {}".format(long_name, mask.long_name)

    attrs = dict(units="{} {}".format(v1.units, v2.units), long_name=long_name)

    return xr.DataArray(c_vv, dims=v1.dims, coords=coords_cvv, attrs=attrs, name=name)


def identify_principle_axis(C, sI_N=100):
    """
    Using 2nd-order cumulant identify principle axis of correlation in 2D.
    `sI_N` denotes window over which to look for maximum correlation
    """
    if C.dims == ("x", "y"):
        Nx, Ny = C.shape
    else:
        Ny, Nx = C.shape

    x_ = C.coords["x"]
    y_ = C.coords["y"]

    I_func = lambda x, y, m: np.array(  # noqa
        [
            [np.sum(m * y**2.0), np.sum(m * x * y)],
            [np.sum(m * y * x), np.sum(x**2.0 * m)],
        ]
    )

    sI_x = slice(Nx // 2 - sI_N // 2, Nx // 2 + sI_N // 2)
    sI_y = slice(Ny // 2 - sI_N // 2, Ny // 2 + sI_N // 2)

    # if the correlation is negative we still want to find the width over which
    # this is true, so we invert the cumulant so that we still can integrate
    # over its "mass"
    s = np.sign(C[Nx // 2, Ny // 2])

    if C.dims == ("x", "y"):
        x, y = np.meshgrid(x_[sI_x], y_[sI_y], indexing="ij")
        I = I_func(x, y, s * C[sI_x, sI_y])  # noqa
    else:
        x, y = np.meshgrid(x_[sI_y], y_[sI_x], indexing="ij")
        I = I_func(x, y, s * C[sI_y, sI_x])  # noqa

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
            raise Exception(
                "Encountered imaginary eigenvector, not sure" " how to deal with this"
            )
        else:
            v0 = v0.real

    theta = np.arctan2(v0[0], v0[1])

    # easier to work with positive angles
    if theta < 0.0:
        theta += pi

    return xr.DataArray(theta, attrs=dict(units="radians"), coords=dict(zt=C.zt))


def _extract_cumulant_center(C_vv):
    """
    Mask everything but the contiguous part of the cumulant which has the same
    sign as its origin. This is needed by the mass-weighted width method to
    avoid including correlating regions outside of the principle region
    """
    nx, ny = C_vv.shape

    mask = np.sign(C_vv) == np.sign(C_vv[nx // 2, ny // 2])
    labels = skimage.measure.label(mask.values)

    mask_center = labels == labels[nx // 2, ny // 2]

    da_mask_center = xr.DataArray(mask_center, coords=C_vv.coords, dims=C_vv.dims)

    return C_vv.where(da_mask_center, drop=True)


def covariance_plot(
    v1,
    v2,
    s_N=200,
    extra_title="",
    theta_win_N=100,
    mask=None,
    sample_angle=None,
    ax=None,
    add_colorbar=True,
    log_scale=True,
    autoscale_dist=True,
    line_color="green",
    robust_cmap=True,
):
    """
    Make a 2D covariance plot of v1 and v2 (both are expected to be
    xarray.DataArray) and identify principle axis. Covariance analysis is
    plotted over a window of s_N x s_N
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    assert v1.shape == v2.shape
    if v1.dims == ("x", "y"):
        Nx, Ny = v1.shape
    else:
        Ny, Nx = v1.shape

    s_N = min(s_N, Nx)
    theta_win_N = min(theta_win_N, Nx)

    assert np.all(v1.coords["x"] == v2.coords["x"])
    assert np.all(v1.coords["y"] == v2.coords["y"])
    all_dist_units = set([v1.x.units, v2.x.units, v1.y.units, v2.y.units])
    assert len(all_dist_units)
    dist_units = all_dist_units.pop()

    x, y = v1.coords["x"], v1.coords["y"]
    x_c, y_c = x[Nx // 2], y[Ny // 2]
    x = x - x_c
    y = y - y_c
    x.attrs["units"] = dist_units
    y.attrs["units"] = dist_units

    C_vv = calc_2nd_cumulant(v1, v2, mask=mask)
    if sample_angle is not None:
        theta = xr.DataArray(
            sample_angle * pi / 180.0,
            attrs=dict(units="radians"),
            coords=dict(zt=v1.zt),
        )
    else:
        theta = identify_principle_axis(C_vv, sI_N=theta_win_N)

    s_x = slice(Nx // 2 - s_N // 2, Nx // 2 + s_N // 2)
    s_y = slice(Ny // 2 - s_N // 2, Ny // 2 + s_N // 2)

    if v1.dims == ("x", "y"):
        C_vv_ = C_vv[s_x, s_y]
    else:
        C_vv_ = C_vv[s_y, s_x]

    if autoscale_dist:
        lx = x.max() - x.min()
        if x.units == "m" and lx > 1.0e3:
            for v in ["x", "y"]:
                v_new = "{}_scaled".format(v)
                C_vv_[v_new] = C_vv_[v] / 1000.0
                C_vv_[v_new].attrs["units"] = "km"
                C_vv_[v_new].attrs["long_name"] = "{}-distance".format(v)

            C_vv_ = C_vv_.swap_dims(dict(x="x_scaled", y="y_scaled"))

    if log_scale:
        C_vv_ = np.sign(C_vv_) * np.log(np.abs(C_vv_))

    if add_colorbar:
        # add a latex formatted name for xarray to print on the colorbar
        C_vv_.attrs["long_name"] = fix_cumulant_name(C_vv_.name)
    C_vv_.attrs["units"] = C_vv.units

    im = C_vv_.plot.pcolormesh(
        rasterized=True, robust=robust_cmap, add_colorbar=add_colorbar, ax=ax
    )
    # im = ax.pcolormesh(x_, y_, C_vv_, rasterized=True)
    # if add_colorbar:
    # plt.gcf().colorbar(im)

    if add_colorbar and C_vv_.min() < 1.0e-3:
        # use scientific notation on the colorbar
        cb = im.colorbar
        cb.formatter.set_powerlimits((0, 0))
        cb.formatter.set_useMathText(True)
        cb.update_ticks()

    ax.set_aspect(1)

    mu_l = x[s_x]
    fn_line = _get_line_sample_func(C_vv, theta)
    (line,) = ax.plot(*fn_line(mu=mu_l)[0], linestyle="-", color=line_color)
    text = ax.text(
        0.1,
        0.1,
        r"$\theta^{{p}}={:.1f}^{{\circ}}$" "".format(theta.values * 180.0 / pi),
        transform=ax.transAxes,
        color=line.get_color(),
    )
    text.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="lightgrey"))

    # perpendicular line
    fn_line = _get_line_sample_func(C_vv, theta + pi / 2.0)
    ax.plot(*fn_line(mu=mu_l)[0], linestyle="--", color=line.get_color())

    try:
        v1.zt
        z_var = "zt"
    except AttributeError:
        z_var = "zm"

    try:
        t_units = "s" if "seconds" in v1.time.units else v1.time.units

        ax.set_title(
            """Covariance length-scale for\n{C_vv}
            t={t}{t_units} z={z}{z_units}
            """.format(
                C_vv=fix_cumulant_name(C_vv.name),
                t=float(v1.time),
                t_units=t_units,
                z=float(v1[z_var]),
                z_units=v1[z_var].units,
            )
        )
    except AttributeError:
        pass

    return ax


def _get_line_sample_func(data, theta):
    x = data.coords["x"]
    y = data.coords["y"]

    # the interpolation method works best for data in the -1:1 range
    d_scale = max(data.values.max(), -data.values.min())
    if d_scale == 0.0:
        d_scale = 1.0

    maps = [x, y]
    if data.dims == ("x", "y"):
        lo = np.array([x[0], y[0]])
        hi = np.array([x[-1], y[-1]])
    else:
        lo = np.array([y[0], x[0]])
        hi = np.array([y[-1], x[-1]])

    interp_f = intergrid.Intergrid(
        data.values / d_scale, lo=lo, hi=hi, maps=maps, verbose=0
    )

    def sample(mu):
        """
        mu is the distance along the rotated coordinate
        """
        x_ = np.cos(float(theta)) * mu
        y_ = np.sin(float(theta)) * mu
        if data.dims == ("x", "y"):
            p = np.array([x_, y_]).T
        else:
            p = np.array([y_, x_]).T
        return (x_, y_), interp_f(p) * d_scale

    return sample


def _line_sample(data, theta, max_dist):
    """
    Sample 2D dataset along a line define by points in (x,y)
    """
    sample = _get_line_sample_func(data, theta)

    x = data.coords["x"]
    mu_l = x[np.abs(x) < max_dist]

    return mu_l, sample(mu_l)[1]


class WidthEstimationMethod(Enum):
    MASS_WEIGHTED = 0
    CUTOFF = 1

    def __str__(self):
        return self.name


def _find_width_through_mass_weighting(data, theta, max_width=5000.0, center_only=True):
    """
    Integrates the central cumulant over distance to calculate a weighted
    length-scale. If `center_only` is true only the part that has the same sign
    as correlation at the origin (0,0) is included.
    """
    assert data.x.units == "m"
    sample_fn = _get_line_sample_func(data, theta)

    if center_only:
        data = _extract_cumulant_center(data)

    def mass_weighted_edge(dir):
        # we only integrate positive "mass" contributions, including negative
        # contributions didn't work...

        # if the correlation is negative we still want to be able to calculate
        # a width, the distance over which the correlation is negative. And so
        # we multiply by the sign at the origin to flip the function
        s = np.sign(sample_fn(0.0)[1])

        def fn(mu):
            val = sample_fn(mu)[1]
            return np.maximum(0, s * val)

        fn_inertia = lambda mu: fn(mu) * np.abs(mu)  # noqa
        fn_mass = lambda mu: fn(mu)  # noqa

        if dir == 1:
            kw = dict(a=0, b=max_width / 2.0)
        else:
            kw = dict(a=-max_width / 2.0, b=0)

        # try to work out if function has a local minimum, i.e. whether we get
        # a flip in correlation, if so we only want to integrate up to this
        # limit so that we avoid getting contributions if the correlation
        # becomes positive again
        x_ = np.linspace(kw["a"], kw["b"], 100)
        vals_ = s * sample_fn(x_)[1]
        if np.all(np.isnan(vals_)):
            raise Exception("No valid datapoints found")
        elif np.min(vals_) < 0.0:
            max_width_local = x_[np.argmin(vals_)]
            if dir == 1:
                kw = dict(a=0, b=max_width_local)
            else:
                kw = dict(a=-max_width_local, b=0)

        # import matplotlib.pyplot as plt
        # if dif == 1:
        # plt.figure()
        # plt.plot(x_, fn(x_))
        # else:
        # plt.plot(x_, fn(x_))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inertia, _ = scipy.integrate.quad(fn_inertia, **kw)
            mass, _ = scipy.integrate.quad(fn_mass, **kw)

        dist = inertia / mass

        return dir * dist

    try:
        width = mass_weighted_edge(1.0) - mass_weighted_edge(-1.0)
    except ZeroDivisionError:
        width = np.nan

    return xr.DataArray(width, coords=dict(zt=data.zt), attrs=dict(units="m"))


def _find_width_through_cutoff(data, theta, width_peak_fraction=0.5, max_width=5000.0):
    x = data.coords["x"]
    assert x.units == "m"

    sample_fn = _get_line_sample_func(data, theta)

    d_max = data.values.max()
    # d_at_inf = sample_fn(x.max())[1]
    # use corner of domain as reference
    # d_at_inf = data[0,0].values
    d_at_inf = 0.0

    def sample_fn_normed(mu):
        """
        Normalize with value at infinity so that root finding method can find
        halfway value if there is net positive correlation of the entire domain
        """
        pts_xy, d_val = sample_fn(mu)

        return pts_xy, (d_val - d_at_inf) / (d_max - d_at_inf)

    def root_fn(mu):
        return sample_fn_normed(mu)[1] - width_peak_fraction

    def find_edge(dir):
        # first find when data drops below zero away from x=0, this will set limit
        # range for root finding
        mu_coarse = np.linspace(0.0, dir * max_width, 100)
        _, d_coarse = sample_fn_normed(mu=mu_coarse)

        try:
            i_isneg = np.min(np.argwhere(d_coarse < 0.0))
            mu_lim = mu_coarse[i_isneg]
        except ValueError:
            mu_lim = dir * max_width

        try:
            x_hwhm = scipy.optimize.brentq(f=root_fn, a=0.0, b=mu_lim)
        except ValueError:
            warnings.warn(
                "Couldn't find width smaller than `{}` assuming"
                " that the cumulant spreads to infinity".format(max_width)
            )
            x_hwhm = np.inf

        return x_hwhm

    width = find_edge(1.0) - find_edge(-1.0)

    return xr.DataArray(width, coords=dict(zt=data.zt), attrs=dict(units="m"))


def add_width_indicator_legend(ax):
    """
    Render a cumulant width indicator below the bottom right corner of the the
    axes `ax`
    """
    # set the width and height of the axes we'll draw into in inches
    # so that they're always the same size
    fig = ax.figure
    fw, fh = (fig.dpi_scale_trans - fig.transFigure).transform([1.6, 0.5])

    # and we want to attach to the bottom corner of the provided axes
    fx, fy = (ax.transAxes - fig.transFigure).transform([1.0, 0.0])
    ax_inset = fig.add_axes([fx - fw, fy - 2 * fh, fw, fh], transform=fig.transFigure)

    labels = [r"$L^p$", r"$L^{\bot}$"]
    linestyles = ["-", "--"]

    ax_inset.axis("off")
    for n, (label, ls) in enumerate(zip(labels, linestyles)):
        w = 0.2
        eb = ax_inset.errorbar(x=n, y=0.0, xerr=w, color="black", capsize=5.0)
        eb[-1][0].set_linestyle(ls)
        ax_inset.text(x=n - 2.5 * w, y=0.0, s=label, va="center")


def covariance_direction_plot(
    v1,
    v2,
    s_N=200,
    theta_win_N=100,
    width_peak_fraction=0.5,
    mask=None,
    max_dist=2000.0,
    with_45deg_sample=False,
    sample_angle=None,
    ax=None,
    width_est_method=WidthEstimationMethod.MASS_WEIGHTED,
    line_color="green",
    width_indicator="full",
):
    """
    Compute 2nd-order cumulant between v1 and v2 and sample and perpendicular
    to pricinple axis. `s_N` sets plot window

    width_indicator:
        `full`: draw width indicator with label
        `floating_legend`: width indicator with label in legend below axes
        `no_label`: draw indicator without label
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    # add grid lines
    ax.axhline(0.0, color="grey", linestyle=":", alpha=0.3)
    ax.axvline(0.0, color="grey", linestyle=":", alpha=0.3)

    assert v1.shape == v2.shape
    if v1.dims == ("x", "y"):
        Nx, Ny = v1.shape
    elif v1.dims == ("y", "x"):
        Ny, Nx = v1.shape
    else:
        raise NotImplementedError

    assert np.all(v1.coords["x"] == v2.coords["x"])
    assert np.all(v1.coords["y"] == v2.coords["y"])

    if width_est_method == WidthEstimationMethod.CUTOFF:
        width_func = _find_width_through_cutoff
    elif width_est_method == WidthEstimationMethod.MASS_WEIGHTED:
        width_func = _find_width_through_mass_weighting
    else:
        raise NotImplementedError(width_est_method)

    # TODO: don't actually need 2D coords here, but would have to fix indexing
    # below
    if v1.dims == ("x", "y"):
        x, y = np.meshgrid(v1.coords["x"], v1.coords["y"], indexing="ij")
    else:
        x, y = np.meshgrid(v1.coords["x"], v1.coords["y"])

    C_vv = calc_2nd_cumulant(v1, v2, mask=mask)
    if sample_angle is not None:
        theta = xr.DataArray(
            sample_angle * pi / 180.0,
            attrs=dict(units="radians"),
            coords=dict(zt=v1.zt),
        )
    else:
        theta = identify_principle_axis(C_vv, sI_N=theta_win_N)

    # principle direction line
    mu_l, C_vv_l = _line_sample(data=C_vv, theta=theta, max_dist=max_dist)
    (line_1,) = ax.plot(
        mu_l,
        C_vv_l,
        label=r"$\theta^{{p}}={:.1f}^{{\circ}}$" "".format(theta.values * 180.0 / pi),
        color=line_color,
    )
    width = width_func(C_vv, theta)

    def _make_width_indicator(label, y0, width, color, linestyle="-"):
        eb = ax.errorbar(
            x=0.0, xerr=width / 2.0, y=y0, color=color, capsize=4.0, linestyle=linestyle
        )
        eb[-1][0].set_linestyle(linestyle)
        if width_indicator == "full":
            ax.text(
                s=label,
                x=0.0,
                y=y0,
                verticalalignment="bottom",
                horizontalalignment="center",
            )

    if width_indicator is not None:
        # we'll place the width indicator based on the y-range of the plot using
        # the side of the y=0 line where there is the most space
        ylim = np.array(ax.get_ylim())
        i_max = np.argmax(np.abs(ylim))
        y_max = ylim[i_max]
        if abs(ylim[[1, 0][i_max]] / ylim[i_max]) > 0.3:
            y_max = 1.3 * y_max
        y_ref = sorted([0.4 * y_max, 0.2 * y_max])
        _make_width_indicator(
            label="$L^p$",
            width=width,
            y0=y_ref[0],
            color=line_1.get_color(),
            linestyle="-",
        )

    # perpendicular direction line
    mu_l, C_vv_l = _line_sample(data=C_vv, theta=theta + pi / 2.0, max_dist=max_dist)
    (line_2,) = ax.plot(
        mu_l,
        C_vv_l,
        label=r"$\theta^\bot=\theta^p + 90^{\circ}$",
        linestyle="--",
        color=line_1.get_color(),
    )
    if width_indicator is not None:
        width = width_func(C_vv, theta + pi / 2.0)
        _make_width_indicator(
            label=r"$L^{\bot}$",
            width=width,
            y0=y_ref[1],
            color=line_1.get_color(),
            linestyle="--",
        )
    if width_indicator == "floating_legend":
        add_width_indicator_legend(ax=ax)

    if with_45deg_sample:
        mu_l, C_vv_l = _line_sample(
            data=C_vv, theta=theta + pi / 4.0, max_dist=max_dist
        )
        (line_2,) = ax.plot(mu_l, C_vv_l, label=r"$\theta=\theta^{p} + 45^{\circ}$")
        width = width_func(C_vv, theta + pi / 2.0, width_peak_fraction)
        ax.axvline(-0.5 * width, linestyle="--", color=line_2.get_color())
        ax.axvline(0.5 * width, linestyle="--", color=line_2.get_color())

    lines = [line_1, line_2]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax.set_xlabel("distance [m]")
    ax.set_ylabel("covariance [{}]".format(C_vv.units))
    # ax.text(0.05, 0.8, r"$\theta_{{p}}={:.1f}^{{\circ}}$".format(theta.values*180./pi),
    # transform=ax.transAxes
    # )

    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

    try:
        v1.zt
        z_var = "zt"
    except AttributeError:
        z_var = "zm"

    ax.set_title(
        "{} sampled along and\n perpendicular to principle axis "
        "at z={z}{z_units}\n".format(
            C_vv.long_name, z=float(v1[z_var]), z_units=v1[z_var].units
        )
    )

    return [line_1, line_2]


def charactistic_scales(
    v1,
    v2=None,
    l_theta_win=1000.0,
    mask=None,
    sample_angle=None,
    width_est_method=WidthEstimationMethod.MASS_WEIGHTED,
):
    """
    From 2nd-order cumulant of v1 and v2 compute principle axis angle,
    characteristic length-scales along and perpendicular to principle axis (as
    full width at half maximum)
    """
    if v2 is not None:
        assert v1.shape == v2.shape
        assert np.all(v1.coords["x"] == v2.coords["x"])
        assert np.all(v1.coords["y"] == v2.coords["y"])

    if width_est_method == WidthEstimationMethod.CUTOFF:
        width_func = _find_width_through_cutoff
    elif width_est_method == WidthEstimationMethod.MASS_WEIGHTED:
        width_func = _find_width_through_mass_weighting
    else:
        raise NotImplementedError

    Nx, Ny = v1.shape

    dx = np.max(np.gradient(v1.x))

    C_vv = calc_2nd_cumulant(v1, v2, mask=mask)

    l_win = l_theta_win
    m = 0
    if sample_angle is not None:
        theta = xr.DataArray(
            sample_angle * pi / 180.0,
            attrs=dict(units="radians"),
            coords=dict(zt=v1.zt),
        )

        width_principle_axis = width_func(C_vv, theta)
        width_perpendicular = width_func(C_vv, theta + pi / 2.0)
    else:
        while True:
            s_N = int(l_win / dx) * 2
            theta = identify_principle_axis(C_vv, sI_N=s_N)

            width_principle_axis = width_func(C_vv, theta)
            width_perpendicular = width_func(C_vv, theta + pi / 2.0)

            d_width = np.abs(width_perpendicular - width_principle_axis)
            mean_width = 0.5 * (width_perpendicular + width_principle_axis)

            if np.isnan(width_perpendicular) or np.isnan(width_principle_axis):
                break

            if (
                d_width / mean_width > 0.30
                and width_perpendicular > width_principle_axis
            ):
                l_win = 1.2 * l_win

                m += 1
                if m > 10:
                    warnings.warn("Couldn't find principle axis")
                    width_principle_axis = np.nan
                    width_perpendicular = np.nan
                    theta = np.nan
                    # break
            else:
                break

            # else:

    theta_deg = xr.DataArray(
        theta.values * 180.0 / pi, dims=theta.dims, attrs=dict(units="deg")
    )

    dataset = xr.Dataset(
        dict(
            principle_axis=theta_deg,
            width_principle=width_principle_axis,
            width_perpendicular=width_perpendicular,
            is_covariant=C_vv[Nx // 2, Ny // 2] > 0,
        )
    )
    dataset["cumulant"] = C_vv.name

    return dataset
