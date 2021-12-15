import os
import warnings

import matplotlib.font_manager
import matplotlib.pyplot as plot
import numpy as np
import scipy.optimize
import seaborn as sns
import xarray as xr

try:
    import tephigram

    HAS_TEPHIGRAM = True
except ImportError:
    HAS_TEPHIGRAM = False

from tqdm import tqdm

sns.set(style="ticks", color_codes=True)
sns.despine()

VARS = [
    ("zbmn", "zcmn", "zb", "zc"),
    (
        "lwp_bar",
        "rwp_bar",
    ),
    ("cfrac",),
]

N_rows = len(VARS) + 2


def _plot_overview(dataset_name, t_hrs_used):
    fn = os.path.join(
        "cross_sections", "runtime_slices", "{}.out.xy.lwp.nc".format(dataset_name)
    )

    if not os.path.exists(fn):
        raise Exception("Can't find `{}`, needed for overview plots" "".format(fn))
    da = xr.open_dataarray(fn, decode_times=False)

    if not da.time.units.startswith("seconds"):
        raise Exception(
            "The `{}` has the incorrect time units (should be"
            " seconds) which is likely because cdo mangled it."
            " Recreate the file making sure that cdo is explicitly"
            " told to use a *relative* time axis. The current"
            " units are `{}`.".format(fn, da.time.units)
        )

    print(
        "Using times `{}` for overview plots".format(
            ", ".join([str(v) for v in t_hrs_used])
        )
    )

    # rescale distances to km
    def scale_dist(da, c):
        assert da[c].units == "m"
        da[c].values = da[c].values / 1000.0
        da[c].attrs["units"] = "km"
        da[c].attrs["standard_name"] = "horz. dist."

    scale_dist(da, "xt")
    scale_dist(da, "yt")

    axes_overview = []

    for n, t_ in enumerate(tqdm(t_hrs_used)):
        ax = plot.subplot2grid((N_rows, len(t_hrs_used)), (N_rows - 1, n))

        da_ = da.sel(
            time=t_ * 60.0 * 60.0, drop=True, tolerance=5.0 * 60.0, method="nearest"
        ).squeeze()

        (da_ > 0.001).plot.pcolormesh(
            cmap=plot.get_cmap("Greys_r"),
            rasterized=True,
            ax=ax,
            add_colorbar=False,
        )

        plot.title("t={}hrs".format(t_))
        plot.gca().set_aspect(1)

        axes_overview.append(ax)

    axes_overview[0].get_shared_y_axes().join(*axes_overview)
    [ax.autoscale() for ax in axes_overview]
    [ax.set_ylabel("") for ax in axes_overview[1:]]
    [ax.set_yticklabels([]) for ax in axes_overview[1:]]


def _plot_timeseries(dataset_name):
    fn = os.path.join("other", "{}.ts.nc".format(dataset_name))
    ds_ts = xr.open_dataset(fn, decode_times=False)

    assert ds_ts.time.units.startswith("seconds since 2000-01-01 0000")
    ds_ts.time.values = ds_ts.time.values / 60.0 / 60.0
    ds_ts.time.attrs["units"] = "hrs"

    font = matplotlib.font_manager.FontProperties()
    font.set_weight("bold")

    axes_ts = []
    for n, varset in enumerate(VARS):
        ax = plot.subplot2grid((N_rows, 1), (n, 0))
        axes_ts.append(ax)
        for var in varset:
            ax = plot.gca()
            ax.set_xlabel("time [hours]")

            da = ds_ts[var]
            da.plot(ax=ax, label=da.longname)

            if var in ("lwp_bar", "rwp_bar"):
                ax.set_ylim(0, None)
            else:
                ax.set_ylim(0, 1.2 * da[:].max())

            ax.grid(True)
            # ticks = 240.*np.arange(0, 12)
            # ax.xticks(ticks)

        if len(varset) > 1:
            ax.legend()
        else:
            ax.set_title(da.longname)

    axes_ts[0].get_shared_x_axes().join(*axes_ts)
    [ax.autoscale() for ax in axes_ts]

    # use the tick marks to determine where we'll make the overview and profile
    # plots
    t_hrs_used = list(filter(lambda v: v > 0.0, ax.get_xticks()[1:-1]))

    return t_hrs_used


def _plot_3d_data_indicators(dataset_name):
    fn_3d = os.path.join("raw_data", "{}.00000000.nc".format(dataset_name))
    if os.path.exists(fn_3d):
        data_3d = xr.open_dataset(fn_3d, decode_times=False)

        x_, y_ = data_3d.time / 60.0 / 60.0, np.zeros_like(data_3d.time)
        plot.plot(x_, y_, marker="o", linestyle="", color="red")

        ax = plot.gca()
        for tn_, (x__, y__) in enumerate(zip(x_, y_)):
            ax.annotate(
                tn_,
                xy=(x__, y__),
                xytext=(0, 10),
                color="red",
                textcoords="offset pixels",
                horizontalalignment="center",
                verticalalignment="bottom",
            )


def calc_temperature(q_l, p, theta_l):
    # constants from UCLALES
    cp_d = 1.004 * 1.0e3  # [J/kg/K]
    R_d = 287.04  # [J/kg/K]
    L_v = 2.5 * 1.0e6  # [J/kg]
    p_theta = 1.0e5

    # XXX: this is *not* the *actual* liquid potential temperature (as
    # given in B. Steven's notes on moist thermodynamics), but instead
    # reflects the form used in UCLALES where in place of the mixture
    # heat-capacity the dry-air heat capacity is used
    def temp_func(T):
        return theta_l - T * (p_theta / p) ** (R_d / cp_d) * np.exp(
            -L_v * q_l / (cp_d * T)
        )

    if np.all(q_l == 0.0):
        # no need for root finding
        return theta_l / ((p_theta / p) ** (R_d / cp_d))

    # XXX: brentq solver requires bounds, I don't expect we'll get below -100C
    T_min = -100.0 + 273.0
    T_max = 50.0 + 273.0
    T = scipy.optimize.brentq(f=temp_func, a=T_min, b=T_max)

    # check that we're within 1.0e-4
    assert np.all(np.abs(temp_func(T)) < 1.0e-4)

    return T


def _UCLALES_est_temperature(ds):

    da_pressure = ds.p
    da_thetal = ds.t
    da_rl = ds.l
    da_ql = da_rl / (da_rl + 1.0)

    try:
        from multiprocessing import Pool

        arr_temperature = Pool().starmap(
            calc_temperature, zip(da_ql, da_pressure, da_thetal)
        )
    except Exception:
        arr_temperature = np.vectorize(calc_temperature)(
            q_l=da_ql, p=da_pressure, theta_l=da_thetal
        )

    da_temperature = xr.DataArray(
        arr_temperature,
        dims=da_pressure.dims,
        attrs=dict(longname="temperature", units="K"),
    )

    return da_temperature


def _plot_profiles(dataset_name, t_hrs_used):
    fn = os.path.join("other", "{}.ps.nc".format(dataset_name))
    ds_ps = xr.open_dataset(fn, decode_times=False)

    assert ds_ps.time.units.startswith("seconds since 2000-01-01 0000")
    ds_ps.time.values = ds_ps.time.values / 60.0 / 60.0
    ds_ps.time.attrs["units"] = "hrs"

    for n, t_ in enumerate(tqdm(t_hrs_used)):
        N_t = len(t_hrs_used)
        subplotshape = (N_rows, N_t, len(VARS) * N_t + n + 1)
        fig = plot.gcf()

        ds_ = ds_ps.sel(time=t_, method="nearest")

        da_pressure = ds_.p
        warnings.warn("Assuming that output is from UCLALES")
        da_temperature = _UCLALES_est_temperature(ds=ds_)
        da_rh = ds_.RH

        tephi = tephigram.Tephigram(
            fig=fig,
            subplotshape=subplotshape,
            default_lines="no_labels",
            y_range=(0.0, 15),
            x_range=(0.0, 60),
        )
        tephi.plot_temp(T=da_temperature - 273.15, P=da_pressure / 100.0, marker="")
        tephi.plot_RH(
            T=da_temperature - 273.15, P=da_pressure / 100.0, RH=da_rh, marker=""
        )


def main(dataset_name, dt_overview_hours=5):
    _ = plot.figure(figsize=(10, 12))

    t_hrs_used = _plot_timeseries(dataset_name=dataset_name)
    _plot_3d_data_indicators(dataset_name=dataset_name)
    _plot_overview(dataset_name=dataset_name, t_hrs_used=t_hrs_used)

    if HAS_TEPHIGRAM:
        _plot_profiles(dataset_name=dataset_name, t_hrs_used=t_hrs_used)


# x = np.linspace(0, 20, 100)

# tns = [480, 960, 1440, 1920]


# x, y = cloud_data.grid
# data = cloud_data.get('lwp', tn=tn)

# plot.pcolormesh(x/1000, y/1000, data > 0.001, cmap=plot.get_cmap('Greys_r'))
# plot.xlabel('horizontal dist. [km]')
# plot.xlim(-25, 25)
# plot.ylim(-25, 25)
# if n == 0:
# plot.ylabel('horizontal dist. [km]')
# plot.title('t={}min'.format(tn))
# plot.gca().set_aspect(1)

# for p_label, p_bounds in periods.items():
# ax.axvspan(p_bounds[0], p_bounds[1], alpha=0.5, color='grey')
# ylim = ax.get_ylim()
# ax.text(.5*(p_bounds[0] + p_bounds[1]), 0.75*ylim[1], p_label, color='white',
# fontsize=16, fontproperties=font, horizontalalignment='center')

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument("--filetype", default="pdf")
    argparser.add_argument("--dt_overview_hours", type=int)
    args = argparser.parse_args()

    import glob

    files = glob.glob("other/*.ts.nc")
    if len(files) == 0:
        raise Exception("Can't find *.ts.nc file, needed for this plot")
    dataset_name = os.path.basename(files[0]).split(".")[0]

    print("Plotting evolution for `{}`".format(dataset_name))

    main(dataset_name, dt_overview_hours=args.dt_overview_hours)

    plot.tight_layout()
    plot.savefig("{}.evolution.{}".format(dataset_name, args.filetype))
