#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
import xarray as xr


def plot_ballistic(object_id, da_):
    da_obj = da_.sel(object_id=object_id)

    fig, ax = plt.subplots()
    da_obj.plot(ax=ax, y="cldtop")

    idx = da_obj.argmax(dim="cldtop")
    idx = idx.where(idx != 0, other=np.nan).dropna(dim="time_relative").astype(int)
    z_top = da_obj.sel(time_relative=idx.time_relative).isel(cldtop=idx).cldtop
    z_top["t"] = z_top.time_relative * 60.0
    z_top.t.attrs["units"] = "seconds"
    z_top = z_top.swap_dims(dict(time_relative="t"))

    v = z_top.differentiate(coord="t") / z_top.t.differentiate(coord="t")
    v.attrs["units"] = "m/s"
    v.attrs["long_name"] = "vertical velocity"

    a = v.differentiate(coord="t") / v.t.differentiate(coord="t")
    a.attrs["units"] = "m/s^2"
    a.attrs["long_name"] = "accelleration"

    # t = z_top.time_relative

    fig, axes = plt.subplots(nrows=3, figsize=(5, 6), sharex=True)
    z_top.plot(ax=axes[0])
    ax = axes[1]
    v.plot(ax=ax)
    ax.axhline(0.0, linestyle="--", color="grey")
    ax = axes[2]
    a.plot(ax=ax)
    ax.axhline(0.0, linestyle="--", color="grey")
    sns.despine()
    plt.tight_layout()

    ax.set_xlim(0, 30 * 60)

    return z_top["t"], z_top


def parcel_rise(t, p):
    x0, v0, a0 = p
    return x0 + v0 * t + 0.5 * a0 * 1.0e-3 * t ** 2.0


def fit_model(z, t, verbose=False):
    """
    Fit parcel-rise model with height `z` [m] at time `t` [s]
    """
    y_data = z.values
    x_data = t.values

    with pm.Model() as model_g:
        a0 = pm.Normal("a0", mu=0.0)  # Prior for the acceleration
        v0 = pm.Normal("v0", mu=1.0, sigma=2.0)
        x0 = pm.Normal("x0", mu=800.0, sigma=10.0)

        # prior for our estimated standard deviation of the error
        mu = pm.HalfNormal("mu", sigma=10)

        pm.Normal(
            "z_pred", mu=parcel_rise(x_data, [x0, v0, a0]), sd=mu, observed=y_data
        )

        # Explore and Sample the Parameter Space!
        trace_g = pm.sample(1000, tune=500, progressbar=verbose)

    return trace_g, model_g


def _extract_time(da_z, time_dim="time_relative"):
    # make the units be seconds
    z_top = da_z.where(da_z > 0.0, drop=True)
    z_top["t"] = z_top[time_dim] * 60.0
    z_top.t.attrs["units"] = "seconds"
    z_top = z_top.swap_dims(dict(time_relative="t"))

    return z_top, z_top["t"]


def fit_model_and_summarise(
    da_z, time_dim="time_relative", predictions=None, verbose=False
):
    """
    predictions:
        None: only returns the fitting parameters
        mean: includes the mean fit over time
        mean_with_quantiles: include lower 2.5% and upper 92.5% quantiles of predictions
    """

    z, t = _extract_time(da_z, time_dim=time_dim)

    trace_g, model_g = fit_model(z=z, t=t, verbose=verbose)

    z0, v0, a0 = trace_g["x0"].mean(), trace_g["v0"].mean(), trace_g["a0"].mean()
    z0_stderr = trace_g["x0"].std()

    ds = xr.Dataset(coords=dict(t=t))
    ds["z0"] = z0
    ds["z0_stderr"] = z0_stderr
    ds["mu"] = trace_g["mu"].mean()
    ds["v0"] = v0
    ds["a0"] = a0
    ds["z_data"] = z

    p = [z0, v0, a0]
    ds["z"] = parcel_rise(t, p)

    if predictions == "mean_with_quantiles":
        chain_count = trace_g.get_values("a0").shape[0]
        y_pred_g = pm.sample_posterior_predictive(
            trace_g,
            samples=chain_count,
            model=model_g,
            progressbar=verbose,
        )

        crit_l = np.percentile(
            y_pred_g["z_pred"], q=2.5, axis=0
        )  # grab lower 2.5% quantiles
        crit_u = np.percentile(
            y_pred_g["z_pred"], q=97.5, axis=0
        )  # grab Upper 97.5% quantiles
        mean_spp = np.mean(y_pred_g["z_pred"], axis=0)  # Median

        ds["z__median"] = ("t",), mean_spp
        ds["z__upperlim"] = ("t",), crit_u
        ds["z__lowerlim"] = ("t",), crit_l

    return ds


def plot_fitted_model(ds_fit):
    ds_fit.z.plot()
    ds_fit.z_data.plot()

    ax = plt.gca()
    ax.fill_between(ds_fit.t, ds_fit.z__lowerlim, ds_fit.z__upperlim, alpha=0.2)
    ax.set_ylim(0, None)
