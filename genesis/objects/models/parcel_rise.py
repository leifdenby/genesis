#!/usr/bin/env python
# coding: utf-8

# In[1]:


import luigi
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from genesis.utils.pipeline.data.tracking_2d import (
    ExtractCloudbaseState,
    PerformObjectTracking2D,
    TrackingType,
    AllObjectsAll2DCrossSectionAggregations,
    TrackingVariable2D
)

from genesis.utils.pipeline.viz.tracking_2d import (
    CloudCrossSectionAnimationFrame
)

import datetime

import pymc3 as pm
from pymc3.ode import DifferentialEquation
import theano
from scipy.integrate import odeint
import arviz as az


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:




def plot_ballistic(object_id):
    da_obj = da_.sel(object_id=object_id)
    
    fig, ax = plt.subplots()
    da_obj.plot(ax=ax, y='cldtop')

    idx = da_obj.argmax(dim="cldtop")
    idx = idx.where(idx!=0, other=np.nan).dropna(dim="time_relative").astype(int)
    z_top = da_obj.sel(time_relative=idx.time_relative).isel(cldtop=idx).cldtop
    z_top['t'] = z_top.time_relative*60.0
    z_top.t.attrs['units'] = 'seconds'
    z_top = z_top.swap_dims(dict(time_relative='t'))

    v = z_top.differentiate(coord='t')/z_top.t.differentiate(coord='t')
    v.attrs['units'] = 'm/s'
    v.attrs['long_name'] = 'vertical velocity'

    a = v.differentiate(coord='t')/v.t.differentiate(coord='t')
    a.attrs['units'] = 'm/s^2'
    a.attrs['long_name'] = 'accelleration'
    
    t = z_top.time_relative

    fig, axes = plt.subplots(nrows=3, figsize=(5, 6), sharex=True)
    z_top.plot(ax=axes[0])
    ax = axes[1]
    v.plot(ax=ax)
    ax.axhline(0.0, linestyle='--', color='grey')
    ax = axes[2]
    a.plot(ax=ax)
    ax.axhline(0.0, linestyle='--', color='grey')
    sns.despine()
    plt.tight_layout()
    
    ax.set_xlim(0, 30*60)
    
    return z_top['t'], z_top
    
#plot_ballistic(5007)
#plot_ballistic(5074)
plot_ballistic(da_.isel(object_id=-500).object_id)

def parcel_rise(t, p):
    x0, v0, a0 = p
    return x0 + v0*t + 0.5*a0*1.0e-3*t**2.0


def fit_model(z, t):
    """
    Fit parcel-rise model with height `z` [m] at time `t` [s]
    """
    y_data = z.values
    x_data = t.values

    with pm.Model() as model_g:
        a0 = pm.Normal('a0', mu=0.0)  # Prior for the acceleration
        v0 = pm.Normal('v0', mu=1.0, sigma=2.0)
        x0 = pm.Normal('x0', mu=800., sigma=10.0)

        # prior for our estimated standard deviation of the error
        mu = pm.HalfNormal('mu', sigma=10)

        z_pred = pm.Normal('z_pred', mu=parcel_rise(x_data, [x0, v0, a0]), sd=mu, observed=y_data)

        # Explore and Sample the Parameter Space!
        trace_g = pm.sample(1000, tune=500, cores=2)

    return trace_g, model_g


def fit_model_and_summarise(z, t, predictions=None):
    """
    predictions:
        None: only returns the fitting parameters
        mean: includes the mean fit over time
        mean_with_quantiles: include lower 2.5% and upper 92.5% quantiles of predictions
    """
    trace_g, model_g = fit_model()

    chain_count =  trace_g.get_values('a0').shape[0]
    y_pred_g = pm.sample_posterior_predictive(trace_g, samples=chain_count, model=model_g)

    crit_l = np.percentile(y_pred_g["z_pred"],q=2.5,axis=0)  # grab lower 2.5% quantiles
    crit_u = np.percentile(y_pred_g["z_pred"],q=97.5,axis=0)  # grab Upper 92.5% quantiles
    mean_spp = np.mean(y_pred_g["z_pred"], axis=0) # Median

    y0, v0, a0 = trace_g['x0'].mean(), trace_g['v0'].mean(), trace_g['a0'].mean()


def plot_fitted_model():
    p = [y0, alpha, beta]
    y = parcel_rise(times, p)
    yobs = z

    fig, ax = plt.subplots()
    plt.plot(times, yobs, label='observed height', linestyle='dashed', marker='o', color='red')
    plt.plot(times, y, label='model', color='k', alpha=0.5)

    crit_l = np.percentile(y_pred_g["z_pred"], q=2.5, axis=0)  # grab lower 2.5% quantiles
    crit_u = np.percentile(y_pred_g["z_pred"], q=97.5, axis=0)  # grab Upper 92.5% quantiles
    mean_spp = np.mean(y_pred_g["z_pred"], axis=0) # Median
    plt.plot(times, mean_spp, linewidth=4, color="#5500ff")
    plt.fill_between(times, crit_l, crit_u, alpha=0.2, color="#00cc66")

    plt.legend()
    plt.xlabel('Time (Seconds)')
    plt.ylabel(r'$z(t)$');
    plt.show()

    az.summary(trace_g)
