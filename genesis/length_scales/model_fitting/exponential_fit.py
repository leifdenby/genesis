import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from .pystan_cache import StanModel_cache
from .utils import dist_plot

SM_CODE = """
data {
    int<lower=1> n;
    real x[n];
}
parameters {
    real<lower=0> beta;
}
model {
    x ~ exponential(1.0/beta);
}
"""


def _sample_exp(Ntot, vrange, beta):
    vmin, vmax = vrange
    # alpha = Ntot / (beta * (np.exp(-vmin / beta) - np.exp(-vmax / beta)))
    x = np.random.exponential(scale=beta, size=Ntot)
    return x


def _fit_exp(v_data, debug=False):
    sm = StanModel_cache(model_code=SM_CODE)
    fit = sm.sampling(data=dict(x=v_data, n=len(v_data)))
    beta = fit["beta"]
    if debug:
        plt.hist(beta)
        print(fit)
        print(np.mean(beta), np.std(beta))
    return beta


def fit(da_v, dv=None, plot_to=None, debug=False, plot_components="default"):
    """
    Fit an exponential model to da_v, returns the mean and std div of beta the
    scale parameter of the distribution
    """
    # remove all nans and infs
    v_data = da_v
    v_data = v_data[~np.logical_or(np.isnan(v_data), np.isinf(v_data))]

    # only fit from the minimum limit, otherwise we'll be fitting from v=0
    # where aren't any objects
    vmin_fit = v_data.min().values
    vrange_fit = (vmin_fit, np.max(v_data).values)

    beta = _fit_exp(v_data[v_data > vmin_fit] - vmin_fit, debug=debug)

    if plot_to is not None:
        axes = None
        if isinstance(plot_to, np.ndarray) and isinstance(plot_to[0], plt.Axes):
            axes = plot_to
        log_dists = False
        fig, axes = dist_plot(
            v_data,
            dv_bin=dv,
            axes=axes,
            fit=("exp", (np.mean(beta), np.std(beta)), vrange_fit),
            log_dists=log_dists,
            components=plot_components,
        )
        v_sample = vmin_fit + _sample_exp(len(v_data), vrange_fit, np.mean(beta))
        dist_plot(
            v_sample,
            dv_bin=dv,
            axes=axes,
            alpha=0.6,
            log_dists=log_dists,
            components=plot_components,
        )
        fig.legend(axes[-1].get_lines(), ["data", "model"], loc="lower left")

    return xr.DataArray.from_dict(
        dict(
            dims="part",
            coords=dict(part=dict(dims="part", data=np.array(["mean", "std"]))),
            data=[np.mean(beta), np.std(beta)],
            attrs=da_v.attrs,
            name=da_v.name,
        )
    )

    return np.mean(beta), np.std(beta)
