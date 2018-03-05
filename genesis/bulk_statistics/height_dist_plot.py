"""
Functions in this module produce analysis hierarchy level 1 plots of the
boundary layer distribution of varius scalars.

To produce a full set the following variables are needed in 3D datasets from
UCLALES:

- w: vertical velocity
- q: total water vapour
- t: liquid potential temperature
"""

import xarray as xr
import matplotlib.pyplot as plot
import numpy as np


def height_dist_plot(dataset, var_name, t, dz, scaling=25.*4, z_max=700.,
                     dvar=0.05, mask_zero=False):
    calc_flux = var_name.endswith('flux')
    if calc_flux:
        var_name = var_name.replace('_flux', '')

    z_var = dataset[var_name].coords.values()[1]

    def calc_bins(v_, dv_):
        try:
            v_min = dv_*int(v_.min()/dv_)
            v_max = dv_*int(v_.max()/dv_)
        except ValueError:
            return None, 1

        n = int((v_max - v_min)/dv_)

        if n == 0:
            return None, 1
        else:
            return (v_min, v_max), n

    def get_zdata(zvar_name, z_):
        return dataset.sel(**{zvar_name: z_, "drop": True, "time":t})

    for k, z_ in enumerate(z_var):
        dataset_ = get_zdata(z_var.name, z_=z_)

        d = np.array(dataset_[var_name])

        units = dataset_[var_name].units

        if calc_flux:
            z_up = float(z_) + 0.5*dz
            w_up = get_zdata('zm', z_up)['w']
            z_down = float(z_) - 0.5*dz
            if np.any(z_down > 0.0):
                w_down = get_zdata('zm', z_down)['w']
            else:
                w_down = np.zeros_like(d)
            w_c = 0.5*(w_up + w_down)

            d = np.array(w_c*(d - d.mean()))

            units += ' m/s'

        if mask_zero:
            d = d[~(d == 0)]

        bin_range, n_bins = calc_bins(d, dvar)
        bin_counts, bin_edges = np.histogram(
            d, normed=True, bins=n_bins, range=bin_range
        )
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        s = scaling/bin_counts.max()
        plot.plot(bin_centers, bin_counts*s+np.array(z_), marker='.',)
        plot.axhline(z_, linestyle=':', color='grey', alpha=0.3)

        # print float(z_), d.min(), d.max(), dataset_[var_name].shape

        if z_ > z_max:
            break

    if d.min() < 0.0 < d.max():
        plot.axvline(0.0, linestyle=':', color='grey')

    if calc_flux:
        var_label = var_name
        if var_name == 't':
            var_label = '\\theta_l'

        plot.xlabel(r"$\overline{w'%s'}$ [%s]" % (var_label, units))
    else:
        plot.xlabel('{} [{}]'.format(dataset[var_name].longname, units))
    # plot.ylabel('{} [{}]'.format(z_var.longname, z_var.units))
    plot.ylabel('{} [{}]'.format("height", z_var.units))

    plot.title("t={}s".format(t))


def find_dz(dataset):
    dz_ = np.diff(dataset.zt)

    assert np.min(dz_) == np.max(dz_)

    return np.min(dz_)


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(__doc__)

    argparser.add_argument('input.nc')
    default_vars = "q t w q_flux t_flux".split(" ")
    argparser.add_argument('--vars', nargs="+", default=default_vars)
    argparser.add_argument('--time', nargs="+", type=float, required=True)
    argparser.add_argument('--z_max', type=float, default=650.)

    args = argparser.parse_args()

    num_vars = len(args.vars)
    num_timesteps = len(args.time)
    plot.figure(figsize=(5*num_timesteps,3*num_vars))

    default_scalings = dict(
        q=400.,
        w=200.
    )

    default_binsize = dict(
        q=0.00008,
        t=0.005,
        t_flux=1.0e-2,
        q_flux=4.0e-5
    )

    dataset = xr.open_dataset(vars(args)['input.nc'], decode_times=False)

    dz = find_dz(dataset)

    for n, var_name in enumerate(args.vars):
        for m, t in enumerate(args.time):
            plot.subplot(num_vars, num_timesteps, n*num_timesteps + m+1)
            scaling = default_scalings.get(var_name, 100.)
            binsize = default_binsize.get(var_name, 0.1)
            height_dist_plot(dataset, var_name, dz=dz, scaling=scaling,
                             t=t, dvar=binsize, z_max=args.z_max)

    out_filename = 'bl_hist_plots.pdf'

    plot.tight_layout()

    plot.savefig(out_filename)

    print("Plots saved to {}".format(out_filename))
