"""
Functions in this module produce analysis hierarchy level 1 plots of the
boundary layer distribution of varius scalars.

To produce a full set the following variables are needed in 3D datasets from
UCLALES:

- w: vertical velocity
- q: total water vapour
- t: liquid potential temperature
"""
import os
import matplotlib
matplotlib.use("Agg")

import xarray as xr
import matplotlib.pyplot as plot
import numpy as np
import textwrap


def height_dist_plot(dataset, var_name, t, scaling=25.*4, z_max=700.,
                     dvar=0.05, cumulative=False, marker='.',
                     offset=True, skip_interval=1, mask=None):

    z_var = dataset[var_name].coords[
        'zt' if 'zt' in dataset[var_name].coords else 'zm'
    ]

    def calc_bins(v_, dv_):
        try:
            v_min = dv_*np.floor(v_.min()/dv_)
            v_max = dv_*np.ceil(v_.max()/dv_)
        except ValueError:
            return None, 1

        n = int(np.round((v_max - v_min)/dv_))

        if n == 0:
            return None, 1
        else:
            return (v_min, v_max), n

    def get_zdata(zvar_name, z_, dataset=dataset):
        return dataset.sel(**{zvar_name: z_, "drop": True, "time":t})

    lines = []

    for k, z_ in enumerate(z_var[::skip_interval]):
        dataset_ = get_zdata(z_var.name, z_=z_)

        d = dataset_[var_name].values
        nx, ny = d.shape

        units = dataset_[var_name].units.replace(' ', '*')

        if not mask is None:
            mask_slice = get_zdata(zvar_name=z_var.name, z_=z_, dataset=mask)
            d = d[mask_slice.values]

        bin_range, n_bins = calc_bins(d, dvar)
        bin_counts, bin_edges = np.histogram(
            d, normed=True, bins=n_bins, range=bin_range
        )

        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

        if cumulative:
            # bin_counts = np.cumsum(bin_counts[::-1])[::-1]
            total_flux = bin_counts*bin_centers
            # bin_counts = np.cumsum(total_flux)
            ncells = nx*ny
            bin_counts = np.cumsum(total_flux[::-1])[::-1]/ncells

        if not cumulative:
            s = scaling/bin_counts.max()
        else:
            s = 1.0

        l, = plot.plot(
            bin_centers,
            bin_counts*s+offset*np.array(z_),
            marker=marker,
            label="z={}m".format(z_.values)
        )

        if offset:
            plot.axhline(z_, linestyle=':', color='grey', alpha=0.3)

        lines.append(l)

        # print float(z_), d.min(), d.max(), dataset_[var_name].shape

        if z_ > z_max:
            break

    if d.min() < 0.0 < d.max():
        plot.axvline(0.0, linestyle=':', color='grey')

    is_flux = var_name.endswith('flux')
    if is_flux:
        var_label = var_name.replace('_flux', '')
        if var_label == 't':
            var_label = '\\theta_l'

        plot.xlabel(r"$\overline{w'%s'}$ [%s]" % (var_label, units))
    else:
        plot.xlabel('{} [{}]'.format(dataset[var_name].longname, units))
    # plot.ylabel('{} [{}]'.format(z_var.longname, z_var.units))

    if cumulative:
        text = 'cumulative {} [{}]'.format(
            dataset[var_name].longname,dataset[var_name].units.replace(' ', '*')
        )
        plot.ylabel(textwrap.fill(text, 20))
    else:
        plot.ylabel('{} [{}]'.format("height", z_var.units))

    plot.title("t={}s".format(t))

    return lines


def _add_flux_var(dataset, var_name):
    def find_dz(dataset):
        z_var = 'zt' if 'zt' in dataset[var_name].coords else 'zm'
        dz_ = np.diff(dataset[z_var])

        assert np.min(dz_) == np.max(dz_)

        return np.min(dz_)

    dz = find_dz(dataset)
    z_up = float(z_) + 0.5*dz
    w_up = get_zdata('zm', z_up)['w']
    z_down = float(z_) - 0.5*dz
    if np.any(z_down > 0.0):
        w_down = get_zdata('zm', z_down)['w']
    else:
        w_down = np.zeros_like(d)
    w_c = 0.5*(w_up + w_down)

    d = dataset[var_name].values
    d = np.array(w_c*(d - d.mean()))

    units = '{} m/s'.format(dataset[var_name].units)

    dataset['{}_flux'.format(var_name)] = (
        dataset[var_name].dims,
        d,
        dict(units=units)
    )

    return dataset


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(__doc__)

    argparser.add_argument('input')
    default_vars = "q t w q_flux t_flux".split(" ")
    argparser.add_argument('--vars', nargs="+", default=default_vars)
    argparser.add_argument('--time', nargs="+", type=float, required=False)
    argparser.add_argument('--z_max', type=float, default=650.)
    argparser.add_argument('--cumulative', default=False, action="store_true")
    argparser.add_argument('--bin-marker', default='', type=str)
    argparser.add_argument('--mask-name', default=None, type=str)
    argparser.add_argument('--invert-mask', default=False, action="store_true")
    argparser.add_argument('--output-in-cwd', default=False, action='store_true')
    argparser.add_argument('--skip-interval', default=1, type=int)
    argparser.add_argument('--no-offset', default=False, action='store_true')
    argparser.add_argument('--with-legend', default=False, action='store_true')

    args = argparser.parse_args()

    num_vars = len(args.vars)

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

    input_name = args.input

    if input_name.endswith('.nc'):
        dataset = xr.open_dataset(input_name, decode_times=False)

        for var_name in enumerate(args.vars):
            if var_name.endswith('_flux') and not var_name in dataset:
                _add_flux_var(dataset, var_name.replace('_flux', ''))

        out_filename = input_name.replace('.nc', '.bl_hist_plots.pdf')
    else:
        out_filename = input_name + '.bl_hist_plots.pdf'

        # have to handle `w` seperately because it is staggered
        filenames = [
            "{}.{}.nc".format(input_name, var_name) for var_name in args.vars
            if not var_name == 'w'
        ]
        missing = [
            fn for fn in filenames if not os.path.exists(fn)
        ]

        if len(filenames) > 0:
            if len(missing) > 0:
                raise Exception("Missing files: {}".format(", ".join(missing)))

            dataset = xr.open_mfdataset(filenames, decode_times=False,
                                        concat_dim=None, chunks=dict(zt=20))
        else:
            dataset = None

        if 'w' in args.vars:
            d2 = xr.open_dataset("{}.w.nc".format(input_name),
                                 decode_times=False, chunks=dict(zm=20))

            if not dataset is None:
                dataset = xr.merge([dataset, d2])
            else:
                dataset = d2

    if args.cumulative is True:
        out_filename = out_filename.replace('.pdf', '.cumulative.pdf')

    if not args.mask_name is None:
        fn_mask = "{}.{}.mask.nc".format(input_name, args.mask_name)
        if not os.path.exists(fn_mask):
            raise Exception("Can't find mask file `{}`".format(fn_mask))
        mask = xr.open_dataarray(fn_mask, decode_times=False)
        if args.invert_mask:
            mask_attrs = mask.attrs
            mask = ~mask
            out_filename = out_filename.replace(
                '.pdf', '.masked.not__{}.pdf'.format(args.mask_name)
            )
            mask.attrs.update(mask_attrs)
        else:
            out_filename = out_filename.replace(
                '.pdf', '.masked.{}.pdf'.format(args.mask_name)
            )
    else:
        mask = None

    if args.time is None:
        num_timesteps = 1
        times = dataset.time.values
    else:
        num_timesteps = len(args.time)
        times = args.time

    if args.output_in_cwd:
        out_filename = out_filename.replace('/', '__')

    plot.figure(figsize=(5*num_timesteps,3*num_vars))

    for n, var_name in enumerate(args.vars):
        for m, t in enumerate(times):
            plot.subplot(num_vars, num_timesteps, n*num_timesteps + m+1)
            scaling = default_scalings.get(var_name, 100.)
            binsize = default_binsize.get(var_name, 0.1)
            lines = height_dist_plot(dataset, var_name, scaling=scaling,
                                     t=t, dvar=binsize, z_max=args.z_max,
                                     cumulative=args.cumulative,
                                     marker=args.bin_marker, mask=mask,
                                     skip_interval=args.skip_interval,
                                     offset=not args.no_offset)


    if args.with_legend:
        plot.figlegend(
            lines, [l.get_label() for l in lines],
            loc='lower center', ncol=3
        )

    plot.tight_layout()

    if args.with_legend:
        plot.subplots_adjust(top=0.85, bottom=0.2)

    print fn_mask, mask.attrs

    mask_description = ''
    if not mask is None:
        if 'longname' in mask.attrs:
            mask_description = mask.attrs['longname']
        else:
            mask_description = args.mask_name

        if args.invert_mask:
            mask_description = "not " + mask_description

    plot.suptitle("{}distribution in {}{}".format(
        ["", "Cumulative "][args.cumulative],
        input_name,
        ["", "\nwith '{}' mask".format(mask_description)][not mask is None]
    ))

    plot.savefig(out_filename)

    print("Plots saved to {}".format(out_filename))
