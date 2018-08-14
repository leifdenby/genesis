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
# matplotlib.use("Agg")

import xarray as xr
import matplotlib.pyplot as plot
import numpy as np
import textwrap
from tqdm import tqdm

from . import get_dataset


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

cp_d = 1005.46 # [J/kg/K]
L_v = 2.5008e6  # [J/kg]
rho0 = 1.2  # [kg/m^3]

def calc_distribution_in_cross_sections(s_da, ds_bin):
    """
    `s_da`: scalar being binned in horizontal cross-sections
    `ds_bin`: bin-size
    """
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

    z_var = s_da.coords[
        'zt' if 'zt' in s_da.coords else 'zm'
    ]

    def get_zdata(zvar_name, z_, dataset=dataset):
        return dataset.sel(**{zvar_name: z_, "drop": True})

    lines = []

    # get data will be working on
    z = z_var.sel(zt=slice(z_min, z_max, skip_interval))
    nz = z.shape[0]
    da_in = s_da.sel(**{ zvar.name: z, "drop": True })

    # setup array for output
    v_range, n_bins = calc_bins(da_in, dv=ds_bin)
    bins = np.linspace(v_range[0], v_range[1], num=n_bins)

    def calc_distribution_in_cross_section(da_cross):
        da_counts = da_cross.groupby_bins(bins=bins)
        pass

    return da_in.groupby('z').apply(calc_distribution_in_cross_section)


    for k, z_ in enumerate(tqdm(z__)):
        if z_ < z_min:
            continue
        elif z_ > z_max:
            break

        dataset_ = get_zdata(z_var.name, z_=z_)

        d = dataset_[var_name].values
        nx, ny = d.shape

        units = dataset_[var_name].units.replace(' ', '*')

        if not mask is None:
            if not 'zt' in mask and len(mask.shape):
                if not mask.dims == dataset_[var_name].dims:
                    raise Exception("Problem with dimensions on 2D mask: "
                                    "mask: {} vs data: {}".format(mask.dims,
                                    dataset_[var_name].dims))

                mask_slice = mask
            else:
                mask_slice = get_zdata(zvar_name=z_var.name, z_=z_, dataset=mask)
            d = d[mask_slice.values]

        if 'g/kg' in units:
            # XXX: UCLALES has the wrong units it its output files!!!
            # it says that g/kg is outputted, but it is actually kg/kg
            # d *= 1000.
            units = units.replace('g/kg', 'kg/kg')

        is_flux = var_name.endswith('flux')
        if is_flux:
            var_label = var_name.replace('_flux', '')

            if scale_fluxes:
                if var_label == 't':
                    scaling = cp_d*rho0
                    var_label = r"$\rho_0 c_{p,d} w'\theta_l'$"
                elif var_label == 'q':
                    scaling = L_v*rho0
                    var_label = r"$\rho_0 L_v w'q_t'$"
                else:
                    raise NotImplementedError

                units = "W/m$^2$"
                xlabel = r"{} [{}]".format(var_label, units)
            else:
                if var_label == 't':
                    var_label = r'\theta_l'

                var_label = "$w'{}'$".format(var_label)

                xlabel = r"%s [%s]" % (var_label, units)

        else:
            xlabel = '{} [{}]'.format(dataset[var_name].longname, units)


        bin_range, n_bins = calc_bins(d, binsize)
        bin_counts, bin_edges = np.histogram(
            d, normed=not cumulative, bins=n_bins, range=bin_range
        )

        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

        if cumulative:
            # bin_counts = np.cumsum(bin_counts[::-1])[::-1]
            total_flux = bin_counts*bin_centers
            # bin_counts = np.cumsum(total_flux)
            ncells = nx*ny
            if reverse_cumulative:
                bin_counts = np.cumsum(total_flux[::-1])[::-1]/ncells
            else:
                bin_counts = np.cumsum(total_flux)/ncells

        if not cumulative:
            s = scaling/bin_counts.max()
        else:
            if scaling is not None:
                s = scaling
            else:
                s = 1.0

def height_dist_plot(dataset, var_name, t, scaling=None, z_max=700.,
                     binsize=None, cumulative=False, z_min=0.0,
                     offset=True, skip_interval=1, mask=None, 
                     reverse_cumulative=True, ax=None, scale_fluxes=False,
                     **kwargs):

    if binsize is None:
        binsize = default_binsize.get(var_name, 0.1)
    if scaling is None:
        scaling = default_scalings.get(var_name, 100.)

    if ax is None:
        ax = plot.gca()


        l, = ax.plot(
            bin_centers,
            bin_counts*s+offset*np.array(z_),
            label="z={}m".format(z_.values),
            **kwargs
        )

        if offset:
            ax.axhline(z_, linestyle=':', color='grey', alpha=0.3)

        lines.append(l)

        # print float(z_), d.min(), d.max(), dataset_[var_name].shape

    if d.min() < 0.0 < d.max():
        ax.axvline(0.0, linestyle=':', color='grey')

    ax.set_xlabel(xlabel)
    # ax.set_ylabel('{} [{}]'.format(z_var.longname, z_var.units))

    if cumulative:
        text = r"cumulative {} [{}]".format(
            var_label, units
        )
        ax.set_ylabel(textwrap.fill(text, 40))
    else:
        ax.set_ylabel('{} [{}]'.format("height", z_var.units))

    ax.set_title("t={}s".format(t))

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
    argparser.add_argument('--mask-field', default=None, type=str)
    argparser.add_argument('--invert-mask', default=False, action="store_true")
    argparser.add_argument('--output-in-cwd', default=False, action='store_true')
    argparser.add_argument('--skip-interval', default=1, type=int)
    argparser.add_argument('--no-offset', default=False, action='store_true')
    argparser.add_argument('--with-legend', default=False, action='store_true')

    args = argparser.parse_args()

    num_vars = len(args.vars)

    input_name = args.input

    dataset, out_filename = get_dataset(
        input_name=input_name, generate_output_filename=True,
        variables=args.vars,
        output_fn_for='height_dist_plot'
    )

    if args.cumulative is True:
        out_filename = out_filename.replace('.pdf', '.cumulative.pdf')

    if not args.mask_name is None:
        if args.mask_field is None:
            mask_field = args.mask_name
            mask_description = args.mask_name
        else:
            mask_field = args.mask_field
            mask_description = "{}__{}".format(args.mask_name, args.mask_field)

        fn_mask = "{}.{}.mask.nc".format(input_name, args.mask_name)
        if not os.path.exists(fn_mask):
            raise Exception("Can't find mask file `{}`".format(fn_mask))

        ds_mask = xr.open_dataset(fn_mask, decode_times=False)
        if not mask_field in ds_mask:
            raise Exception("Can't find `{}` in mask, loaded mask file:\n{}"
                            "".format(mask_field, str(ds_mask)))
        else:
            mask = ds_mask[mask_field]

        if args.invert_mask:
            mask_attrs = mask.attrs
            mask = ~mask
            out_filename = out_filename.replace(
                '.pdf', '.masked.not__{}.pdf'.format(mask_description)
            )
            mask.attrs.update(mask_attrs)
        else:
            out_filename = out_filename.replace(
                '.pdf', '.masked.{}.pdf'.format(mask_description)
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
            lines = height_dist_plot(dataset, var_name,
                                     t=t, z_max=args.z_max,
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

    mask_description = ''
    if not mask is None:
        if 'longname' in mask.attrs:
            mask_description = mask.attrs['longname']
        else:
            mask_description = mask_description

        if args.invert_mask:
            mask_description = "not " + mask_description

    plot.suptitle("{}distribution in {}{}".format(
        ["", "Cumulative "][args.cumulative],
        input_name,
        ["", "\nwith '{}' mask".format(mask_description)][not mask is None]
    ))

    plot.savefig(out_filename, bbox_inches='tight')

    print("Plots saved to {}".format(out_filename))
