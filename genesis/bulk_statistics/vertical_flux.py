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
if __name__ == "__main__":
    matplotlib.use("Agg")

    # register a progressbar so we can see progress of dask'ed operations with xarray
    from dask.diagnostics import ProgressBar
    ProgressBar().register()

import matplotlib.pyplot as plt

from . import load_mask, load_field, scale_field
from . import get_dataset

import genesis.objects


def main(args=None, ax=None):
    import argparse
    argparser = argparse.ArgumentParser(__doc__)

    argparser.add_argument('input')
    argparser.add_argument('var_name')
    argparser.add_argument('--time', nargs="+", type=float, required=False)
    argparser.add_argument('--z_max', type=float, default=650.)

    mask_type_args = argparser.add_mutually_exclusive_group()
    mask_type_args.add_argument('--mask-name', default=None, type=str)
    mask_type_args.add_argument('--objects', default=None, type=str)

    argparser.add_argument('--mask-field', default=None, type=str)
    argparser.add_argument('--invert-mask', default=False, action="store_true")
    argparser.add_argument('--output-in-cwd', default=False, action='store_true')
    argparser.add_argument('--with-legend', default=False, action='store_true')
    argparser.add_argument('--skip-interval', default=1, type=int)
    argparser.add_argument('--output-format', default='png', type=str, choices=['png', 'pdf'])
    argparser.add_argument('--mask-mean', default=False, action="store_true")
    argparser.add_argument('--scale-by-area', default=False, action="store_true")

    args = argparser.parse_args(args=args)

    output_fn = "{}.{}.vertical_flux.{}".format(
        args.input.replace('/', '__'), args.var_name,
        args.output_format
    )

    if args.mask_name is not None:
        mask = load_mask(input_name=args.input, mask_name=args.mask_name,
                         mask_field=args.mask_field, invert=args.invert_mask)
        output_fn = output_fn.replace(
            '.{}'.format(args.output_format),
            '.masked.{}.{}'.format(mask.name, args.output_format)
        )
    elif args.objects is not None:
        s = args.input.replace('/3d_blocks', '').replace('/', '__')
        objects_fn = "{}.objects.{}".format(s, args.objects)
        mask = genesis.objects.make_mask_from_objects_file(filename=objects_fn)

        if args.invert_mask:
            mask_attrs = mask.attrs
            mask = ~mask
            mask.attrs.update(mask_attrs)
            mask.name = "{}__inverted".format(mask.name)

        output_fn = output_fn.replace(
            '.{}'.format(args.output_format),
            '.masked.{}.{}'.format(mask.name, args.output_format)
        )
    else:
        mask = None

    input_name = args.input
    case_name = input_name.split('/')[0]
    dataset_name_with_time = input_name.split('/')[-1]

    if not args.mask_mean:
        ds_3d = get_dataset(dataset_name_with_time,
                            variables=['{}_flux'.format(args.var_name)],
                            p='{}/3d_blocks/full_domain/'.format(case_name))
        da_3d = ds_3d['{}_flux'.format(args.var_name)]
        if mask is not None:
            da_3d = da_3d.where(mask)

        if args.z_max is not None:
            da_3d = da_3d.sel(zt=slice(None, args.z_max))

        da_3d.name = '{}_flux'.format(args.var_name)
        da_3d = scale_field(da_3d)

        da_mean = da_3d.mean(dim=('xt', 'yt'))
        da_mean.attrs['longname'] = 'horizontal mean {}'.format(da_3d.longname)
        da_mean.attrs['units'] = da_3d.units

    else:
        if mask is None:
            raise Exception("Must provide mask to compute mask mean")
        v = 'd_{}'.format(args.var_name)
        ds_3d = get_dataset(dataset_name_with_time,
                            variables=[v, 'w_zt'],
                            p='{}/3d_blocks/full_domain/'.format(case_name))

        w_mean = ds_3d[v].where(mask).mean(dim=('xt', 'yt'))
        v_mean = ds_3d['w'].where(mask).mean(dim=('xt', 'yt'))
        da_mean = w_mean*v_mean

        da_mean.attrs['longname'] = '{} flux'.format(v)
        da_mean.attrs['units'] = "m/s {}".format(ds_3d[v].units)
        da_mean.name = '{}_flux'.format(args.var_name)
        da_mean = scale_field(da_mean)
        print(42)
        output_fn = output_fn.replace('_flux', '_mask_mean_flux')

    if args.scale_by_area and mask is not None:
        N_cells_mask = mask.sum(dim=('xt', 'yt')).sel(zt=da_mean.zt)
        nx, ny = len(mask.xt), len(mask.yt)
        N_cells_total = nx*ny
        da_mean *= N_cells_mask.astype(float)/float(N_cells_total)
        output_fn = output_fn.replace('.vertical_flux.', '.area_weighted_vertical_flux.')

    da_mean.plot(y='zt', ax=ax)

    mask_description = ''
    if not mask is None:
        if 'longname' in mask.attrs:
            mask_description = mask.attrs['longname']

        if args.invert_mask:
            mask_description = "not " + mask_description


    plt.title("Vertical {} flux {} in {}{}".format(
        args.var_name,
        ["", "scaled by area"][not args.scale_by_area is None],
        args.input,
        ["", "\nwith '{}' mask".format(mask_description)][not mask is None]
    ))

    if ax is None:
        plt.savefig(output_fn, bbox_inches='tight')
        print("Plots saved to {}".format(output_fn))


if __name__ == "__main__":
    main()
