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

from . import load_mask, load_field
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

    args = argparser.parse_args(args=args)

    field_fn = '{}.{}_flux.nc'.format(args.input, args.var_name)
    output_fn = "flux_plot__{}.{}".format(
        field_fn.replace('/', '__'), args.output_format
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
        output_fn = output_fn.replace(
            '.{}'.format(args.output_format),
            '.masked.{}.{}'.format(mask.name, args.output_format)
        )
    else:
        mask = None

    try:
        da_3d = load_field(fn=field_fn, mask=mask)
    except:
        input_name = args.input
        case_name = input_name.split('/')[0]
        dataset_name_with_time = input_name.split('/')[-1]
        ds_3d = get_dataset(dataset_name_with_time,
                            variables=['{}_flux'.format(args.var_name)],
                            p='{}/3d_blocks/full_domain/'.format(case_name))
        da_3d = ds_3d['{}_flux'.format(args.var_name)]
        if mask is not None:
            da_3d = da_3d.where(mask)

    if args.z_max is not None:
        da_3d = da_3d.sel(zt=slice(None, args.z_max))

    da_mean = da_3d.mean(dim=('xt', 'yt'))
    da_mean.attrs['longname'] = 'horizontal mean {}'.format(da_3d.longname)
    da_mean.attrs['units'] = da_3d.units
    da_mean.plot(y='zt', ax=ax)

    mask_description = ''
    if not mask is None:
        if 'longname' in mask.attrs:
            mask_description = mask.attrs['longname']

        if args.invert_mask:
            mask_description = "not " + mask_description


    plt.title("Vertical {} flux in {}{}".format(
        args.var_name,
        args.input,
        ["", "\nwith '{}' mask".format(mask_description)][not mask is None]
    ))

    if ax is None:
        plt.savefig(output_fn, bbox_inches='tight')
        print("Plots saved to {}".format(output_fn))


if __name__ == "__main__":
    main()
