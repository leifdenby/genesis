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

from . import get_distribution_in_cross_sections, load_mask
from . import make_cumulative_from_bin_counts

import genesis.objects


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__)

    argparser.add_argument("input")
    argparser.add_argument("var_name")
    default_vars = "q t w q_flux t_flux".split(" ")
    argparser.add_argument("--time", nargs="+", type=float, required=False)
    argparser.add_argument("--z_max", type=float, default=650.0)

    mask_type_args = argparser.add_mutually_exclusive_group()
    mask_type_args.add_argument("--mask-name", default=None, type=str)
    mask_type_args.add_argument("--objects", default=None, type=str)

    argparser.add_argument("--mask-field", default=None, type=str)
    argparser.add_argument("--invert-mask", default=False, action="store_true")
    argparser.add_argument("--output-in-cwd", default=False, action="store_true")
    argparser.add_argument("--with-legend", default=False, action="store_true")
    argparser.add_argument("--cumulative", default=False, action="store_true")
    argparser.add_argument("--skip-interval", default=1, type=int)
    argparser.add_argument(
        "--output-format", default="png", type=str, choices=["png", "pdf"]
    )
    argparser.add_argument("--y-max", default=None, type=float)

    args = argparser.parse_args()

    field_fn = "{}.{}.nc".format(args.input, args.var_name)
    output_fn = "height_dist_plot__{}.{}".format(
        field_fn.replace("/", "__"), args.output_format
    )

    if args.mask_name is not None:
        mask = load_mask(
            input_name=args.input,
            mask_name=args.mask_name,
            mask_field=args.mask_field,
            invert=args.invert_mask,
        )
        output_fn = output_fn.replace(
            ".{}".format(args.output_format),
            ".masked.{}.{}".format(mask.name, args.output_format),
        )
    elif args.objects is not None:
        s = args.input.replace("/3d_blocks", "").replace("/", "__")
        objects_fn = "{}.objects.{}".format(s, args.objects)
        mask = genesis.objects.make_mask_from_objects_file(filename=objects_fn)
        output_fn = output_fn.replace(
            ".{}".format(args.output_format),
            ".masked.{}.{}".format(mask.name, args.output_format),
        )
    else:
        mask = None

    var_bin_counts = get_distribution_in_cross_sections(
        fn=field_fn,
        dv_bin=5,
        z_slice=slice(0, args.z_max, args.skip_interval),
        mask=mask,
    )

    if len(var_bin_counts.time) == 1:
        var_bin_counts = var_bin_counts.squeeze()
    else:
        if args.time is None:
            raise Exception(
                "More than one timesteps is available in {} "
                "please indicate which timestep to plot"
                "".format(field_fn)
            )
        else:
            var_bin_counts = var_bin_counts.sel(time=args.time).squeeze()

    if args.cumulative:
        output_fn = output_fn.replace("height_dist_", "heigth_cumulative_dist_")

        var_cumulative = make_cumulative_from_bin_counts(var_bin_counts)
        plot_var = var_cumulative
    else:
        plot_var = var_bin_counts

    if hasattr(plot_var, "tex_label"):
        plot_var.attrs["long_name"] = plot_var.tex_label

    v = plot_var.dims[-1]
    if hasattr(plot_var[v], "tex_label"):
        plot_var[v].attrs["long_name"] = plot_var[v].tex_label

    plot_var.plot.line(hue="zt")

    if args.y_max is not None:
        plt.gca().set_ylim(None, args.y_max)

    mask_description = ""
    if not mask is None:
        if "longname" in mask.attrs:
            mask_description = mask.attrs["longname"]

        if args.invert_mask:
            mask_description = "not " + mask_description

    plt.title(
        "{}distribution in {}{}".format(
            ["", "Cumulative "][args.cumulative],
            args.input,
            ["", "\nwith '{}' mask".format(mask_description)][not mask is None],
        )
    )

    plt.savefig(output_fn, bbox_inches="tight")

    print("Plots saved to {}".format(output_fn))
