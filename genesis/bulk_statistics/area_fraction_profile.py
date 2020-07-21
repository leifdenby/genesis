"""
"""
import matplotlib

if __name__ == "__main__": # noqa
    matplotlib.use("Agg")

    # register a progressbar so we can see progress of dask'ed operations with xarray
    from dask.diagnostics import ProgressBar

    ProgressBar().register()

import matplotlib.pyplot as plt

from . import load_mask

import genesis.objects


def main(args=None, ax=None):
    import argparse

    argparser = argparse.ArgumentParser(__doc__)

    argparser.add_argument("input")
    argparser.add_argument("--time", nargs="+", type=float, required=False)
    argparser.add_argument("--z_max", type=float, default=650.0)

    mask_type_args = argparser.add_mutually_exclusive_group(required=True)
    mask_type_args.add_argument("--mask-name", default=None, type=str)
    mask_type_args.add_argument("--objects", default=None, type=str)

    argparser.add_argument("--mask-field", default=None, type=str)
    argparser.add_argument("--invert-mask", default=False, action="store_true")
    argparser.add_argument("--output-in-cwd", default=False, action="store_true")
    argparser.add_argument("--with-legend", default=False, action="store_true")
    argparser.add_argument("--skip-interval", default=1, type=int)
    argparser.add_argument(
        "--output-format", default="png", type=str, choices=["png", "pdf"]
    )

    args = argparser.parse_args(args=args)

    if args.mask_name is not None:
        mask = load_mask(
            input_name=args.input,
            mask_name=args.mask_name,
            mask_field=args.mask_field,
            invert=args.invert_mask,
        )
    elif args.objects is not None:
        s = args.input.replace("/3d_blocks", "").replace("/", "__")
        objects_fn = "{}.objects.{}".format(s, args.objects)
        mask = genesis.objects.make_mask_from_objects_file(filename=objects_fn)
    else:
        raise Exception("A mask is required")

    output_fn = "{}__area_fraction.{}.{}".format(
        args.input.replace("/", "__"), mask.name, args.output_format
    )

    nx, ny = len(mask.xt), len(mask.yt)
    area_fraction = mask.sum(dim=("xt", "yt")).astype("float") / (nx * ny)

    area_fraction.attrs["units"] = "1"
    area_fraction.attrs["longname"] = "area fraction"
    area_fraction.name = "area_fraction"

    area_fraction.plot(y="zt")

    mask_description = ""
    if mask is not None:
        if "longname" in mask.attrs:
            mask_description = mask.attrs["longname"]

        if args.invert_mask:
            mask_description = "not " + mask_description

    plt.title("Area fraction flux in {} of {}".format(args.input, mask_description))

    if ax is None:
        plt.savefig(output_fn, bbox_inches="tight")
        print("Plots saved to {}".format(output_fn))


if __name__ == "__main__":
    main()
