"""
Utilities for calculating characteristics scales of objects using Minkowski
functionals
"""
import os

import xarray as xr

from . import topology


def main(da_objects):
    dx = topology.minkowski.find_grid_spacing(da_objects)
    return topology.minkowski.calc_scales(object_labels=da_objects, dx=dx)


FN_FORMAT = "{base_name}.objects.{objects_name}.minkowski_scales.nc"

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__)

    argparser.add_argument("objects_file", type=str)
    argparser.add_argument("--make-plot", action="store_true")

    args = argparser.parse_args()

    object_file = args.objects_file.replace(".nc", "")

    if "objects" not in object_file:
        raise Exception()

    base_name, objects_mask = object_file.split(".objects.")

    out_filename = FN_FORMAT.format(base_name=base_name, objects_name=objects_mask)

    fn_objects = "{}.nc".format(object_file)
    if not os.path.exists(fn_objects):
        raise Exception("Couldn't find objects file `{}`".format(fn_objects))
    objects = xr.open_dataarray(fn_objects, decode_times=False)

    ds = main(objects)

    ds.attrs["input_name"] = args.objects_file
    ds.attrs["mask"] = objects_mask

    ds.to_netcdf(out_filename)
    print("Wrote output to `{}`".format(out_filename))

    if args.make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import topology.plots.overview

        topology.plots.overview.filamentarity_planarity.fp_plot(ds=ds)
        fn_plot = out_filename.replace(".nc", ".pdf")
        plt.savefig(fn_plot)
        print("Saved plot to {}".format(fn_plot))
