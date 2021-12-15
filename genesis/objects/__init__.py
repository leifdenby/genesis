import glob
import os

import xarray as xr

try:
    # reduce moved in python3
    from functools import reduce
except ImportError:
    pass


def get_data(base_name, mask_identifier="*", debug=False):
    print(base_name, mask_identifier)
    glob_patterns = [
        "{}.objects.{}.minkowski_scales.nc".format(base_name, mask_identifier),
        "{}.objects.{}.integral.*.nc".format(base_name, mask_identifier),
    ]

    fns = reduce(lambda a, s: glob.glob(s) + a, glob_patterns, [])

    if debug:
        print("Loading:\n\t" + "\n\t".join(fns))

    if len(fns) == 0:
        raise Exception(
            "No files found with glob patterns: {}".format(", ".join(glob_patterns))
        )

    try:
        ds = xr.open_mfdataset(fns)
    except Exception:
        print("Error while loading:\n\t{}".format("\n\t".join(fns)))
        raise

    # 4/3*pi*r**3 => r = (3/4*1/pi*v)**(1./3.)
    if "volume__sum" in ds.data_vars:
        ds["r_equiv"] = (3.0 / (4.0 * 3.14) * ds.volume__sum) ** (1.0 / 3.0)
        ds.r_equiv.attrs["units"] = "m"
        ds.r_equiv.attrs["longname"] = "equivalent radius"

    return ds


def make_mask_from_objects_file(filename):
    object_file = filename.replace(".nc", "")

    if "objects" not in object_file:
        raise Exception()

    base_name, mask_name = object_file.split(".objects.")

    fn_objects = "{}.nc".format(object_file)
    if not os.path.exists(fn_objects):
        raise Exception("Couldn't find objects file `{}`".format(fn_objects))
    objects = xr.open_dataarray(fn_objects, decode_times=False)

    mask = objects != 0
    mask.name = "{}_objects".format(objects.mask_name)
    mask.attrs["longname"] = "mask from {} objects".format(objects.mask_name)

    return mask


from . import flux_contribution  # noqa
from . import topology  # noqa
from . import filter, identify, integrate  # noqa
