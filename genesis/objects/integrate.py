import os
import warnings

import dask_image.ndmeasure
import numpy as np
import xarray as xr
from scipy.constants import pi
from tqdm import tqdm

from ..utils import find_grid_spacing
from . import integral_properties, minkowski_scales

# forget about using dask for now, dask_ndmeasure takes a huge amount of memory
# try:
# # raise ImportError
# # import dask_ndmeasure as ndimage
# # register a progressbar so we can see progress of dask'ed operations with xarray
# from dask.diagnostics import ProgressBar
# ProgressBar().register()
# except ImportError:
# from scipy import ndimage
# warnings.warn("Using standard serial scipy implementation instead of "
# "dask'ed dask-ndmeasure. Install `dask-ndmeasure` for much "
# "faster computation")


CHUNKS = 200  # forget about using dask for now, np.unique is too slow

FN_OUT_FORMAT = "{base_name}.objects.{objects_name}.integral.{name}.nc"


def make_name(variable, operator=None):
    if operator:
        return "{variable}.{operator}".format(**locals())
    else:
        return variable


def _integrate_scalar(objects, da, operator):
    if "object_ids" in da.coords:
        object_ids = da.object_ids
    else:
        # print("Finding unique values")
        object_ids = np.unique(objects.chunk(None).values)
        # ensure object 0 (outside objects) is excluded
        if object_ids[0] == 0:
            object_ids = object_ids[1:]

    if len(da.dims) == 1 and len(objects.dims) == 3:
        # special case for allowing integration of coordinates
        da = xr.broadcast(objects, da)[1]
    else:
        import ipdb

        with ipdb.launch_ipdb_on_exception():
            assert objects.dims == da.dims
            assert objects.shape == da.shape

    dx = find_grid_spacing(da)

    s = None
    if operator == "volume_integral":
        # fn = ndimage.sum
        fn = dask_image.ndmeasure.sum
        s = dx ** 3.0
        operator_units = "m^3"
    elif operator == "maximum_pos_z":
        fn = dask_image.ndmeasure.maximum_position
        operator_units = "m"
    else:
        # fn = getattr(ndimage, operator)
        fn = getattr(dask_image.ndmeasure, operator)
        operator_units = ""

    # NB: the argument to `ndmeasure` functions used to be `labels` rather than
    # `label_image` before dask-iamge v0.5.0
    vals = fn(da, label_image=objects.values, index=object_ids)
    if hasattr(vals, "compute"):
        vals = vals.compute()

    if s is not None:
        vals *= s

    if operator == "maximum_pos_z":
        longname = "per-object z-pos of maximum {} value".format(da.name)
        units = "m"
        z_idxs = vals[:, da.dims.index("zt")]
        vals = da.zt[z_idxs]
    else:
        longname = "per-object {} of {}".format(operator.replace("_", " "), da.name)
        units = ("{} {}".format(da.units, operator_units)).strip()

    da_integrated = xr.DataArray(
        vals,
        coords=dict(object_id=object_ids),
        dims=("object_id",),
        attrs=dict(longname=longname, units=units),
        name="{}__{}".format(da.name, operator),
    )

    if da.name == "volume":
        da_integrated.name = "volume"

    return da_integrated


def _integrate_per_object(da_objects, fn_int):
    if "object_ids" in da_objects.coords:
        object_ids = da_objects.object_ids
    else:
        # print("Finding unique values")
        object_ids = np.unique(da_objects.chunk(None).values)
        # ensure object 0 (outside objects) is excluded
        if object_ids[0] == 0:
            object_ids = object_ids[1:]

    if "xt" in da_objects.coords:
        da_objects = da_objects.rename(dict(xt="x", yt="y", zt="z"))

    ds_per_object = []
    for object_id in tqdm(object_ids):
        da_object = da_objects.where(da_objects == object_id, drop=True)

        ds_object = fn_int(da_object)
        ds_object["object_id"] = object_id
        ds_per_object.append(ds_object)

    return xr.concat(ds_per_object, dim="object_id")


def get_integration_requirements(variable):
    if variable.endswith("_vertical_flux"):
        var_name = variable[: len("_vertical_flux")]
        return dict(w="w", scalar=var_name)
    else:
        return {}


def integrate(objects, variable, operator=None, **kwargs):
    """
    Integrate over the labelled objects in `objects` the variable (named by a
    string, .e.g `r_equiv` would be the equivalent spherical radius). Can also
    integrate for example a scalar field provided through an extra kwarg to
    find for example the maximum value.

    Available variables:

        {avail_vars}

    Calculating equivalent radius for each object:

    >> integrate(da_objects, variable='r_equiv')

    Calculate the maximum value of vertical velocity for each object

    >> integrate(da_objects, variable='w', operator='maximum', w=da_w)

    Calculate the volume integral of water vapour for each object

    >> integrate(da_objects, variable='q', operator='volume_integral', q=ds.q)
    """

    ds_out = None

    if variable in objects.coords:
        da_scalar = objects.coords[variable]
    elif variable == "com_angles":
        fn_int = integral_properties.calc_com_incline_and_orientation_angle
        ds_out = _integrate_per_object(da_objects=objects, fn_int=fn_int)
    elif hasattr(integral_properties, "calc_{}__dask".format(variable)):
        fn_int = getattr(integral_properties, "calc_{}__dask".format(variable))
        da_objects = objects
        if "xt" in da_objects.dims:
            da_objects = da_objects.rename(xt="x", yt="y", zt="z")
        ds_out = fn_int(da_objects)
        try:
            ds_out.name = variable
        except AttributeError:
            # we can't actually set the name of a dataset, this only works with
            # data arrays
            pass
    elif hasattr(integral_properties, "calc_{}".format(variable)):
        fn_int = getattr(integral_properties, "calc_{}".format(variable))
        ds_out = _integrate_per_object(da_objects=objects, fn_int=fn_int)
        try:
            ds_out.name = variable
        except AttributeError:
            # we can't actually set the name of a dataset, this only works with
            # data arrays
            pass
    # XXX: volume is actually calculated by the minkowski routines which have
    # been verified against those below (keeping in case I forget)
    # elif variable == 'volume':
    # dx = find_grid_spacing(objects)
    # da_scalar = xr.DataArray(
    # np.ones_like(objects, dtype=np.float),
    # coords=objects.coords, attrs=dict(units='1')
    # )
    # da_scalar.name = 'volume'
    elif variable in [
        "length_m",
        "width_m",
        "thickness_m",
        "genus_m",
        "volume",
        "num_cells",
        "filamentarity",
        "planarity",
    ]:
        ds_minkowski = minkowski_scales.main(da_objects=objects)
        ds_out = ds_minkowski[variable]
    elif variable == "r_equiv":
        da_volume = integrate(objects, "volume", operator="sum")
        # V = 4/3 pi r^3 => r = (3/4 V/pi)**(1./3.)
        da_scalar = (3.0 / (4.0 * pi) * da_volume) ** (1.0 / 3.0)
        da_scalar.attrs["units"] = "m"
        da_scalar.attrs["long_name"] = "equivalent sphere radius"
        da_scalar.name = "r_equiv"
        ds_out = da_scalar
    elif variable in kwargs and operator in [
        "volume_integral",
        "maximum",
        "maximum_pos_z",
        "mean",
        "sum",
    ]:
        da_scalar = kwargs[variable].squeeze()
        if not objects.zt.equals(da_scalar.zt):
            warnings.warn(
                "Objects span smaller range than scalar field to "
                "reducing domain of scalar field"
            )
            da_scalar = da_scalar.sel(zt=objects.zt)

        # ds_out = _integrate_scalar(objects=objects.squeeze(),
        # da=da_scalar,
        # operator=operator)
        import ipdb

        with ipdb.launch_ipdb_on_exception():
            ds_out = _integrate_scalar(objects=objects, da=da_scalar, operator=operator)
    else:
        if operator:
            raise NotImplementedError(
                f"Don't know how to calculate `{operator}` of `{variable}` with fields"
                f"{', '.join(kwargs.keys())}`"
            )
        else:
            raise NotImplementedError(
                "Don't know how to calculate `{}`" "".format(variable)
            )
    # else:
    # fn_scalar = "{}.{}.nc".format(base_name, variable)
    # if not os.path.exists(fn_scalar):
    # raise Exception("Couldn't find scalar file `{}`".format(fn_scalar))

    # da_scalar = xr.open_dataarray(
    # fn_scalar, decode_times=False, chunks=CHUNKS
    # ).squeeze()

    if ds_out is None:
        if objects.zt.max() < da_scalar.zt.max():
            warnings.warn(
                "Objects span smaller range than scalar field to "
                "reducing domain of scalar field"
            )
            zt_ = da_scalar.zt.values
            da_scalar = da_scalar.sel(zt=slice(None, zt_[25]))

        ds_out = _integrate_scalar(objects=objects, da=da_scalar, operator=operator)

    return ds_out


# hack to set docstring at runtime so we can include the available variables
integrate.__doc__ = integrate.__doc__.format(
    avail_vars=", ".join(integral_properties.VAR_MAPPINGS.keys())
)

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(__doc__)
    argparser.add_argument("object_file")
    argparser.add_argument("scalar_field")
    argparser.add_argument("--operator", default="volume_integral", type=str)

    args = argparser.parse_args()
    object_file = args.object_file.replace(".nc", "")

    op = args.operator

    if "objects" not in object_file:
        raise Exception()

    base_name, objects_mask = object_file.split(".objects.")

    fn_objects = "{}.nc".format(object_file)
    if not os.path.exists(fn_objects):
        raise Exception("Couldn't find objects file `{}`".format(fn_objects))
    objects = xr.open_dataarray(fn_objects, decode_times=False, chunks=CHUNKS).squeeze()

    name = make_name(
        variable=args.scalar_field,
        operator=op,
    )
    out_filename = FN_OUT_FORMAT.format(
        base_name=base_name.replace("/", "__"), objects_name=objects.name, name=name
    )

    ds_out = integrate(
        objects=objects, variable=args.scalar_field, operator=args.operator
    )

    import ipdb

    with ipdb.launch_ipdb_on_exception():
        ds_out.to_netcdf(out_filename)
    print("Wrote output to `{}`".format(out_filename))
