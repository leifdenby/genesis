"""
Use 2D mask or object tracking to filter out objects and create a new objects
file
"""
import os
import re

import cloud_identification
import numpy as np
import tqdm
import xarray as xr

import genesis.objects

try:
    import cloud_tracking_analysis
    from cloud_tracking_analysis import CloudData

    HAS_CLOUD_TRACKING = True
except ImportError:
    HAS_CLOUD_TRACKING = False

from ..utils.pipeline.data.tracking_2d import TrackingType
from . import property_filters


def latex_format(filter_defs):
    if filter_defs is None:
        return ""
    s_latex = []

    for (f_type, f_def) in _defs_iterator(filter_defs):
        if f_type == "prop":
            s_latex.append(property_filters.latex_format(f_type, f_def))
        elif f_type == "tracking":
            s_latex.append("tracked {}".format(f_def[0]))
        else:
            raise NotImplementedError(f_type, f_def)
    return " and ".join(s_latex)


def _defs_iterator(filter_defs):
    s_filters = filter_defs.split(",")
    for s_filter in s_filters:
        try:
            f_type, f_cond = s_filter.split(":")
            if f_type == "prop":
                s_prop_and_op, s_value = f_cond.split("=")
                i = s_prop_and_op.rfind("__")
                if i == -1:
                    prop_name = s_prop_and_op
                    op_name = "eq"
                else:
                    prop_name, op_name = s_prop_and_op[:i], s_prop_and_op[i + 2 :]
                yield f_type, (prop_name, op_name, s_value)
            elif f_type == "tracking":
                yield f_type, (f_cond,)
            else:
                raise NotImplementedError(
                    "Filter type `{}` not recognised" "".format(f_type)
                )
        except (IndexError, ValueError):
            raise Exception("Malformed filter definition: `{}`".format(s_filter))


def parse_defs(filter_defs):
    filters = []

    for (f_type, f_def) in _defs_iterator(filter_defs):
        filter = dict(fn=None, reqd_props=[], extra=[])
        if f_type == "prop":
            prop_name, op_name, s_value = f_def
            fn = property_filters.make_filter_fn(prop_name, op_name, s_value)
            filter["reqd_props"].append(prop_name)
            filter["fn"] = fn
        elif f_type == "tracking":
            (tracking_type,) = f_def

            m = re.match(
                r"(?P<tracking_type>\w+)__(?P<prop>\w+)__(?P<op>\w+)=(?P<value>\d+)",
                tracking_type,
            )
            if m is not None:
                filter["extra_kws"] = dict(extra_filter=m.groupdict())
                filter["extra_kws"]["extra_filter"]["value"] = float(
                    filter["extra_kws"]["extra_filter"]["value"]
                )

            fn = filter_objects_by_tracking
            filter["fn"] = fn
            filter["extra"].append("cloud_tracking_data")
            filter["extra"].append("objects_3d")
        else:
            raise NotImplementedError
        filters.append(filter)

    return filters


def label_objects(mask, splitting_scalar=None, remove_at_edge=True):
    def _remove_at_edge(object_labels):
        mask_edge = np.zeros_like(mask)
        mask_edge[
            :, :, 1
        ] = True  # has to be k==1 because object identification codes treats actual edges as ghost cells
        mask_edge[:, :, -2] = True
        cloud_identification.remove_intersecting(object_labels, mask_edge)

    if splitting_scalar is None:
        splitting_scalar = np.ones_like(mask)
    else:
        assert mask.shape == splitting_scalar.shape
        assert mask.dims == splitting_scalar.dims
        # NB: should check coord values too

    object_labels = cloud_identification.number_objects(splitting_scalar, mask=mask)

    if remove_at_edge:
        _remove_at_edge(object_labels)

    return object_labels


def filter_objects_by_mask(objects, mask):
    if mask.dims != objects.dims:
        if mask.dims == ("yt", "xt"):
            assert objects.dims[:2] == mask.dims
            # mask_3d = np.zeros(objects.shape, dtype=bool)
            _, _, nz = objects.shape
            # XXX: this is pretty disguisting, there must be a better way...
            # inspired from https://stackoverflow.com/a/44151668
            mask_3d = np.moveaxis(np.repeat(mask.values[None, :], nz, axis=0), 0, 2)
        else:
            raise Exception(mask.dims)
    else:
        mask_3d = mask

    cloud_identification.remove_intersecting(objects, ~mask_3d)

    return objects


def filter_objects_by_tracking(
    da_objects, ds_objects_scales, cloud_tracking_data, extra_filter=None
):
    cloud_data = cloud_tracking_data

    t0 = da_objects.time.values

    ds_tracked = cloud_data._fh_track.sel(time=t0)
    objects_tracked_2d = ds_tracked.nrthrm

    if extra_filter is not None:
        if extra_filter["prop"] == "max_height" and extra_filter["op"] == "gt":

            trac_labels_2d = objects_tracked_2d.fillna(0.0).astype(int).values
            # remove object zero
            trac_ids_present = np.unique(trac_labels_2d)[1:]

            ds_tracked__max_height = ds_tracked.sel(
                smthrm=trac_ids_present
            ).smthrmtop.max(dim="smthrmt")

            are_ok = ds_tracked__max_height > float(extra_filter["value"])
            idxs_to_keep = are_ok.where(are_ok, drop=True).smthrm.astype(int).values

            labels__fake_3d = np.expand_dims(trac_labels_2d, axis=-1).astype(np.uint32)

            cloud_identification.filter_labels(
                labels=labels__fake_3d, idxs_keep=idxs_to_keep
            )

            da_mask = xr.DataArray(
                labels__fake_3d[..., 0],
                coords=objects_tracked_2d.coords,
                dims=objects_tracked_2d.dims,
            )
            objects_tracked_2d = da_mask.where(da_mask != 0)
        else:
            raise NotImplementedError(extra_filter)

    # mask the 3D objects identified so that we can find out what object IDs
    # should be kept
    da_objects_filtered = da_objects.where(~objects_tracked_2d.isnull())

    def filter_nans(v):
        return v[~np.isnan(v)]

    # remove nan values and object with id `0` (which is outside object)
    object_ids_filtered = filter_nans(np.unique(da_objects_filtered))[1:]

    ds_objects_scales_filtered = ds_objects_scales.sel(
        object_id=object_ids_filtered[1:]
    )

    return ds_objects_scales_filtered


def filter_objects_by_tracking_old(
    objects,
    base_name,
    dt_pad,
):
    t0 = objects.time.values
    valid_units = ["seconds since 2000-01-01 00:00:00", "seconds since 2000-01-01"]
    assert objects.time.units in valid_units
    t_min, t_max = t0 - dt_pad, t0 + dt_pad
    tracking_identifier = "{}s-{}s".format(t_min, t_max)

    def get_dataset_name_and_path(input_name):
        dataset_name = input_name.split("/")[-1].split(".")[0]
        dataset_path = input_name.split("/")[0]

        return dataset_name, dataset_path

    dataset_name, dataset_path = get_dataset_name_and_path(objects.input_name)

    # ensure cloud-tracking analysis is loading files relative to current path
    CloudData.set_root_path(os.getcwd())
    cloud_data = cloud_tracking_analysis.CloudData(
        dataset_name=dataset_name,
        dataset_pathname=dataset_path,
        tracking_identifier=tracking_identifier,
        tracking_type=TrackingType.THERMALS_ONLY,
    )

    ds_track_2d = cloud_data._fh_track.sel(time=objects.time)
    objects_tracked_2d = ds_track_2d.nrthrm

    # project 3D objects
    objects_projected_2d = objects.where(objects > 1, other=0).max(dim="zt")
    objects_projected_2d.name = "objects_projected_2d"

    # create mask which only includes regions which were tracked
    m = np.logical_and(~np.isnan(objects_tracked_2d), objects_projected_2d != 0)

    # mask out projected ids
    projected_labels_tracked = objects_projected_2d.where(m, other=np.nan)
    projected_labels_tracked.name = "projected_labels_tracked"

    # find out which object ids exist in this projected region
    def filter_nans(v):
        return v[~np.isnan(v)]

    id3d_tracked_from_projected = filter_nans(np.unique(projected_labels_tracked))

    objects_filtered = np.zeros_like(objects)

    print("Picking out objects which were tracked...")
    for object_id in tqdm.tqdm(id3d_tracked_from_projected):
        objects_filtered += objects.where(objects == object_id, other=0)

    return objects_filtered


def filter_objects_by_property(objects, da_property, op, value):
    N_objects = len(da_property.object_id)

    op_fn = getattr(np, op.replace("_than", ""))

    ids_filtered = da_property.where(op_fn(da_property, value), drop=True).object_id

    objects_filtered = np.zeros_like(objects)

    print(
        "Picking out objects for which {} is {} {} ({}/{}~{}%)...".format(
            da_property.name,
            op.replace("_", " "),
            value,
            len(ids_filtered),
            N_objects,
            int(float(len(ids_filtered)) / float(N_objects) * 100.0),
        )
    )

    for object_id in tqdm.tqdm(ids_filtered):
        objects_filtered += objects.where(objects == object_id, other=0)

    objects_filtered.attrs["input_name"] = objects.name
    objects_filtered.attrs["mask_name"] = "{}.filtered_by.{}_{}_{}".format(
        objects.mask_name, da_property.name, op, value
    )

    return objects_filtered


def filter_objects_by_id(objects, ids_to_keep):
    labels = np.asarray(objects)
    labels[~np.isin(labels, ids_to_keep)] = 0
    return labels


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument("object_file", type=str)

    subparsers = argparser.add_subparsers(dest="subparser_name")

    args_mask = subparsers.add_parser("mask")
    args_mask.add_argument("mask-name", type=str)
    args_mask.add_argument("--mask-field", default=None, type=str)

    args_tracking = subparsers.add_parser("tracking")
    args_tracking.add_argument("--dt-pad", default=20 * 60, type=float)

    args_property = subparsers.add_parser("property")
    args_property.add_argument("property", type=str)
    op_group = args_property.add_mutually_exclusive_group(required=True)
    op_group.add_argument("--less-than", type=float, metavar="value")
    op_group.add_argument("--equals", type=float, metavar="value")
    op_group.add_argument("--greater-than", type=float, metavar="value")

    args = argparser.parse_args()

    object_file = args.object_file.replace(".nc", "")

    if "objects" not in object_file:
        raise Exception()

    base_name, objects_mask = object_file.split(".objects.")

    fn_objects = "{}.nc".format(object_file)
    if not os.path.exists(fn_objects):
        raise Exception("Couldn't find objects file `{}`".format(fn_objects))
    objects = xr.open_dataarray(fn_objects, decode_times=False)

    if args.subparser_name == "mask":
        fn_mask = "{}.{}.mask.nc".format(base_name, args.mask_name)
        if not os.path.exists(fn_mask):
            raise Exception("Couldn't find mask file `{}`".format(fn_mask))

        if args.mask_field is None:
            mask_field = args.mask_name
        else:
            mask_field = args.mask_field
        mask_description = mask_field

        ds_mask = xr.open_dataset(fn_mask, decode_times=False)
        if mask_field not in ds_mask:
            raise Exception(
                "Can't find `{}` in mask, loaded mask file:\n{}"
                "".format(mask_field, str(ds_mask))
            )
        else:
            mask = ds_mask[mask_field]

        ds = filter_objects_by_mask(objects=objects, mask=mask)

        ds.attrs["input_name"] = object_file
        ds.attrs["mask_name"] = "{} && {}".format(ds.mask_name, mask_description)

        out_filename = "{}.objects.{}.{}.nc".format(
            base_name.replace("/", "__"), objects_mask, mask_description
        )
    elif args.subparser_name == "tracking":
        ds = filter_objects_by_tracking(
            objects=objects, base_name=base_name, dt_pad=args.dt_pad
        )
        ds.attrs["input_name"] = object_file
        ds.attrs["mask_name"] = "tracked__dt_pad{}s".format(args.dt_pad)

        out_filename = "{}.objects.{}.{}.nc".format(
            base_name.replace("/", "__"), objects_mask, "tracked"
        )
    elif args.subparser_name == "property":
        if args.less_than is not None:
            op = "less_than"
            value = args.less_than
        elif args.greater_than is not None:
            op = "greater_than"
            value = args.greater_than
        elif args.equals is not None:
            op = "equals"
            value = args.equals
        else:
            raise NotImplementedError

        base_name, objects_mask = object_file.split(".objects.")
        object_properties = genesis.objects.get_data(
            base_name, mask_identifier=objects_mask
        )
        da_property = object_properties[args.property]

        ds = filter_objects_by_property(
            objects=objects, da_property=da_property, op=op, value=value
        )

        out_filename = "{}.objects.{}.{}.{}_{}_{}.nc".format(
            base_name.replace("/", "__"),
            objects_mask,
            "filtered",
            args.property,
            op,
            value,
        )
    else:
        raise NotImplementedError

    ds.to_netcdf(out_filename)
    print("Wrote output to `{}`".format(out_filename))
