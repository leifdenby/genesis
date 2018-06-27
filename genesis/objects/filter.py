"""
Using 2D mask filter out objects and create a new objects file
"""
import os
import warnings

import xarray as xr
import numpy as np

import cloud_identification


def label_objects(mask, splitting_scalar=None, remove_at_edge=True):
    def _remove_at_edge(object_labels):
        mask_edge = np.zeros_like(mask)
        mask_edge[:,:,1] = True  # has to be k==1 because object identification codes treats actual edges as ghost cells
        mask_edge[:,:,-2] = True
        cloud_identification.remove_intersecting(object_labels, mask_edge)

    if splitting_scalar is None:
        splitting_scalar = np.ones_like(mask)
    else:
        assert mask.shape == splitting_scalar.shape
        assert mask.dims == splitting_scalar.dims
        # NB: should check coord values too

    object_labels = cloud_identification.number_objects(
        splitting_scalar, mask=mask
    )

    if remove_at_edge:
        _remove_at_edge(object_labels)

    return object_labels


def filter_objects_by_mask(objects, mask):
    if mask.dims != objects.dims:
        if mask.dims == ('yt', 'xt'):
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


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(__doc__)

    argparser.add_argument('object_file', type=str)
    argparser.add_argument('--mask-name', default=None, type=str)
    argparser.add_argument('--mask-field', default=None, type=str)

    args = argparser.parse_args()

    object_file = args.object_file

    if not 'objects' in object_file:
        raise Exception()

    base_name, objects_mask = object_file.split('.objects.')

    fn_mask = "{}.{}.mask.nc".format(base_name, args.mask_name)
    if not os.path.exists(fn_mask):
        raise Exception("Couldn't find mask file `{}`".format(fn_mask))

    if args.mask_field is None:
        mask_field = args.mask_name
    else:
        mask_field = args.mask_field
    mask_description = mask_field

    ds_mask = xr.open_dataset(fn_mask, decode_times=False)
    if not mask_field in ds_mask:
        raise Exception("Can't find `{}` in mask, loaded mask file:\n{}"
                        "".format(mask_field, str(ds_mask)))
    else:
        mask = ds_mask[mask_field]

    fn_objects = "{}.nc".format(object_file)
    if not os.path.exists(fn_objects):
        raise Exception("Couldn't find objects file `{}`".format(fn_objects))
    objects = xr.open_dataarray(fn_objects, decode_times=False)

    ds = filter_objects_by_mask(objects=objects, mask=mask)

    ds.attrs['input_name'] = object_file
    ds.attrs['mask_name'] = "{} && {}".format(ds.mask_name, mask_description)

    out_filename = "{}.objects.{}.{}.nc".format(
        base_name.replace('/', '__'), objects_mask, mask_description
    )

    ds.to_netcdf(out_filename)
    print("Wrote output to `{}`".format(out_filename))
