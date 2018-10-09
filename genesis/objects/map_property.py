"""
Map a object property onto 3D object field
"""
import os
import warnings

import xarray as xr
import numpy as np
import tqdm

import genesis.objects


def map_property_onto_objects(objects, object_file, property):
    base_name, objects_mask = object_file.split('.objects.')

    object_properties = genesis.objects.get_data(base_name, mask_identifier=objects_mask)
    object_property = object_properties[property]
    N_objects = len(object_properties.object_id)

    properties_mapped = xr.full_like(objects, fill_value=np.nan, dtype=object_property.dtype)
    properties_mapped.attrs.update(object_property.attrs)
    properties_mapped.name = object_property.name

    print("Mapping {} onto {} objects...".format(property, N_objects))

    for object_id in tqdm.tqdm(object_properties.object_id):
        if object_id == 0:
            continue
        v = object_property.sel(object_id=object_id).values
        properties_mapped = properties_mapped.where(objects != object_id, other=v)

    return properties_mapped


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('object_file', type=str)
    argparser.add_argument('property', type=str)

    args = argparser.parse_args()

    object_file = args.object_file.replace('.nc', '')

    if not 'objects' in object_file:
        raise Exception()

    base_name, objects_mask = object_file.split('.objects.')

    fn_objects = "{}.nc".format(object_file)
    if not os.path.exists(fn_objects):
        raise Exception("Couldn't find objects file `{}`".format(fn_objects))
    objects = xr.open_dataarray(fn_objects, decode_times=False)


    ds = map_property_onto_objects(objects=objects, object_file=object_file,
                                    property=args.property)

    out_filename = "{}.objects.{}.mapped.{}.nc".format(
        base_name.replace('/', '__'), objects_mask,
        args.property
    )

    ds.to_netcdf(out_filename)
    print("Wrote output to `{}`".format(out_filename))
