"""
Create a 3D mask from labelled objects
"""
import os

import xarray as xr


def create_mask_from_objects(objects):
    return objects != 0


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('object_file', type=str)

    args = argparser.parse_args()

    object_file = args.object_file.replace('.nc', '')

    if not 'objects' in object_file:
        raise Exception()

    base_name, mask_name = object_file.split('.objects.')

    fn_objects = "{}.nc".format(object_file)
    if not os.path.exists(fn_objects):
        raise Exception("Couldn't find objects file `{}`".format(fn_objects))
    objects = xr.open_dataarray(fn_objects, decode_times=False)


    ds = create_mask_from_objects(objects=objects)

    ds.attrs['input_name'] = object_file
    ds.attrs['mask_name'] = mask_name

    out_filename = "{}.mask_3d.objects.{}.nc".format(
        base_name.replace('/', '__'), mask_name
    )

    ds.to_netcdf(out_filename)
    print("Wrote output to `{}`".format(out_filename))
