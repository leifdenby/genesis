import xarray as xr
import numpy as np

FIELD_NAME_MAPPING = dict(
    w='WT',
    w_zt='WT',
    xt='W_E_direction',
    yt='S_N_direction',
    zt='vertical_levels',
    qv='RVT',
    qc='RCT',
    t='THT',
)

FIELD_DESCRIPTIONS = dict(
    w='vertical_velocity',
    qv='water vapour',
    qc='cloud liquid water',
    t='potential temperature',
)

UNITS_FORMAT = {
    'METERS/SECOND': 'm/s',
    'KELVIN': 'K',
    'KG/KG': 'kg/kg',
}

def _get_meso_nh_field(field_name):
    field_name_src = FIELD_NAME_MAPPING.get(field_name)

    if field_name_src is None:
        raise NotImplementedError("please define a mapping for the field `{}`"
                                  " in {}".format(field_name, __file__))

    return field_name_src

def _get_meso_nh_field_description(field_name):
    field_description = FIELD_DESCRIPTIONS.get(field_name)

    if field_description is None:
        raise NotImplementedError("please define a description for the field "
                                  "`{}` in {}".format(field_name, __file__))

    return field_description

def _scale_field(da):
    if da.units == 'km':
        da.values *= 1000.
        da.attrs['units'] = 'm'
    elif da.units == 'kg/kg':
        da.values *= 1000.
        da.attrs['units'] = 'g/kg'

    return da

def _cleanup_units(units):
    return UNITS_FORMAT.get(units, units)

def extract_field_to_filename(path_in, path_out, field_name):
    ds = xr.open_dataset(path_in)

    field_name_src = _get_meso_nh_field(field_name)

    da = ds[field_name_src]
    if field_name == 'w_zt':
        field_name = 'w'
    da.name = field_name

    new_coords = "xt yt".split(" ")
    old_coords = [_get_meso_nh_field(c) for c in new_coords]
    coord_map = dict(zip(old_coords, new_coords))
    da = da.rename(coord_map)
    da.attrs['longname'] = _get_meso_nh_field_description(field_name)
    da.attrs['long_name'] = _get_meso_nh_field_description(field_name)
    da.attrs['units'] = _cleanup_units(da.units)

    # all height levels should be the same, we do a quick check and then use
    # those values
    z__min = ds.VLEV.min(dim=old_coords)
    z__max = ds.VLEV.max(dim=old_coords)
    assert np.all(z__max - z__min < 1.0e-10)
    da['zt'] = np.round(z__min, decimals=3)
    da.zt.attrs['units'] = ds[old_coords[0]].units

    da = da.swap_dims(dict(vertical_levels='zt'))

    for c in "xt yt zt".split(" "):
        da.coords[c] = _scale_field(da[c])

    _scale_field(da)

    da.to_netcdf(path_out)
